import torch
import torch.distributed as dist
import rpyc
import time
from typing import Dict, List, Tuple, Optional, Union
from rpyc.utils.classic import obtain
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.common.basemodel.infer_lock import acquire_lock_until_ready, release_acquired_lock, g_router_lock
from .decode_task_cache import g_kv_move_task_cache, g_success_kv_move_task_cache
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDDecodeInferRpcServer(rpyc.Service):
    def __init__(self, backend: ContinuesBatchBackendForDecodeNode) -> None:
        super().__init__()
        self.backend = backend
        self.rank_id = self.backend.tp_rank
        return

    def on_connect(self, conn):
        torch.cuda.set_device(f"cuda:{self.rank_id}")
        return

    def judge_token_is_ok(self, key_len, max_new_token):
        if self.rank_id == 0:
            with g_router_lock.obj:
                shared_token_load = self.backend.shared_token_load
                peak_num = shared_token_load.get_estimated_peak_token_count()
                peak_num += shared_token_load.get_frozened_token_count()
                peak_num += key_len + max_new_token

                if peak_num < self.backend.get_max_total_token_num():
                    object_list = [True]
                    shared_token_load.add_frozened_token_count(key_len + max_new_token)
                else:
                    object_list = [False]
            dist.broadcast_object_list(object_list, src=0, group=self.backend.lock_nccl_group)
        else:
            object_list = [None]
            dist.broadcast_object_list(object_list, src=0, group=self.backend.lock_nccl_group)
        return object_list[0]

    def recover_frozen_token(self, key_len, max_new_token):
        if self.rank_id == 0:
            with g_router_lock.obj:
                shared_token_load = self.backend.shared_token_load
                shared_token_load.add_frozened_token_count(-(key_len + max_new_token))
        return

    # 返回 None 代表服务繁忙已经无法调度新的请求进入了
    def exposed_alloc_to_frozen_some_tokens(self, move_task: KVMoveTask) -> Optional[List[int]]:
        logger.info("exposed_alloc_to_frozen_some_tokens start")
        move_task = obtain(move_task)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        try:
            is_ok = self.judge_token_is_ok(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
            if not is_ok:
                return None

            key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
            tree_node, kv_len, fused_token_indexes = self.backend.radix_cache.match_prefix(key, update_refs=True)
            # 如果没匹配到，说明长度是0， 将fused_token_indexes做一下转换
            fused_token_indexes = [] if fused_token_indexes is None else fused_token_indexes.tolist()
            need_len = len(move_task.input_tokens) - kv_len
            if need_len == 0:
                alloc_token_indexes = []
            else:
                self.backend.radix_cache.free_radix_cache_to_get_enough_token(need_len)
                alloc_token_indexes = self.backend.model.mem_manager.alloc(need_len)
                if alloc_token_indexes is not None:
                    alloc_token_indexes = alloc_token_indexes.detach().cpu().tolist()

            if alloc_token_indexes is None:
                self.backend.radix_cache.dec_node_ref_counter(tree_node)
                self.recover_frozen_token(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
                return None

            move_task.decode_token_indexes = alloc_token_indexes
            move_task.move_kv_len = need_len

            g_kv_move_task_cache[move_task.group_request_id] = (move_task, tree_node, fused_token_indexes)
            return move_task.decode_token_indexes
        except BaseException as e:
            logger.exception(str(e))
            return -1
        finally:
            release_acquired_lock()
            logger.info("exposed_alloc_to_frozen_some_tokens end")

    def exposed_put_kv_received_to_radix_cache(self, group_req_id: int):
        group_req_id = obtain(group_req_id)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        radix_cache = self.backend.radix_cache
        key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
        value = torch.tensor(fused_token_indexes + move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        prefix_len = radix_cache.insert(key, value)
        assert len(fused_token_indexes) <= prefix_len
        self.backend.model.mem_manager.free(value[len(fused_token_indexes) : prefix_len].cuda())
        self.backend.radix_cache.dec_node_ref_counter(tree_node)

        # 申请一段key，把 radix cache 锁住，防止极端情况下被刷掉, decode 端通过减两次引用计数来修正。
        tree_node, kv_len, _ = self.backend.radix_cache.match_prefix(key, update_refs=True)
        assert len(key) == kv_len
        g_success_kv_move_task_cache[group_req_id] = (move_task, tree_node, time.time())
        release_acquired_lock()
        return

    def exposed_fail_to_realese_forzen_tokens(self, group_req_id: int):
        group_req_id = obtain(group_req_id)
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        value = torch.tensor(move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        self.backend.model.mem_manager.free(value.cuda())
        self.backend.radix_cache.dec_node_ref_counter(tree_node)
        self.recover_frozen_token(len(move_task.input_tokens), move_task.decode_node.max_new_tokens)
        release_acquired_lock()
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to info_queues ok")
        return

    def exposed_unfrozen_time_out_reqs_tokens(self):
        acquire_lock_until_ready(self.backend.lock_nccl_group)
        if self.rank_id == 0:
            need_release_reqs = []
            for req_id, (task, tree_node, time_mark) in g_success_kv_move_task_cache.items():
                # 4s 这个请求都没有被调度使用，就会主动被删除掉锁定，释放其锁定的token
                if time.time() - time_mark > 4:
                    need_release_reqs.append(req_id)
            logger.info(f"kv time out reqs: {need_release_reqs}")
            dist.broadcast_object_list([need_release_reqs], src=0, group=self.backend.lock_nccl_group)
        else:
            receive_objs = [None]
            dist.broadcast_object_list(receive_objs, src=0, group=self.backend.lock_nccl_group)
            need_release_reqs = receive_objs[0]

        remove_tokens = 0
        for req_id in need_release_reqs:
            task, tree_node, _ = g_success_kv_move_task_cache.pop(req_id)
            self.backend.radix_cache.dec_node_ref_counter(tree_node)
            remove_tokens += len(task.input_tokens) + task.decode_node.max_new_tokens

        if self.rank_id == 0 and remove_tokens != 0:
            with g_router_lock.obj:
                self.backend.shared_token_load.add_frozened_token_count(-remove_tokens)

        release_acquired_lock()
        return
