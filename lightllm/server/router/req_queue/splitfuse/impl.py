import uuid
import numpy as np
from typing import List
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Batch, Req
from lightllm.server.io_struct import ReqRunStatus
from lightllm.server.router.req_queue.base_queue import BaseQueue


class SplitFuseQueue(BaseQueue):
    def __init__(self, args, router) -> None:
        super().__init__(args, router)

    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy, new_batch_first_router_need_tokens):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len))  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        ok_token_num = (
            need_max_token_num + self.router.shared_token_load.get_frozened_token_count() < self.max_total_tokens
        )

        if req.req_status != ReqRunStatus.PAUSED_AND_OFFLOAD:
            ok_req_num = len(self.cache_len_list) + len(self.pause_req_dict) <= self.running_max_req_size
        else:
            ok_req_num = len(self.cache_len_list) + len(self.pause_req_dict) - 1 <= self.running_max_req_size

        new_batch_first_router_need_tokens += req.get_first_router_need_tokens()
        # splitfuse decode ok
        ok_splitfuse_decode = new_batch_first_router_need_tokens <= self.batch_max_tokens

        if ok_token_num and ok_req_num and ok_splitfuse_decode:
            self.router.shared_token_load.set_estimated_peak_token_count(need_max_token_num)
            self.router.shared_token_load.set_dynamic_max_load(
                (need_max_token_num + self.router.shared_token_load.get_frozened_token_count()) / self.max_total_tokens
            )
            return True, new_batch_first_router_need_tokens
        else:
            return False, new_batch_first_router_need_tokens

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = (0 if current_batch is None else len(current_batch.reqs)) + len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        is_busy = self.is_busy()

        # 得到当前batch 往前 decode 一次，需要的token量，在 splitfuse 模式下才有用，因为splitfuse
        # 模式下 类似prefill 和 deocde 是在一起进行的，所以需要合并考虑历史当前Batch
        new_batch_first_router_need_tokens = 0 if current_batch is None else current_batch.batch_decode_need_tokens

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE:
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉.
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                continue
            ok_insert, new_batch_first_router_need_tokens = self._can_add_new_req(
                req, is_busy, new_batch_first_router_need_tokens
            )
            if ok_insert:
                can_run_list.append(req)
                if req.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count :]
            return new_batch
        else:
            return None

    def _calcu_batch_token_load_batch_not_none(self, current_batch: Batch):
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        self.cache_len_list.sort(key=lambda x: -x[1])
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return (
            need_max_token_num,
            (need_max_token_num + self.router.shared_token_load.get_frozened_token_count()) / self.max_total_tokens,
        )
