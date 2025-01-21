import torch
import os
import torch.distributed as dist
from lightllm.server.pd_io_struct import KVMoveTask
from .deepseek2_mem_manager import Deepseek2MemoryManager
from typing import List
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans

logger = init_logger(__name__)


class Deepseek2FP8KVMemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy, mem_fraction)

    def get_cell_size(self):
        return self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(
            torch.float8_e4m3fn
        ) + self.head_num * self.layer_num * torch._utils._element_size(self.dtype)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.scale_buffer = torch.empty((layer_num, size, head_num, 1), dtype=dtype, device="cuda")
        self.kv_buffer = torch.empty((layer_num, size, head_num, head_dim), dtype=torch.float8_e4m3fn, device="cuda")
