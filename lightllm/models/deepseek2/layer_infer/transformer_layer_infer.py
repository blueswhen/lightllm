from typing import Tuple
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_with_v import (
    context_attention_fwd_with_v,
    context_attention_fwd_no_prompt_cache_with_v,
)

from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from functools import partial
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
import os
from lightllm.common.quantization import vLLMFP8w8a8QuantizationMethod


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(
        self, layer_num, tp_rank, world_size, network_config, mode=[], disable_qk_absorb=False, disable_vo_absorb=False
    ):
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.q_lora_rank = network_config["q_lora_rank"]
        self.kv_lora_rank = network_config["kv_lora_rank"]

        self.n_routed_experts = network_config["n_routed_experts"]

        self.is_moe = (
            network_config["n_routed_experts"] is not None
            and layer_num >= network_config["first_k_dense_replace"]
            and layer_num % network_config["moe_layer_freq"] == 0
        )
        self.disable_qk_absorb = disable_qk_absorb
        self.disable_vo_absorb = disable_vo_absorb

        self.n_shared_experts = network_config["n_shared_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.n_group = network_config["n_group"]
        self.topk_group = network_config["topk_group"]
        self.routed_scaling_factor = network_config["routed_scaling_factor"]

        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        if network_config.get("rope_scaling", None) is not None:
            self.rope_scaling = network_config["rope_scaling"]
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_o_head_num_ = self.tp_q_head_num_

        self.num_heads = network_config["num_attention_heads"]
        self.num_kv_heads = network_config["num_key_value_heads"]
        self.enable_opt_decoding_mha = os.getenv("ENABLE_OPT_DECODE_MHA", "False").upper() in ["ON", "TRUE", "1"]
        self.mla_type = "ACCM"

        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(Deepseek2TransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(
            Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashdecoding, self
        )
        self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        if self.is_moe:
            if os.environ.get("ETP_MODE_ENABLED") == "true":
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_etp, self)
            else:
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv,
        infer_state: Deepseek2InferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if not self.disable_qk_absorb:
            if self.q_lora_rank is None:
                q_nope = layer_weight.fuse_qk_weight_.mm(input)
                q_rope = layer_weight.q_rope_proj_.mm(input)
            else:
                q = layer_weight.q_a_proj_.mm(input)
                rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_, out=q)
                q_nope = layer_weight.fuse_qk_weight_.mm(q)
                q_rope = layer_weight.q_rope_proj_.mm(q)
            q_nope = q_nope.view(-1, self.tp_q_head_num_, self.kv_lora_rank)
            q_rope = q_rope.view(-1, self.tp_q_head_num_, self.qk_rope_head_dim)
        else:
            if self.q_lora_rank is None:
                q = layer_weight.q_weight_.mm(input)
            else:
                q = layer_weight.q_a_proj_.mm(input)
                rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_, out=q)
                q = layer_weight.q_b_proj_.mm(q)

            q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
            q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            if infer_state.use_dynamic_prompt_cache and infer_state.is_prefill:
                self.mla_type = "ACCM"
            else:
                self.mla_type = layer_weight.mla_type
            if self.mla_type == "ACCM":
                q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        layer_weight.kv_a_proj_with_mqa_.mm(input, out=cache_kv.view(-1, self.kv_lora_rank + self.qk_rope_head_dim))

        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return (q_nope, q_rope), cache_kv

    def _get_o(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if not self.disable_vo_absorb:
            input = input.view(-1, self.tp_q_head_num_ * self.kv_lora_rank)
            o_tensor = layer_weight.fuse_vo_weight_.mm(input)
        else:
            if self.mla_type == "ACCM":
                input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
            o_tensor = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.qk_nope_head_dim))
        return o_tensor

    def _CC_method(
        self, q, compressed_kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ):
        num_local_heads = self.num_heads
        num_local_kv_heads = self.num_kv_heads
        if self.world_size_ > 1:
            num_local_heads //= self.world_size_
            num_local_kv_heads //= self.world_size_
        if infer_state.use_dynamic_prompt_cache:
            compressed_kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        # CC
        compressed_kv, k_pe = torch.split(  # (b*s, 1, kv_lora + qk_r)
            compressed_kv, [layer_weight.kv_lora_rank, layer_weight.qk_rope_head_dim], dim=-1
        )
        compressed_kv = compressed_kv.view(-1, layer_weight.kv_lora_rank)
        k_nope = self.alloc_tensor(
            [compressed_kv.shape[0], num_local_kv_heads, layer_weight.qk_nope_head_dim],
            dtype=q[0].dtype,
        )
        wk = layer_weight.k_b_proj_.weight.view(-1, layer_weight.k_b_proj_.weight.shape[-1])
        torch.mm(compressed_kv, wk.transpose(0, 1), out=k_nope.reshape(k_pe.shape[0], -1))

        trans_weight = layer_weight.v_b_proj_.weight.transpose(1, 2)
        wv = trans_weight.view(-1, trans_weight.shape[-1])
        v = self.alloc_tensor(
            [compressed_kv.shape[0], num_local_kv_heads, layer_weight.qk_nope_head_dim],
            dtype=q[0].dtype,
        )
        torch.mm(compressed_kv, wv.transpose(0, 1), out=v.reshape(compressed_kv.shape[0], -1))
        return self._context_attention_kernel_with_v(q, [k_nope, k_pe], v, infer_state, layer_weight)

    def _ACC_method(
        self, q, compressed_kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ):
        q_nope, q_rope = q
        num_local_heads = self.num_heads
        num_local_kv_heads = self.num_kv_heads
        if self.world_size_ > 1:
            num_local_heads //= self.world_size_
            num_local_kv_heads //= self.world_size_
        # ACC
        q_nope = layer_weight.k_b_proj_.bmm(
            q_nope.transpose(0, 1),
        ).transpose(0, 1)
        if self.enable_opt_decoding_mha:
            import lightllm_ppl_mla

            o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype)
            q = torch.cat([q_nope, q_rope], dim=-1)
            lightllm_ppl_mla.decode_mla(
                o_tensor,
                q,
                compressed_kv,
                infer_state.req_manager.req_to_token_indexs,
                infer_state.kv_starts,
                infer_state.b_req_idx,
                self.softmax_scale,
                q.shape[-1],
                q_nope.shape[-1],
            )
            output_parallel = o_tensor
        else:
            output_parallel = self._token_gqa_decode_attention_flashdecoding_origin(
                (q_nope, q_rope), infer_state, layer_weight
            )
        vo = layer_weight.v_b_proj_.bmm(output_parallel.transpose(0, 1)).transpose(0, 1)
        return vo

    def _context_attention_kernel(
        self, q, kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ) -> torch.Tensor:
        if self.mla_type == "MIX":
            return self._context_attention_kernel_with_CC(q, kv, infer_state, layer_weight, out)
        else:
            return self._context_attention_kernel_origin(q, kv, infer_state, layer_weight, out)

    def _context_attention_kernel_with_CC(
        self, q, kv, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ) -> torch.Tensor:
        return self._CC_method(q, kv, infer_state, layer_weight)

    def _context_attention_kernel_with_v(
        self, q: Tuple[torch.Tensor, torch.Tensor], k, v, infer_state: Deepseek2InferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        k_nope, k_rope = k
        nope_head_dim = q_nope.shape[-1]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            context_attention_fwd_with_v(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                v,
                o_tensor.view(-1, self.tp_q_head_num_, nope_head_dim),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.softmax_scale,
            )
        else:
            context_attention_fwd_no_prompt_cache_with_v(
                q_nope,
                q_rope,
                k_nope,
                k_rope,
                v,
                o_tensor.view(-1, self.tp_q_head_num_, nope_head_dim),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )
        q_nope = None
        q_rope = None
        return o_tensor

    def _context_attention_kernel_origin(
        self, q: Tuple[torch.Tensor, torch.Tensor], kv, infer_state: Deepseek2InferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out

        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.softmax_scale,
            )
        else:
            context_attention_fwd_no_prompt_cache(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )
        q_nope = None
        q_rope = None
        return o_tensor

    def _token_gqa_decode_attention_flashdecoding(
        self, q, infer_state: Deepseek2InferStateInfo, layer_weight, out=None
    ):
        if self.mla_type == "MIX":
            return self._token_gqa_decode_attention_flashdecoding_with_ACC(q, infer_state, layer_weight, out)
        else:
            return self._token_gqa_decode_attention_flashdecoding_origin(q, infer_state, layer_weight, out)

    def _token_gqa_decode_attention_flashdecoding_with_ACC(
        self, q, infer_state: Deepseek2InferStateInfo, layer_weight, out=None
    ):
        compressed_kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        return self._ACC_method(q, compressed_kv, infer_state, layer_weight)

    def _token_gqa_decode_attention_flashdecoding_origin(
        self, q, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank]
        kv_rope = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :]
        # quant_method = vLLMFP8w8a8QuantizationMethod()
        # q_nope_quant = quant_method.quantize(q_nope.reshape(-1, q_nope.shape[-1]), False)
        # q_rope_quant = quant_method.quantize(q_rope.reshape(-1, q_rope.shape[-1]), False)
        # kv_quant = quant_method.quantize(kv.reshape(-1, kv.shape[-1]), False)
        # kv_rope_quant = quant_method.quantize(kv_rope.reshape(-1, kv_rope.shape[-1]), False)
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device) if out is None else out
        gqa_token_decode_attention_flash_decoding(
            q_nope,
            q_rope,
            kv,
            kv_rope,
            infer_state,
            self.tp_q_head_num_,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.qk_nope_head_dim,
            self.softmax_scale,
            o_tensor,
            use_fp8_w8a8=False,
        )
        return o_tensor

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(
            buffer[:, :, : self.kv_lora_rank],
            buffer[:, :, self.kv_lora_rank :],
            mem_index,
            mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank],
            mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :],
        )
        return

    def _moe_ffn(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        hidden_states.mul_(self.routed_scaling_factor)

        if self.n_shared_experts is not None:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _splitfuse_attention_kernel(
        self, q, infer_state: SplitFuseInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        if self.mla_type == "MIX":
            return self._splitfuse_attention_kernel_with_CC(q, infer_state, layer_weight, out)
        else:
            return self._splitfuse_attention_kernel_origin(q, infer_state, layer_weight, out)

    def _splitfuse_attention_kernel_origin(
        self, q, infer_state: SplitFuseInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        infer_state.start_event.record(torch.cuda.default_stream())
        if infer_state.decode_req_num > 0:
            o_tensor[0 : infer_state.decode_req_num, :] = self._token_gqa_decode_attention_flashdecoding_origin(
                (q_nope[0 : infer_state.decode_req_num, :], q_rope[0 : infer_state.decode_req_num, :]),
                infer_state.inner_decode_infer_status,
                layer_weight,
            )
        if infer_state.prefill_req_num > 0:
            infer_state.parrall_stream.wait_event(infer_state.start_event)
            with torch.cuda.stream(infer_state.parrall_stream):
                self._context_attention_kernel_origin(
                    (q_nope[infer_state.decode_req_num :, :], q_rope[infer_state.decode_req_num :, :]),
                    infer_state.mem_manager.kv_buffer[self.layer_num_],
                    infer_state.inner_prefill_infer_status,
                    layer_weight,
                    out=o_tensor[infer_state.decode_req_num :, :],
                )
            infer_state.end_event.record(infer_state.parrall_stream)
            torch.cuda.default_stream().wait_event(infer_state.end_event)
        return o_tensor

    def _splitfuse_attention_kernel_with_CC(
        self, q, infer_state: SplitFuseInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        infer_state.start_event.record(torch.cuda.default_stream())
        if infer_state.decode_req_num > 0:
            o_tensor[0 : infer_state.decode_req_num, :] = self._token_gqa_decode_attention_flashdecoding_with_ACC(
                (q_nope[0 : infer_state.decode_req_num, :], q_rope[0 : infer_state.decode_req_num, :]),
                infer_state.inner_decode_infer_status,
                layer_weight,
            )
        if infer_state.prefill_req_num > 0:
            infer_state.parrall_stream.wait_event(infer_state.start_event)
            with torch.cuda.stream(infer_state.parrall_stream):
                o_tensor[infer_state.decode_req_num :, :] = self._context_attention_kernel_with_CC(
                    (q_nope[infer_state.decode_req_num :, :], q_rope[infer_state.decode_req_num :, :]),
                    infer_state.mem_manager.kv_buffer[self.layer_num_],
                    infer_state.inner_prefill_infer_status,
                    layer_weight,
                )
            infer_state.end_event.record(infer_state.parrall_stream)
            torch.cuda.default_stream().wait_event(infer_state.end_event)
        return o_tensor

    def _moe_ffn_etp(
        self, input, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        world_size_ = self.world_size_
        # num_local_experts = self.n_shared_experts // world_size_
        # local_expert_offset = self.tp_rank_ * num_local_experts
        num_experts_per_token = self.num_experts_per_tok
        num_experts = self.n_routed_experts
        # num_expert_groups = self.n_group
        # num_groups_per_token = self.topk_group
        gating_scaling_factor = self.routed_scaling_factor
        # gating_normalize_prob = self.norm_topk_prob
        rank_self = self.tp_rank_

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        final_hidden_states = torch.empty(
            num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # router_logits_len = hidden_states.shape[0]*layer_weight.moe_gate.shape[1]
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # now some parameter is not supported yet
        # assert gating_normalize_prob is False
        # assert num_expert_groups<=1

        import lightllm_moe_etp_kernel

        lightllm_moe_etp_kernel.moe_fused_all(
            router_logits.contiguous(),
            hidden_states.contiguous(),
            layer_weight.gate_up_proj.weight.contiguous(),  # transpose
            layer_weight.down_proj.weight.contiguous(),  # transpose
            layer_weight.experts.expert_gate_up_proj_etp.contiguous(),
            layer_weight.experts.expert_down_proj_etp.contiguous(),
            infer_state.mem_manager.work_buffer.contiguous(),
            infer_state.mem_manager.work_buffer.nelement(),
            final_hidden_states.contiguous(),
            rank_self,
            gating_scaling_factor,
            num_experts,
            num_experts_per_token,
            num_tokens,
            world_size_,
            hidden_dim,
            layer_weight.gate_up_proj.weight.size(1) // 2,
            layer_weight.experts.expert_gate_up_proj_etp.size(1) // 2,
            self.n_shared_experts is not None,
        )

        router_logits = None

        return final_hidden_states.view(num_tokens, hidden_dim)
