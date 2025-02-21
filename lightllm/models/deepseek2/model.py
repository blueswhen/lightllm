import torch
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.common.deepseek2_fp8kv_mem_manager import Deepseek2FP8KVMemoryManager
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale


logger = init_logger(__name__)


class Deepseek2TpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Deepseek2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek2TransformerLayerInfer

    # infer state class
    infer_state_class = Deepseek2InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_some_value(self):
        super()._init_some_value()
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 0

        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self.head_dim_ = self.kv_lora_rank + self.qk_rope_head_dim
        self.tp_q_head_num_ = self.config["num_attention_heads"] // self.world_size_
        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        if self.config["rope_scaling"] is not None:
            rope_scaling = self.config["rope_scaling"]
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_custom(self):
        self._init_to_get_yarn_rotary()

    def _verify_params(self):
        return super()._verify_params()

    def _init_mem_manager(self):
        manager_class = Deepseek2MemoryManager
        if "triton_fp8kv" in self.mode:
            manager_class = Deepseek2FP8KVMemoryManager
        self.mem_manager = manager_class(
            self.max_total_token_num,
            dtype=self.data_type,
            head_num=1,
            head_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
            layer_num=self.config["num_hidden_layers"],
            mem_fraction=self.mem_fraction,
        )
        return

    def _init_weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(
            self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode
        )
        self.trans_layers_weight = [
            self.transformer_weight_class(
                i,
                self.tp_rank_,
                self.world_size_,
                self.data_type,
                network_config=self.config,
                mode=self.mode,
                quant_cfg=self.quant_cfg,
            )
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict,
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]
        return

    def _init_infer_layer(self):
        self.pre_infer = self.pre_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.post_infer = self.post_layer_infer_class(
            tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config, mode=self.mode
        )
        self.layers_infer = [
            self.transformer_layer_infer_class(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=self.mode,
            )
            for i in range(self.config["n_layer"])
        ]
        return

    def _init_to_get_yarn_rotary(self):
        from lightllm.models.llama.yarn_rotary_utils import find_correction_range, linear_ramp_mask, get_deepseek_mscale

        dim = self.qk_rope_head_dim
        max_position_embeddings = self.config.get("max_position_embeddings", 2048)
        base = self.config.get("rope_theta", 10000.0)
        if self.config.get("rope_scaling", {}) is None:
            scale = 1.0
        else:
            rope_scaling = self.config.get("rope_scaling", {})
            scale = rope_scaling.get("factor", 1.0)
            mscale = rope_scaling.get("mscale", 1)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
        extrapolation_factor = 1.0
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)

        pos_freqs = base ** (torch.arange(0, dim, 2).float().cuda() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, dim // 2).float().cuda()
        ) * extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        _mscale = float(
            get_deepseek_mscale(scale, mscale) / get_deepseek_mscale(scale, mscale_all_dim)
        )  # Get n-d magnitude scaling corrected for interpolation

        # Build here to make `torch.jit.trace` work.
        max_seq_len_cached = max_position_embeddings
        t = torch.arange(max_seq_len_cached, device="cuda", dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self._cos_cached = (freqs.cos() * _mscale).to(self.data_type).cuda()
        self._sin_cached = (freqs.sin() * _mscale).to(self.data_type).cuda()

        return
