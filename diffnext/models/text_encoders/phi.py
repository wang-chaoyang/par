# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
# Modified in 2025 by Chaoyang Wang

from typing import Tuple

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    from einops import rearrange, repeat

    apply_rotary_emb = None


import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi.configuration_phi import PhiConfig


def rotate_half(x, interleaved=False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False, inplace=False) -> torch.Tensor:
    ro_dim = cos.shape[-1] * 2
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


apply_rotary_emb = apply_rotary_emb or apply_rotary_emb_torch


class PhiCache(nn.Module):
    """Execution cache."""

    def __init__(self, config: PhiConfig, device=None, dtype=None):
        super(PhiCache, self).__init__()
        self.config, self.device, self.dtype = config, device, dtype
        self.start_pos, self.end_pos, self.cache_dict = 0, 0, {}

    def reset(self, device=None, dtype=None):
        self.device, self.dtype = device, dtype
        max_seq_len = self.config.max_position_embeddings
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rotary_dim = int(self.config.partial_rotary_factor * head_dim)
        self.init_rotary(max_seq_len, rotary_dim, self.config.rope_theta)

    def init_rotary(self, seq_len, dim, theta=10000.0):
        grid = torch.arange(seq_len, dtype=torch.float32).unsqueeze_(-1)
        freq = torch.pow(theta, torch.arange(0, dim, 2)[: dim // 2].float().div_(dim))
        broadcast_freq = grid.mul(freq.reciprocal_().unsqueeze_(0))
        cache_cos = broadcast_freq.cos().view((-1, dim // 2))
        cache_sin = broadcast_freq.sin().view((-1, dim // 2))
        self.cache_dict["cos"] = cache_cos.to(self.device, self.dtype)
        self.cache_dict["sin"] = cache_sin.to(self.device, self.dtype)

    def set_seq(self, start_pos=0, end_pos=None):
        self.start_pos, self.end_pos = start_pos, end_pos
        if "cos" in self.cache_dict and end_pos is not None:
            self.cache_dict["seq_cos"] = self.cache_dict["cos"][self.start_pos : end_pos]
            self.cache_dict["seq_sin"] = self.cache_dict["sin"][self.start_pos : end_pos]

    def forward_rotary(self, q, k, inplace=False) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cache_dict.get("seq_cos", self.cache_dict.get("cos", None))
        sin = self.cache_dict.get("seq_sin", self.cache_dict.get("sin", None))
        q = apply_rotary_emb(q, cos, sin, interleaved=False, inplace=inplace)
        k = apply_rotary_emb(k, cos, sin, interleaved=False, inplace=inplace)
        return q, k


class PhiMLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.activation = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class PhiAttention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.dense = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)


class PhiFlashAttention2(PhiAttention):
    """Multi-headed attention using FA2."""

    def forward(self, x, attn_mask=None) -> torch.Tensor:
        qkv_shape = (-1, x.shape[1], self.num_heads, self.head_dim)
        q, k, v = [f(x).view(qkv_shape) for f in (self.q_proj, self.k_proj, self.v_proj)]
        q, k = self.cache.forward_rotary(q, k, inplace=True)
        if flash_attn_func is None:
            q, k, v = [_.transpose(1, 2) for _ in (q, k, v)]
            o = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.dense(o.transpose(1, 2).flatten(2))
        return self.dense(flash_attn_func(q, k, v, causal=True).flatten(2))


class PhiLayer(nn.Module):
    """Transformer layer."""

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.self_attn = PhiFlashAttention2(config)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attn_mask=None) -> torch.Tensor:
        shortcut, x = x, self.input_layernorm(x)
        return self.self_attn(x, attn_mask).add_(self.mlp(x)).add_(shortcut)


class PhiPreTrainedModel(PreTrainedModel):
    """Base model."""

    config_class = PhiConfig


class PhiModel(PhiPreTrainedModel):
    """Standard model."""

    def __init__(self, config: PhiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(PhiLayer(config) for _ in range(config.num_hidden_layers))
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cache, _ = PhiCache(config), self.post_init()

    def maybe_init_cache(self, **kwargs):
        if self.cache.device is not None:
            return
        self.cache.reset(self.device, self.dtype)
        [layer.self_attn.__dict__.setdefault("cache", self.cache) for layer in self.layers]

    def forward(self, input_ids, attention_mask=None, **kwargs) -> BaseModelOutput:
        self.maybe_init_cache(**kwargs)
        h = kwargs.get("inputs_embeds", None)
        h = self.embed_tokens(input_ids) if h is None else h
        start_pos = 0 if kwargs.get("past_key_values", None) is None else self.cache.end_pos
        self.cache.set_seq(start_pos, start_pos + h.shape[1])
        for layer in self.layers:
            h = layer(h, attention_mask)
        h = self.final_layernorm(h)
        return BaseModelOutput(last_hidden_state=h)


class PhiEncoderModel(PhiPreTrainedModel):
    """Encoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = PhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.vocab_size, _ = config.vocab_size, self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs) -> BaseModelOutput:
        return self.model(input_ids, attention_mask, **kwargs)
