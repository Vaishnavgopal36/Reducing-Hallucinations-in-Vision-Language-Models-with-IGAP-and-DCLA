"""
PyTorch attention patching utilities for Image-Guided Attention Pruning (IGAP).

This module patches LLaVA's LLaMA self-attention to optionally suppress heads
that place low mass on image-token spans during decode-time (q_len == 1).
"""

from __future__ import annotations

import functools
import math
import types
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

_igap_toggle: dict[str, bool] = {"active": False}

igap_debug: dict[str, float | int] = {
    "forward_calls": 0,
    "igap_active_calls": 0,
    "q_len1_calls": 0,
    "suppressed_fraction_sum": 0.0,
}


def _repeat_kv_for_full_heads(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV heads to match full attention head count."""
    if num_heads == num_key_value_heads:
        return key_states, value_states

    groups = num_heads // num_key_value_heads
    if groups <= 1:
        return key_states, value_states

    key_states = key_states.repeat_interleave(groups, dim=1)
    value_states = value_states.repeat_interleave(groups, dim=1)
    return key_states, value_states


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor],
    attn_logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Normalize 2D/4D masks into additive-mask form compatible with logits."""
    if attention_mask is None:
        return None

    if attention_mask.dim() == 2:
        # Binary mask [B, K] -> additive mask [B, 1, 1, K]
        mask = attention_mask.to(dtype=attn_logits.dtype)
        neg_inf = torch.finfo(attn_logits.dtype).min
        return (1.0 - mask[:, None, None, :]) * neg_inf

    if attention_mask.dim() == 4:
        return attention_mask.to(dtype=attn_logits.dtype)

    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")


def llama_igap_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
    """Patched ``LlamaAttention.forward`` with IGAP head suppression.

    IGAP activates only when:
    1. ``self.use_igap_img`` is True,
    2. ``_igap_toggle[\"active\"]`` is True,
    3. ``q_len == 1`` (decode-step query token).
    """
    del use_cache  # maintained for signature compatibility

    igap_debug["forward_calls"] += 1
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None and hasattr(past_key_value, "get_usable_length"):
        if self.layer_idx is None:
            raise ValueError("self.layer_idx is required when cache is active.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    position_embeddings = kwargs.get("position_embeddings")
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        try:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        except TypeError:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        position_ids,
    )

    # Support both new cache objects and legacy tuple caches.
    if past_key_value is not None:
        if hasattr(past_key_value, "update"):
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
        elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
            prev_k, prev_v = past_key_value
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)
            past_key_value = (key_states, value_states)

    key_states, value_states = _repeat_kv_for_full_heads(
        key_states,
        value_states,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_key_value_heads,
    )

    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_logits = attn_logits / math.sqrt(self.head_dim)

    norm_mask = _normalize_attention_mask(attention_mask, attn_logits)
    if norm_mask is not None:
        if norm_mask.shape[-1] != attn_logits.shape[-1]:
            norm_mask = norm_mask[..., : attn_logits.shape[-1]]
        if norm_mask.shape[-2] not in (1, attn_logits.shape[-2]):
            norm_mask = norm_mask[..., : attn_logits.shape[-2], :]
        attn_logits = attn_logits + norm_mask

    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if getattr(self, "use_igap_img", False) and _igap_toggle["active"] and q_len == 1:
        igap_debug["igap_active_calls"] += 1
        igap_debug["q_len1_calls"] += 1

        img_start = max(0, min(int(getattr(self, "img_start_idx", 0)), attn_probs.shape[-1]))
        img_end = max(img_start, min(int(getattr(self, "img_end_idx", img_start)), attn_probs.shape[-1]))
        keep_head_ratio = float(getattr(self, "keep_head_ratio", 1.0))
        suppression_alpha = float(getattr(self, "suppression_alpha", 1.0))

        head_mask_2d = torch.ones(
            (bsz, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        if img_end > img_start:
            num_keep = max(1, min(self.num_heads, int(round(keep_head_ratio * self.num_heads))))
            head_scores = attn_probs[:, :, -1, img_start:img_end].sum(dim=-1)
            _, keep_idx = torch.topk(head_scores, k=num_keep, dim=1)

            head_mask_2d.fill_(suppression_alpha)
            head_mask_2d.scatter_(1, keep_idx, 1.0)

        igap_debug["suppressed_fraction_sum"] += float((head_mask_2d < 1.0).float().mean().item())
        head_mask = head_mask_2d[:, None, :].expand(-1, q_len, -1)
    else:
        head_mask = torch.ones(
            (bsz, q_len, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )

    attn_output = torch.matmul(attn_probs, value_states)  # [B, H, Q, D]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q, H, D]
    attn_output = attn_output * head_mask.unsqueeze(-1)
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs = None

    return attn_output, attn_probs, past_key_value


def get_llama_layers(llava_model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the decoder layer list for a LLaVA model with LLaMA backbone."""
    # Typical LLaVA HF path: model.language_model.model.layers
    candidates = [
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    ]

    for path in candidates:
        current: Any = llava_model
        valid = True
        for attr in path:
            if not hasattr(current, attr):
                valid = False
                break
            current = getattr(current, attr)
        if valid and isinstance(current, torch.nn.ModuleList):
            return current

    raise AttributeError("Could not locate LLaMA decoder layers in the provided model.")


def apply_igap_to_llava(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    img_start_idx: int,
    img_end_idx: int,
    keep_head_ratio: float = 0.95,
    suppression_alpha: float = 0.08,
    use_igap_img: bool = True,
) -> None:
    """Patch LLaVA self-attention layers in ``[start_layer, end_layer)`` with IGAP."""
    layers = get_llama_layers(model)
    start_layer = max(0, int(start_layer))
    end_layer = min(int(end_layer), len(layers))

    for i in range(start_layer, end_layer):
        self_attn = layers[i].self_attn
        self_attn.img_start_idx = int(img_start_idx)
        self_attn.img_end_idx = int(img_end_idx)
        self_attn.keep_head_ratio = float(keep_head_ratio)
        self_attn.suppression_alpha = float(suppression_alpha)
        self_attn.use_igap_img = bool(use_igap_img)

        if isinstance(self_attn.forward, functools.partial):
            self_attn.forward = self_attn.forward.func

        self_attn.forward = types.MethodType(llama_igap_forward, self_attn)

    print(
        f"IGAP patched layers [{start_layer}, {end_layer}) | "
        f"keep_head_ratio={keep_head_ratio}, suppression_alpha={suppression_alpha}"
    )
