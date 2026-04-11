"""
src/model/igap_patch.py
=======================
Image-Guided Attention Pruning (IGAP) — attention patching for LLaVA-1.5 (LLaMA backbone).

_igap_toggle["active"] must be set True only during pass-2 of DCLA decoding.
"""

from __future__ import annotations

import functools
import math
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

_igap_toggle: dict[str, bool] = {"active": False}

igap_debug: dict[str, float | int] = {
    "forward_calls": 0,
    "igap_active_calls": 0,
    "q_len1_calls": 0,
    "avg_suppressed_fraction_sum": 0.0,
}


def llama_igap_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
    """Patched LlamaAttention forward with IGAP head suppression.

    IGAP activates only when q_len == 1 (decode step) and _igap_toggle["active"] is True.
    Top keep_head_ratio of heads (ranked by image-token attention mass) are kept at 1.0;
    the rest are scaled by suppression_alpha.
    """
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
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("layer_idx is required on the attention layer when use_cache=True.")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        try:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        except TypeError:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    num_kv_groups = self.num_heads // self.num_key_value_heads
    if num_kv_groups > 1:
        key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_logits = attn_logits + attention_mask
        attn_logits = torch.maximum(
            attn_logits,
            attn_logits.new_full((), torch.finfo(attn_logits.dtype).min),
        )

    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if (
        getattr(self, "use_igap_img", False)
        and q_len == 1
        and _igap_toggle["active"]
    ):
        igap_debug["igap_active_calls"] += 1
        igap_debug["q_len1_calls"] += 1

        num_keep: int = max(1, int(round(float(self.keep_head_ratio) * self.num_heads)))
        img_start: int = max(0, min(int(self.img_start_idx), attn_probs.shape[-1]))
        img_end: int = max(img_start, min(int(self.img_end_idx), attn_probs.shape[-1]))

        head_scores = attn_probs[:, :, -1, img_start:img_end].sum(dim=-1)
        _, keep_idx = torch.topk(head_scores, k=num_keep, dim=1)

        head_mask = torch.full(
            (bsz, self.num_heads),
            fill_value=float(self.suppression_alpha),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        head_mask.scatter_(1, keep_idx, 1.0)
        igap_debug["avg_suppressed_fraction_sum"] += (head_mask != 1.0).float().mean().item()
        head_mask = head_mask.view(bsz, 1, self.num_heads)
    else:
        head_mask = torch.ones(
            (bsz, q_len, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )

    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = torch.einsum("bqh,bqhd->bqhd", head_mask, attn_output)
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs = None

    return attn_output, attn_probs, past_key_value


def get_llama_layers(llava_model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the ModuleList of LLaMA decoder layers from a LlavaForConditionalGeneration model."""
    lm = llava_model.language_model
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
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
    """Patch LLaMA self-attention layers [start_layer, end_layer) with llama_igap_forward."""
    layers = get_llama_layers(model)
    end_layer = min(end_layer, len(layers))

    for i in range(start_layer, end_layer):
        sa = layers[i].self_attn
        sa.img_start_idx = int(img_start_idx)
        sa.img_end_idx = int(img_end_idx)
        sa.keep_head_ratio = float(keep_head_ratio)
        sa.suppression_alpha = float(suppression_alpha)
        sa.use_igap_img = bool(use_igap_img)

        if isinstance(sa.forward, functools.partial):
            sa.forward = sa.forward.func
        sa.forward = types.MethodType(llama_igap_forward, sa)

    print(
        f"IGAP patched layers [{start_layer}, {end_layer}). "
        f"keep_ratio={keep_head_ratio}, suppression_alpha={suppression_alpha}"
    )
