"""
src/model/igap_patch.py
=======================
Image-Guided Attention Pruning (IGAP) — PyTorch attention patching for LLaVA-1.5.

During decoding, IGAP identifies heads that attend strongly to image tokens
and suppresses the remaining heads by scaling their outputs with a small alpha.
This steers generation away from language-prior hallucinations and keeps
responses grounded in the visual input.

References
----------
Paper: "Reducing Hallucinations in Vision-Language Models with
        Image-Guided Attention Pruning (IGAP) and
        Dynamic Contrastive Logit Adjustment (DCLA)"
"""

from __future__ import annotations

import functools
import math
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# ---------------------------------------------------------------------------
# Global IGAP toggle
# ---------------------------------------------------------------------------
# The dual-pass DCLA decoder sets `_igap_toggle["active"] = True` only for
# pass-2 (the image-attended forward pass).  It is *always* False during the
# prefill that builds the KV cache in pass-1, preventing premature suppression
# of rich prefill context.
_igap_toggle: dict[str, bool] = {"active": False}

# ---------------------------------------------------------------------------
# Debug counters (reset by the caller before each inference run)
# ---------------------------------------------------------------------------
igap_debug: dict[str, float | int] = {
    "forward_calls": 0,
    "igap_active_calls": 0,
    "q_len1_calls": 0,
    "avg_suppressed_fraction_sum": 0.0,
}


# ---------------------------------------------------------------------------
# Patched LLaMA self-attention forward
# ---------------------------------------------------------------------------

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
    """Replacement forward for ``LlamaAttention``.

    Behaviour is identical to the standard forward, *except* when all three
    conditions hold simultaneously:

    1. The layer attribute ``use_igap_img`` is ``True``.
    2. The sequence length ``q_len == 1`` (decode step, not prefill).
    3. The global toggle ``_igap_toggle["active"]`` is ``True``.

    Under those conditions, each head's cumulative attention over the image
    token range ``[img_start_idx, img_end_idx)`` is computed.  The top-k
    image-attending heads (where k = ``keep_head_ratio × num_heads``) are
    kept at full strength; all remaining heads are multiplied by
    ``suppression_alpha``.

    Parameters
    ----------
    self:
        The ``LlamaAttention`` instance (bound via ``types.MethodType``).
    hidden_states:
        Input tensor of shape ``(batch, q_len, hidden_size)``.
    attention_mask:
        Additive causal mask broadcastable to ``(batch, heads, q, kv)``.
    position_ids:
        Token position indices of shape ``(batch, q_len)``.
    past_key_value:
        HF ``Cache`` object (or ``None`` on first call).
    output_attentions:
        If ``True``, return the softmax attention tensor.
    use_cache:
        If ``True``, update and return the KV cache.
    **kwargs:
        Accepts ``position_embeddings`` (cos/sin tuple) forwarded by some
        HF versions.

    Returns
    -------
    attn_output : torch.Tensor
        Shape ``(batch, q_len, hidden_size)``.
    attn_probs : torch.Tensor or None
        Softmax attention weights if ``output_attentions`` else ``None``.
    past_key_value :
        Updated KV cache or ``None``.
    """
    igap_debug["forward_calls"] += 1

    bsz, q_len, _ = hidden_states.size()

    # ---- Project Q / K / V ---------------------------------------------------
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

    # ---- KV-cache sequence length --------------------------------------------
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                "layer_idx is required on the attention layer when use_cache=True."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # ---- Rotary position embeddings ------------------------------------------
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        # transformers >= 4.38 removed the `seq_len` kwarg from rotary_emb;
        # pass position_ids instead for forward compatibility.
        try:
            cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        except TypeError:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # ---- Update KV cache -----------------------------------------------------
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # ---- Repeat K/V for GQA if needed ----------------------------------------
    # (LLaVA-1.5 7B uses MHA so num_key_value_groups == 1, but we handle GQA.)
    num_kv_groups = self.num_heads // self.num_key_value_heads
    if num_kv_groups > 1:
        key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    # ---- Scaled dot-product attention ----------------------------------------
    attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attention_mask is not None:
        attn_logits = attn_logits + attention_mask
        attn_logits = torch.maximum(
            attn_logits,
            attn_logits.new_full((), torch.finfo(attn_logits.dtype).min),
        )

    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    # ---- IGAP head suppression (decode steps only) ---------------------------
    if (
        getattr(self, "use_igap_img", False)
        and q_len == 1
        and _igap_toggle["active"]
    ):
        igap_debug["igap_active_calls"] += 1
        igap_debug["q_len1_calls"] += 1

        keep_ratio: float = float(self.keep_head_ratio)
        num_keep: int = max(1, int(round(keep_ratio * self.num_heads)))

        img_start: int = max(0, min(int(self.img_start_idx), attn_probs.shape[-1]))
        img_end: int = max(img_start, min(int(self.img_end_idx), attn_probs.shape[-1]))

        # Sum attention over image token positions for latest query position
        head_scores = attn_probs[:, :, -1, img_start:img_end].sum(dim=-1)  # [B, H]
        _, keep_idx = torch.topk(head_scores, k=num_keep, dim=1)

        # Build per-head scale mask: 1.0 for kept heads, suppression_alpha elsewhere
        head_mask = torch.full(
            (bsz, self.num_heads),
            fill_value=float(self.suppression_alpha),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        head_mask.scatter_(1, keep_idx, 1.0)

        igap_debug["avg_suppressed_fraction_sum"] += (
            (head_mask != 1.0).float().mean().item()
        )

        # Reshape for broadcasting: [B, 1, H] to scale [B, q_len, H, head_dim]
        head_mask = head_mask.view(bsz, 1, self.num_heads)
    else:
        head_mask = torch.ones(
            (bsz, q_len, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )

    # ---- Weighted sum + output projection ------------------------------------
    attn_output = torch.matmul(attn_probs, value_states)            # [B, H, q, d]
    attn_output = attn_output.transpose(1, 2).contiguous()          # [B, q, H, d]

    # Apply the head-level scale mask
    attn_output = torch.einsum("bqh,bqhd->bqhd", head_mask, attn_output)

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs = None

    return attn_output, attn_probs, past_key_value


# ---------------------------------------------------------------------------
# Helper — locate LLaMA decoder layers
# ---------------------------------------------------------------------------

def get_llama_layers(llava_model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the ``ModuleList`` of LLaMA decoder layers inside a LLaVA model.

    Parameters
    ----------
    llava_model:
        A ``LlavaForConditionalGeneration`` instance (or any wrapper whose
        ``.language_model`` exposes transformer layers).

    Returns
    -------
    torch.nn.ModuleList
        The ordered list of ``LlamaDecoderLayer`` blocks.

    Raises
    ------
    AttributeError
        If the layer list cannot be located.
    """
    lm = llava_model.language_model
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    raise AttributeError(
        "Could not locate LLaMA decoder layers inside the provided LLaVA model. "
        "Ensure you are passing a `LlavaForConditionalGeneration` instance."
    )


# ---------------------------------------------------------------------------
# Public API — patch the model
# ---------------------------------------------------------------------------

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
    """Monkey-patch LLaMA self-attention layers with ``llama_igap_forward``.

    For every layer ``i`` in ``[start_layer, end_layer)``, the following
    attributes are attached to the ``self_attn`` sub-module and the
    ``forward`` method is replaced:

    * ``img_start_idx`` / ``img_end_idx`` — image token positions in the
      merged embedding sequence.
    * ``keep_head_ratio`` — fraction of heads to keep at full strength (e.g.
      0.95 keeps 95 % of heads).
    * ``suppression_alpha`` — scalar applied to non-kept heads (e.g. 0.08).
    * ``use_igap_img`` — master enable/disable flag per layer.

    Parameters
    ----------
    model:
        ``LlavaForConditionalGeneration`` to patch in-place.
    start_layer:
        Index of the first layer to patch (inclusive).
    end_layer:
        Index of the last layer to patch (exclusive).  Clamped to the actual
        number of layers.
    img_start_idx:
        Start position of image tokens in the merged embedding sequence.
    img_end_idx:
        End position (exclusive) of image tokens.
    keep_head_ratio:
        Fraction of heads whose image attention is highest to retain.
        Paper recommendation for LLaVA-1.5 7B: 0.95.
    suppression_alpha:
        Multiplicative scale applied to suppressed head outputs.
        Paper recommendation: 0.08.
    use_igap_img:
        If ``False``, the patch is installed but IGAP is effectively disabled —
        useful for running ablation baselines without re-loading the model.
    """
    layers = get_llama_layers(model)
    end_layer = min(end_layer, len(layers))

    for i in range(start_layer, end_layer):
        sa = layers[i].self_attn

        # Attach IGAP hyperparameters as layer attributes
        sa.img_start_idx = int(img_start_idx)
        sa.img_end_idx = int(img_end_idx)
        sa.keep_head_ratio = float(keep_head_ratio)
        sa.suppression_alpha = float(suppression_alpha)
        sa.use_igap_img = bool(use_igap_img)

        # Unwrap any existing functools.partial wrapper before re-binding
        if isinstance(sa.forward, functools.partial):
            sa.forward = sa.forward.func

        sa.forward = types.MethodType(llama_igap_forward, sa)

    print(
        f"✅ IGAP patched layers [{start_layer}, {end_layer}). "
        f"keep_ratio={keep_head_ratio}, suppression_alpha={suppression_alpha}, "
        f"use_igap_img={use_igap_img}"
    )
