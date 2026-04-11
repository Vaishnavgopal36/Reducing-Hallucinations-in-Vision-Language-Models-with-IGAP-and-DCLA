"""
src/model/vision_utils.py
=========================
Vision token utilities for LLaVA-1.5.

This module provides two helper functions:

* ``get_image_token_range_hf`` – discover the exact positions of image patch
  tokens inside the merged embedding sequence that LLaVA constructs internally.
* ``build_attended_embeds`` – run a prefill forward pass with *all* attention
  weights captured, identify the top-λ most-attended image tokens, and return
  modified ``inputs_embeds`` in which low-attention tokens are scaled down by
  ``token_alpha`` (the IGAP token-level soft mask).

The IGAP global toggle (``_igap_toggle``) is explicitly set to ``False``
during the prefill pass so that head-suppression does not interfere with the
attention-weight measurement.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.model.igap_patch import _igap_toggle

# The placeholder token ID for the ``<image>`` special token in LLaVA-1.5 7B.
IMAGE_TOKEN_ID: int = 32000


@torch.no_grad()
def get_image_token_range_hf(
    llava_model: torch.nn.Module,
    inputs: dict,
) -> Optional[Tuple[int, int]]:
    """Find the image-token range inside the merged LLaVA embedding sequence.

    HF LLaVA uses a *single* ``<image>`` placeholder token in ``input_ids``
    but internally replaces it with ``num_patches`` vision feature vectors.
    This function locates the placeholder position and queries the vision tower
    to determine how many patch tokens are produced.

    Parameters
    ----------
    llava_model:
        A loaded ``LlavaForConditionalGeneration`` instance.
    inputs:
        A dict returned by ``processor(...)`` containing at least
        ``input_ids`` and ``pixel_values``.

    Returns
    -------
    (img_start, img_end) : Tuple[int, int] or None
        ``img_start`` is the position of the ``<image>`` placeholder;
        ``img_end = img_start + num_patches``.  Returns ``None`` if no image
        placeholder is found or ``pixel_values`` is missing.
    """
    ids = inputs["input_ids"][0]
    pos = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None

    img_start: int = int(pos[0].item())

    pixel_values: Optional[torch.Tensor] = inputs.get("pixel_values", None)
    if pixel_values is None:
        return None

    # Run the vision tower to retrieve the actual patch-token count
    vision_tower = llava_model.vision_tower
    pv = pixel_values.to(next(vision_tower.parameters()).device)

    vt_out = vision_tower(pv, output_hidden_states=True)
    # last_hidden_state: [batch, num_patches, hidden_dim]
    num_patches: int = int(vt_out.last_hidden_state.shape[1])

    img_end: int = img_start + num_patches
    return img_start, img_end


@torch.no_grad()
def build_attended_embeds(
    model: torch.nn.Module,
    vision_inputs: dict,
    img_start: int,
    img_end: int,
    lam: float = 0.2,
    token_alpha: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct image-attended input embeddings for IGAP pass-2.

    Algorithm
    ---------
    1. **Prefill forward** (IGAP toggle=False, ``output_attentions=True``):
       obtain attention tensors from every layer.
    2. **Identify top-λ tokens**: average attention across all layers and
       heads for the *last query position* over image token positions; keep
       the top ``⌈λ × img_len⌉`` indices.
    3. **Build keep_mask**: a float vector of length ``img_len``; value 1.0
       at kept indices, ``token_alpha`` elsewhere.
    4. **Re-compute base embeddings** and vision features; overwrite the
       image slice in-place with ``img_feats * keep_mask``.

    The IGAP toggle is set to ``False`` before the prefill so head-suppression
    does not distort the attention measurement.

    Parameters
    ----------
    model:
        ``LlavaForConditionalGeneration``.
    vision_inputs:
        Processor output dict with ``input_ids``, ``attention_mask``, and
        ``pixel_values``.
    img_start:
        Start index of image tokens in the merged embedding sequence.
    img_end:
        End index (exclusive) of image tokens.
    lam:
        Fraction of image tokens to keep at full scale.  E.g. ``0.2``
        retains the 20 % most-attended patch tokens.
    token_alpha:
        Scale factor applied to non-top image tokens (analogous to
        ``suppression_alpha`` in IGAP head pruning).

    Returns
    -------
    inputs_embeds_att : torch.Tensor
        Modified input embeddings, shape ``(1, seq_len, hidden_dim)``.
    att_mask_att : torch.Tensor
        Attention mask matching the original (unmodified) input length.
    """
    img_len: int = img_end - img_start

    # ---- Pass 0: prefill with attention output (IGAP off) --------------------
    _igap_toggle["active"] = False

    prefill_out = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs.get("pixel_values"),
        output_attentions=True,
        use_cache=False,
    )

    # ---- Compute per-token attention score over image positions --------------
    # all_attns: [num_layers, batch, num_heads, q_len, kv_len]
    all_attns = torch.stack(prefill_out.attentions, dim=0)

    # Focus on the last query position (≈ the "current token" attending to context)
    img_attn = all_attns[:, 0, :, -1, img_start:img_end]  # [layers, heads, img_len]
    avg_attn = img_attn.mean(dim=(0, 1))                   # [img_len]

    num_keep: int = max(1, int(round(lam * img_len)))
    _, top_idx = torch.topk(avg_attn, k=num_keep)

    # ---- Build the token-level scale mask ------------------------------------
    keep_mask = torch.full(
        (img_len,),
        fill_value=float(token_alpha),
        dtype=torch.float16,
        device=avg_attn.device,
    )
    keep_mask[top_idx] = 1.0

    # ---- Re-compute text + vision embeddings from scratch --------------------
    embed_fn = model.get_input_embeddings()
    inputs_embeds: torch.Tensor = embed_fn(vision_inputs["input_ids"])

    vision_tower = model.vision_tower
    projector = model.multi_modal_projector
    pv = vision_inputs["pixel_values"].to(next(vision_tower.parameters()).device)

    # Extract patch features and project them into the language model space
    vt_out = vision_tower(pv, output_hidden_states=True)
    img_feats: torch.Tensor = projector(vt_out.last_hidden_state)

    # ---- In-place overwrite of the image embedding slice --------------------
    inputs_embeds_att: torch.Tensor = inputs_embeds.clone()
    keep_mask = keep_mask.to(
        device=inputs_embeds_att.device, dtype=inputs_embeds_att.dtype
    )
    # Multiply each patch vector by its keep_mask scalar
    inputs_embeds_att[0, img_start:img_end, :] = (
        img_feats[0] * keep_mask.unsqueeze(-1)
    )

    att_mask_att: torch.Tensor = vision_inputs.get("attention_mask")
    if att_mask_att is None:
        # Fallback: create a fully-visible mask matching the embedding sequence length
        seq_len = inputs_embeds_att.shape[1]
        att_mask_att = torch.ones(
            (1, seq_len), dtype=torch.long, device=inputs_embeds_att.device
        )

    # Sanity-check: img_feats spatial dimension must match the embedding slice
    assert img_feats.shape[1] == img_len, (
        f"Projector output length ({img_feats.shape[1]}) does not match "
        f"img_end - img_start ({img_len}).  Check vision tower / projector output."
    )

    return inputs_embeds_att, att_mask_att
