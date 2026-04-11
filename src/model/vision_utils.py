"""
src/model/vision_utils.py
=========================
Vision token utilities for LLaVA-1.5.

get_image_token_range_hf: locates image patch token positions in the merged embedding sequence.
build_attended_embeds: runs a prefill pass to measure image-token attention, then returns
    modified input embeddings with low-attended patches scaled by token_alpha.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.model.igap_patch import _igap_toggle

IMAGE_TOKEN_ID: int = 32000


@torch.no_grad()
def get_image_token_range_hf(
    llava_model: torch.nn.Module,
    inputs: dict,
) -> Optional[Tuple[int, int]]:
    """Return (img_start, img_end) positions in the merged embedding sequence.

    img_start is the position of the <image> placeholder token; img_end = img_start + num_patches.
    Returns None if no image placeholder or pixel_values are found.
    """
    ids = inputs["input_ids"][0]
    pos = (ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None

    pixel_values: Optional[torch.Tensor] = inputs.get("pixel_values", None)
    if pixel_values is None:
        return None

    vision_tower = llava_model.vision_tower
    pv = pixel_values.to(next(vision_tower.parameters()).device)
    vt_out = vision_tower(pv, output_hidden_states=True)
    num_patches: int = int(vt_out.last_hidden_state.shape[1])

    img_start: int = int(pos[0].item())
    return img_start, img_start + num_patches


@torch.no_grad()
def build_attended_embeds(
    model: torch.nn.Module,
    vision_inputs: dict,
    img_start: int,
    img_end: int,
    lam: float = 0.2,
    token_alpha: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct image-attended embeddings for IGAP pass-2.

    Runs one prefill forward (IGAP off) to collect attention weights, identifies the
    top lam-fraction of image tokens by average attention, and returns modified
    inputs_embeds with low-attended tokens scaled by token_alpha.

    Returns (inputs_embeds_att, att_mask).
    """
    img_len: int = img_end - img_start
    _igap_toggle["active"] = False

    prefill_out = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs.get("pixel_values"),
        output_attentions=True,
        use_cache=False,
    )

    all_attns = torch.stack(prefill_out.attentions, dim=0)
    img_attn = all_attns[:, 0, :, -1, img_start:img_end]
    avg_attn = img_attn.mean(dim=(0, 1))

    num_keep: int = max(1, int(round(lam * img_len)))
    _, top_idx = torch.topk(avg_attn, k=num_keep)

    keep_mask = torch.full(
        (img_len,), fill_value=float(token_alpha), dtype=torch.float16, device=avg_attn.device
    )
    keep_mask[top_idx] = 1.0

    embed_fn = model.get_input_embeddings()
    inputs_embeds: torch.Tensor = embed_fn(vision_inputs["input_ids"])

    vision_tower = model.vision_tower
    pv = vision_inputs["pixel_values"].to(next(vision_tower.parameters()).device)
    vt_out = vision_tower(pv, output_hidden_states=True)
    img_feats: torch.Tensor = model.multi_modal_projector(vt_out.last_hidden_state)

    assert img_feats.shape[1] == img_len, (
        f"Projector output length ({img_feats.shape[1]}) does not match "
        f"img_end - img_start ({img_len}). Check vision tower / projector."
    )

    inputs_embeds_att: torch.Tensor = inputs_embeds.clone()
    keep_mask = keep_mask.to(device=inputs_embeds_att.device, dtype=inputs_embeds_att.dtype)
    inputs_embeds_att[0, img_start:img_end, :] = img_feats[0] * keep_mask.unsqueeze(-1)

    att_mask = vision_inputs.get("attention_mask")
    if att_mask is None:
        att_mask = torch.ones(
            (1, inputs_embeds_att.shape[1]), dtype=torch.long, device=inputs_embeds_att.device
        )

    return inputs_embeds_att, att_mask
