"""
Vision-token utilities for LLaVA-style multimodal inputs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.model.igap_patch import _igap_toggle

IMAGE_TOKEN_ID: int = 32000


def _project_image_features(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Project vision features into language embedding space."""
    if hasattr(model, "get_image_features"):
        try:
            image_features = model.get_image_features(pixel_values=pixel_values)
        except TypeError:
            image_features = model.get_image_features(pixel_values)
        if isinstance(image_features, tuple):
            image_features = image_features[0]
        return image_features

    if not hasattr(model, "vision_tower") or not hasattr(model, "multi_modal_projector"):
        raise AttributeError("Model does not expose vision tower/projector helpers.")

    vision_tower = model.vision_tower
    vt_out = vision_tower(pixel_values, output_hidden_states=True)
    return model.multi_modal_projector(vt_out.last_hidden_state)


@torch.no_grad()
def get_image_token_range_hf(
    llava_model: torch.nn.Module,
    inputs: dict,
) -> Optional[Tuple[int, int]]:
    """Return merged-token image span ``(img_start, img_end)`` for one sample.

    ``img_start`` corresponds to the ``<image>`` placeholder token position in
    ``input_ids``. ``img_end`` is computed as ``img_start + num_image_tokens``
    after the model's own vision projection path.
    """
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.numel() == 0:
        return None

    placeholder_pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
    if placeholder_pos.numel() == 0:
        return None

    pixel_values = inputs.get("pixel_values")
    if pixel_values is None:
        return None

    if hasattr(llava_model, "vision_tower"):
        vision_device = next(llava_model.vision_tower.parameters()).device
    else:
        vision_device = next(llava_model.parameters()).device

    pixel_values = pixel_values.to(device=vision_device, dtype=torch.float16)
    image_features = _project_image_features(llava_model, pixel_values)
    num_image_tokens = int(image_features.shape[1])

    img_start = int(placeholder_pos[0].item())
    img_end = img_start + num_image_tokens
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
    """Construct attended embeddings for DCLA pass-2.

    A prefill run (with IGAP disabled) is used to measure image-token attention.
    The top ``lam`` fraction of image tokens are preserved, while the remaining
    tokens are scaled by ``token_alpha``.
    """
    if img_end <= img_start:
        raise ValueError(f"Invalid image token span: [{img_start}, {img_end})")

    _igap_toggle["active"] = False
    prefill_out = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs.get("pixel_values"),
        output_attentions=True,
        use_cache=False,
    )
    _igap_toggle["active"] = False

    attentions = getattr(prefill_out, "attentions", None)
    if not attentions:
        raise RuntimeError("Prefill call did not return attentions; cannot build attended embeds.")

    attn_layers = [attn.to(torch.float32) for attn in attentions if attn is not None]
    if not attn_layers:
        raise RuntimeError("All attention tensors are None during prefill.")

    all_attn = torch.stack(attn_layers, dim=0)  # [L, B, H, Q, K]
    seq_len = int(all_attn.shape[-1])
    img_start = max(0, min(int(img_start), seq_len))
    img_end = max(img_start, min(int(img_end), seq_len))
    if img_end <= img_start:
        raise ValueError("Image token span falls outside merged sequence length.")

    img_len = img_end - img_start
    query_index = int(all_attn.shape[-2] - 1)
    img_attn = all_attn[:, 0, :, query_index, img_start:img_end]
    avg_attn = img_attn.mean(dim=(0, 1))

    num_keep = max(1, min(img_len, int(round(float(lam) * img_len))))
    top_idx = torch.topk(avg_attn, k=num_keep, dim=-1).indices

    keep_mask = torch.full(
        (img_len,),
        fill_value=float(token_alpha),
        dtype=avg_attn.dtype,
        device=avg_attn.device,
    )
    keep_mask[top_idx] = 1.0

    input_ids = vision_inputs["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)

    pixel_values = vision_inputs.get("pixel_values")
    if pixel_values is None:
        raise ValueError("vision_inputs['pixel_values'] is required for attended embeds.")

    if hasattr(model, "vision_tower"):
        vision_device = next(model.vision_tower.parameters()).device
    else:
        vision_device = inputs_embeds.device

    image_features = _project_image_features(
        model,
        pixel_values.to(device=vision_device, dtype=torch.float16),
    )

    if image_features.shape[1] == img_len + 1:
        # Some checkpoints expose CLS + patches; drop CLS for merged-token alignment.
        image_features = image_features[:, 1:, :]

    if image_features.shape[1] != img_len:
        raise ValueError(
            "Projected image feature length does not match merged image token span: "
            f"features={image_features.shape[1]}, span={img_len}."
        )

    image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    inputs_embeds_att = inputs_embeds.clone()
    scale = keep_mask.to(device=inputs_embeds_att.device, dtype=inputs_embeds_att.dtype)
    inputs_embeds_att[:, img_start:img_end, :] = image_features * scale[None, :, None]

    attention_mask = vision_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(
            (inputs_embeds_att.shape[0], inputs_embeds_att.shape[1]),
            dtype=torch.long,
            device=inputs_embeds_att.device,
        )
    else:
        attention_mask = attention_mask.to(device=inputs_embeds_att.device, dtype=torch.long)

    return inputs_embeds_att, attention_mask
