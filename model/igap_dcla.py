"""Notebook-derived hybrid IGAP-DCLA runtime for LLaVA models."""

from __future__ import annotations

import functools
import math
import types
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

IMAGE_TOKEN_ID: int = 32000
CONFIDENCE_GATE: float = 0.65
APC_FRACTION: float = 0.10

_spin_toggle: dict[str, bool] = {"active": False}

spin_debug: dict[str, float | int] = {
    "forward_calls": 0,
    "spin_active_calls": 0,
    "q_len1_calls": 0,
    "avg_suppressed_fraction_sum": 0.0,
}


def _repeat_kv_for_full_heads(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if attention_mask is None:
        return None

    if attention_mask.dim() == 2:
        mask = attention_mask.to(dtype=attn_logits.dtype)
        neg_inf = torch.finfo(attn_logits.dtype).min
        return (1.0 - mask[:, None, None, :]) * neg_inf

    if attention_mask.dim() == 4:
        return attention_mask.to(dtype=attn_logits.dtype)

    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")


def _project_image_features(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
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


def llama_spin_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
):
    del use_cache

    spin_debug["forward_calls"] += 1
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
            raise ValueError("layer_idx missing on attention layer.")
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
        attn_logits = torch.maximum(
            attn_logits,
            attn_logits.new_full((), torch.finfo(attn_logits.dtype).min),
        )

    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

    if getattr(self, "use_spin_img", False) and q_len == 1 and _spin_toggle["active"]:
        spin_debug["spin_active_calls"] += 1
        spin_debug["q_len1_calls"] += 1

        keep_ratio = float(getattr(self, "keep_head_ratio", 0.95))
        num_keep = max(1, min(self.num_heads, int(round(keep_ratio * self.num_heads))))

        img_start = max(0, min(int(getattr(self, "img_start_idx", 0)), attn_probs.shape[-1]))
        img_end = max(img_start, min(int(getattr(self, "img_end_idx", img_start)), attn_probs.shape[-1]))

        mask = torch.ones(
            (bsz, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        if img_end > img_start:
            head_scores = attn_probs[:, :, -1, img_start:img_end].sum(dim=-1)
            _, keep_idx = torch.topk(head_scores, k=num_keep, dim=1)
            mask.fill_(float(getattr(self, "suppression_alpha", 0.08)))
            mask.scatter_(1, keep_idx, 1.0)

        spin_debug["avg_suppressed_fraction_sum"] += float((mask != 1.0).float().mean().item())
        mask = mask.view(bsz, 1, self.num_heads)
    else:
        mask = torch.ones(
            (bsz, q_len, self.num_heads),
            dtype=query_states.dtype,
            device=query_states.device,
        )

    attn_output = torch.matmul(attn_probs, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = torch.einsum("bqh,bqhd->bqhd", mask, attn_output)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_probs = None

    return attn_output, attn_probs, past_key_value


def get_llama_layers(llava_model: torch.nn.Module) -> torch.nn.ModuleList:
    lm = llava_model.language_model
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if hasattr(lm, "layers"):
        return lm.layers
    if hasattr(llava_model, "model") and hasattr(llava_model.model, "layers"):
        return llava_model.model.layers
    raise AttributeError("Could not locate LLaMA layers.")


def apply_spin_to_llava(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    img_start_idx: int,
    img_end_idx: int,
    keep_head_ratio: float = 0.95,
    suppression_alpha: float = 0.08,
    use_spin_img: bool = True,
) -> None:
    layers = get_llama_layers(model)
    end_layer = min(end_layer, len(layers))

    for layer_idx in range(start_layer, end_layer):
        self_attn = layers[layer_idx].self_attn
        self_attn.img_start_idx = int(img_start_idx)
        self_attn.img_end_idx = int(img_end_idx)
        self_attn.keep_head_ratio = float(keep_head_ratio)
        self_attn.suppression_alpha = float(suppression_alpha)
        self_attn.use_spin_img = bool(use_spin_img)

        if isinstance(self_attn.forward, functools.partial):
            self_attn.forward = self_attn.forward.func
        self_attn.forward = types.MethodType(llama_spin_forward, self_attn)


apply_igap_to_llava = apply_spin_to_llava


@torch.no_grad()
def get_image_token_range_hf(
    llava_model: torch.nn.Module,
    inputs: dict,
) -> Optional[Tuple[int, int]]:
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
    token_alpha: float = 0.06,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if img_end <= img_start:
        raise ValueError(f"Invalid image token span: [{img_start}, {img_end})")

    _spin_toggle["active"] = False
    prefill_out = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs.get("pixel_values"),
        output_attentions=True,
        use_cache=False,
    )
    _spin_toggle["active"] = False

    attentions = getattr(prefill_out, "attentions", None)
    if not attentions:
        raise RuntimeError("Prefill call did not return attentions; cannot build attended embeds.")

    all_attn = torch.stack([attn.to(torch.float32) for attn in attentions if attn is not None], dim=0)
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

    inputs_embeds = model.get_input_embeddings()(vision_inputs["input_ids"])
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


@torch.no_grad()
def dynamic_decode_one_sample(
    model: torch.nn.Module,
    processor,
    prompt: str,
    image,
    img_start: int,
    img_end: int,
    max_new_tokens: int = 128,
    js_threshold: float = 0.05,
    alpha1: float = 4.0,
    alpha2: float = 1.0,
    lam: float = 0.2,
    token_alpha: float = 0.06,
    confidence_gate: float = CONFIDENCE_GATE,
    apc_fraction: float = APC_FRACTION,
) -> Tuple[str, List[Tuple[int, float, str]]]:
    lm_device: torch.device = next(model.language_model.parameters()).device

    vision_inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    for key, value in vision_inputs.items():
        if torch.is_tensor(value):
            dtype = torch.float16 if key == "pixel_values" else None
            vision_inputs[key] = value.to(lm_device, dtype=dtype)

    inputs_embeds_att, att_mask_att = build_attended_embeds(
        model,
        vision_inputs,
        img_start,
        img_end,
        lam=lam,
        token_alpha=token_alpha,
    )
    model_dtype = next(model.parameters()).dtype
    inputs_embeds_att = inputs_embeds_att.to(lm_device, dtype=model_dtype)
    att_mask_att = att_mask_att.to(lm_device, dtype=torch.long)

    base_attention_mask = vision_inputs.get("attention_mask")
    if base_attention_mask is None:
        base_attention_mask = torch.ones(
            (vision_inputs["input_ids"].shape[0], vision_inputs["input_ids"].shape[1]),
            dtype=torch.long,
            device=lm_device,
        )
    else:
        base_attention_mask = base_attention_mask.to(device=lm_device, dtype=torch.long)

    generated_ids: List[int] = []
    generated_trace: List[Tuple[int, float, str]] = []

    try:
        _spin_toggle["active"] = False
        out1 = model(
            input_ids=vision_inputs["input_ids"],
            attention_mask=base_attention_mask,
            pixel_values=vision_inputs["pixel_values"],
            use_cache=True,
        )
        past_kv_1 = out1.past_key_values
        mask1 = base_attention_mask.clone()

        _spin_toggle["active"] = True
        out2 = model(
            inputs_embeds=inputs_embeds_att,
            attention_mask=att_mask_att,
            use_cache=True,
        )
        past_kv_2 = out2.past_key_values
        mask2 = att_mask_att.clone()

        eos_token_id = processor.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if step == 0:
                logits_orig = out1.logits[0, -1].float()
                logits_spin = out2.logits[0, -1].float()
            else:
                _spin_toggle["active"] = False
                d1 = model(
                    input_ids=next_id,
                    attention_mask=mask1,
                    past_key_values=past_kv_1,
                    use_cache=True,
                )
                logits_orig = d1.logits[0, -1].float()
                past_kv_1 = d1.past_key_values

                _spin_toggle["active"] = True
                d2 = model(
                    input_ids=next_id,
                    attention_mask=mask2,
                    past_key_values=past_kv_2,
                    use_cache=True,
                )
                logits_spin = d2.logits[0, -1].float()
                past_kv_2 = d2.past_key_values

            eps = 1e-8
            p = F.softmax(logits_orig, dim=-1)
            q = F.softmax(logits_spin, dim=-1)
            m = 0.5 * (p + q)
            js = 0.5 * (
                torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
                + torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
            ).item()

            max_prob = float(torch.max(p).item())
            if max_prob > confidence_gate:
                final_logits = logits_orig.clone()
                mode = "greedy"
            elif js <= js_threshold:
                final_logits = logits_orig + alpha1 * logits_spin
                mode = "complement"
            else:
                final_logits = (1 + alpha2) * logits_orig - alpha2 * logits_spin
                plausible_mask = p >= (apc_fraction * max_prob)
                if not bool(torch.any(plausible_mask)):
                    plausible_mask[torch.argmax(p)] = True
                final_logits = final_logits.masked_fill(
                    ~plausible_mask,
                    torch.finfo(final_logits.dtype).min,
                )
                mode = "contrast"

            next_id = torch.argmax(final_logits, dim=-1).view(1, 1)
            token_id = int(next_id.item())
            generated_ids.append(token_id)
            generated_trace.append((token_id, round(js, 5), mode))

            if eos_token_id is not None and token_id == int(eos_token_id):
                break

            one = torch.ones((1, 1), dtype=torch.long, device=lm_device)
            mask1 = torch.cat([mask1, one], dim=1)
            mask2 = torch.cat([mask2, one], dim=1)

    finally:
        _spin_toggle["active"] = False

    answer = processor.batch_decode(
        torch.tensor([generated_ids], device=lm_device),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, generated_trace


__all__ = [
    "APC_FRACTION",
    "CONFIDENCE_GATE",
    "IMAGE_TOKEN_ID",
    "_spin_toggle",
    "apply_igap_to_llava",
    "apply_spin_to_llava",
    "build_attended_embeds",
    "dynamic_decode_one_sample",
    "get_image_token_range_hf",
    "get_llama_layers",
    "llama_spin_forward",
    "spin_debug",
]
