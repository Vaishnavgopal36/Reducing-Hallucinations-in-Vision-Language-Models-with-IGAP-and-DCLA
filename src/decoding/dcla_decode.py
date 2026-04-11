"""
Dynamic Contrastive Logit Adjustment (DCLA) decoding.

Dual-pass per decode step:
1) Pass-1 (IGAP off): baseline language prior.
2) Pass-2 (IGAP on): image-grounded logits.

Routing:
- Confidence Gate: ``max_prob > c``        -> greedy pass-1 token.
- Complementary Amplification (low JS):    -> ``logits_1 + a1 * logits_2``.
- Contrastive Suppression (high JS):       -> ``(1+a2)*logits_1 - a2*logits_2``
  plus APC plausibility mask.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.model.igap_patch import _igap_toggle
from src.model.vision_utils import build_attended_embeds

CONFIDENCE_GATE: float = 0.65
APC_FRACTION: float = 0.10


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
    """Run DCLA dual-pass decoding on one image-question sample.

    Returns:
    - decoded answer string
    - decode trace as ``[(token_id, js_divergence, mode), ...]``
    """
    lm_device: torch.device = next(model.language_model.parameters()).device

    vision_inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    for key, value in vision_inputs.items():
        if torch.is_tensor(value):
            dtype = torch.float16 if key == "pixel_values" else None
            vision_inputs[key] = value.to(device=lm_device, dtype=dtype)

    if "pixel_values" not in vision_inputs:
        raise ValueError("Processor output missing 'pixel_values' for multimodal decoding.")

    base_attention_mask = vision_inputs.get("attention_mask")
    if base_attention_mask is None:
        base_attention_mask = torch.ones(
            (vision_inputs["input_ids"].shape[0], vision_inputs["input_ids"].shape[1]),
            dtype=torch.long,
            device=lm_device,
        )
    else:
        base_attention_mask = base_attention_mask.to(device=lm_device, dtype=torch.long)

    # Build pass-2 attended embeddings from IGAP-disabled prefill attentions.
    inputs_embeds_att, att_mask_att = build_attended_embeds(
        model=model,
        vision_inputs=vision_inputs,
        img_start=img_start,
        img_end=img_end,
        lam=lam,
        token_alpha=token_alpha,
    )
    model_dtype = next(model.parameters()).dtype
    inputs_embeds_att = inputs_embeds_att.to(device=lm_device, dtype=model_dtype)
    att_mask_att = att_mask_att.to(device=lm_device, dtype=torch.long)

    generated_ids: List[int] = []
    trace: List[Tuple[int, float, str]] = []
    next_token = torch.zeros((1, 1), dtype=torch.long, device=lm_device)

    try:
        # Prefill pass-1 (IGAP OFF)
        _igap_toggle["active"] = False
        out_pass1 = model(
            input_ids=vision_inputs["input_ids"],
            attention_mask=base_attention_mask,
            pixel_values=vision_inputs["pixel_values"],
            use_cache=True,
        )
        past_kv_pass1 = out_pass1.past_key_values
        mask_pass1 = base_attention_mask.clone()

        # Prefill pass-2 (IGAP ON)
        _igap_toggle["active"] = True
        out_pass2 = model(
            inputs_embeds=inputs_embeds_att,
            attention_mask=att_mask_att,
            use_cache=True,
        )
        past_kv_pass2 = out_pass2.past_key_values
        mask_pass2 = att_mask_att.clone()

        logits_pass1 = out_pass1.logits[0, -1].float()
        logits_pass2 = out_pass2.logits[0, -1].float()

        eos_token_id = processor.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if step > 0:
                _igap_toggle["active"] = False
                step_pass1 = model(
                    input_ids=next_token,
                    attention_mask=mask_pass1,
                    past_key_values=past_kv_pass1,
                    use_cache=True,
                )
                logits_pass1 = step_pass1.logits[0, -1].float()
                past_kv_pass1 = step_pass1.past_key_values

                _igap_toggle["active"] = True
                step_pass2 = model(
                    input_ids=next_token,
                    attention_mask=mask_pass2,
                    past_key_values=past_kv_pass2,
                    use_cache=True,
                )
                logits_pass2 = step_pass2.logits[0, -1].float()
                past_kv_pass2 = step_pass2.past_key_values

            p = F.softmax(logits_pass1, dim=-1)
            q = F.softmax(logits_pass2, dim=-1)
            m = 0.5 * (p + q)
            eps = 1e-8
            js_divergence = 0.5 * (
                torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
                + torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
            ).item()

            max_prob = float(torch.max(p).item())
            if max_prob > confidence_gate:
                final_logits = logits_pass1.clone()
                mode = "greedy"
            elif js_divergence <= js_threshold:
                final_logits = logits_pass1 + (alpha1 * logits_pass2)
                mode = "complement"
            else:
                final_logits = ((1.0 + alpha2) * logits_pass1) - (alpha2 * logits_pass2)
                apc_mask = p >= (apc_fraction * max_prob)
                if not bool(torch.any(apc_mask)):
                    apc_mask[torch.argmax(p)] = True
                final_logits = final_logits.masked_fill(
                    ~apc_mask,
                    torch.finfo(final_logits.dtype).min,
                )
                mode = "contrast"

            next_token = torch.argmax(final_logits, dim=-1).view(1, 1)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            trace.append((token_id, round(js_divergence, 6), mode))

            if eos_token_id is not None and token_id == int(eos_token_id):
                break

            one = torch.ones((1, 1), dtype=torch.long, device=lm_device)
            mask_pass1 = torch.cat([mask_pass1, one], dim=1)
            mask_pass2 = torch.cat([mask_pass2, one], dim=1)

    finally:
        _igap_toggle["active"] = False

    answer = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return answer, trace
