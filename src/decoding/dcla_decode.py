"""
src/decoding/dcla_decode.py
===========================
Dynamic Contrastive Logit Adjustment (DCLA) — dual-pass greedy decoding engine.

At each decode step two distributions are computed:
  Pass-1 (IGAP off): standard language prior from the original KV cache.
  Pass-2 (IGAP on): image-grounded distribution from attended-embed KV cache.

Routing (in order of precedence):
  max_prob > CONFIDENCE_GATE  ->  greedy (baseline logits)
  JS <= js_threshold          ->  complementary: logits_orig + a1 * logits_igap
  JS >  js_threshold          ->  contrastive:   (1+a2)*logits_orig - a2*logits_igap
                                  + APC mask (tokens below APC_FRACTION of max set to -inf)
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
) -> Tuple[str, List[Tuple[int, float, str]]]:
    """Run DCLA dual-pass greedy decoding for one vision-language sample.

    Returns (answer_string, per_step_trace).
    trace entries: (token_id, js_divergence, routing_mode)
    routing_mode in {"greedy", "complement", "contrast"}.
    """
    lm_device: torch.device = next(model.language_model.parameters()).device

    vision_inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    for key, val in vision_inputs.items():
        if torch.is_tensor(val):
            vision_inputs[key] = val.to(
                lm_device,
                dtype=(torch.float16 if key == "pixel_values" else None),
            )

    inputs_embeds_att, att_mask_att = build_attended_embeds(
        model, vision_inputs, img_start, img_end, lam=lam, token_alpha=token_alpha
    )
    inputs_embeds_att = inputs_embeds_att.to(lm_device, dtype=torch.float16)
    att_mask_att = att_mask_att.to(lm_device)
    s_merged: int = att_mask_att.shape[1]

    _igap_toggle["active"] = False
    out1 = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs["pixel_values"],
        use_cache=True,
    )
    past_kv_1 = out1.past_key_values
    mask1 = torch.ones((1, s_merged), dtype=torch.long, device=lm_device)

    _igap_toggle["active"] = True
    out2 = model(inputs_embeds=inputs_embeds_att, attention_mask=att_mask_att, use_cache=True)
    past_kv_2 = out2.past_key_values
    mask2 = att_mask_att.clone()
    _igap_toggle["active"] = False

    generated_ids: List[int] = []
    trace: List[Tuple[int, float, str]] = []
    next_id: torch.Tensor = torch.zeros((1, 1), dtype=torch.long, device=lm_device)

    for step in range(max_new_tokens):
        if step == 0:
            logits_orig: torch.Tensor = out1.logits[0, -1].float()
            logits_igap: torch.Tensor = out2.logits[0, -1].float()
        else:
            _igap_toggle["active"] = False
            d1 = model(input_ids=next_id, attention_mask=mask1, past_key_values=past_kv_1, use_cache=True)
            logits_orig = d1.logits[0, -1].float()
            past_kv_1 = d1.past_key_values

            _igap_toggle["active"] = True
            d2 = model(input_ids=next_id, attention_mask=mask2, past_key_values=past_kv_2, use_cache=True)
            logits_igap = d2.logits[0, -1].float()
            past_kv_2 = d2.past_key_values
            _igap_toggle["active"] = False

        eps: float = 1e-8
        p = F.softmax(logits_orig, dim=-1)
        q = F.softmax(logits_igap, dim=-1)
        m = 0.5 * (p + q)
        js: float = 0.5 * (
            torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
            + torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
        ).item()

        base_probs = p
        max_prob: float = torch.max(base_probs).item()

        if max_prob > CONFIDENCE_GATE:
            final_logits = logits_orig.clone()
            mode = "greedy"
        elif js <= js_threshold:
            final_logits = logits_orig + alpha1 * logits_igap
            mode = "complement"
        else:
            final_logits = (1.0 + alpha2) * logits_orig - alpha2 * logits_igap
            plausible_mask = base_probs >= APC_FRACTION * max_prob
            final_logits[~plausible_mask] = -float("inf")
            mode = "contrast"

        next_id = torch.argmax(final_logits, dim=-1).view(1, 1)
        token_id: int = int(next_id.item())
        generated_ids.append(token_id)
        trace.append((token_id, round(js, 5), mode))

        ones = torch.ones((1, 1), dtype=torch.long, device=lm_device)
        mask1 = torch.cat([mask1, ones], dim=1)
        mask2 = torch.cat([mask2, ones], dim=1)

        if token_id == processor.tokenizer.eos_token_id:
            break

    answer_ids = torch.tensor([generated_ids], device=lm_device)
    answer: str = processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return answer, trace
