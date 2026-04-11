"""
src/decoding/dcla_decode.py
===========================
Dynamic Contrastive Logit Adjustment (DCLA) — core generation engine.

Overview
--------
DCLA implements a *dual-pass* autoregressive decoding strategy for LLaVA-1.5.
At every decode step, two probability distributions are produced:

* **Pass-1 (IGAP off)** — the model's standard language prior, conditioned on
  the original prefill KV cache.
* **Pass-2 (IGAP on)** — an image-grounded distribution, conditioned on a
  modified prefill KV cache built from attended image embeddings
  (see ``build_attended_embeds``).

The Jensen-Shannon (JS) divergence between the two distributions determines
which routing mode is selected for the current token:

Routing Logic
-------------
+-----------------------------------+-----------------------------------------+
| Condition                         | Mode                                    |
+===================================+=========================================+
| max_prob > ``c`` (confidence gate)| **Greedy** — use pass-1 logits directly |
+-----------------------------------+-----------------------------------------+
| JS ≤ ``js_threshold``             | **Complementary amplification** —       |
|                                   | logits_orig + α₁ × logits_igap          |
+-----------------------------------+-----------------------------------------+
| JS > ``js_threshold``             | **Contrastive suppression** —           |
|                                   | (1+α₂)×logits_orig − α₂×logits_igap    |
|                                   | masked by the APC (10% of max)          |
+-----------------------------------+-----------------------------------------+

APC (Adaptive Plausibility Constraint)
---------------------------------------
During contrastive suppression, tokens whose baseline probability falls below
10 % of the top token's probability are set to ``-∞`` before argmax.  This
prevents degenerate low-probability hallucinations from "winning" when the
contrastive term amplifies their relative score.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from src.model.igap_patch import _igap_toggle
from src.model.vision_utils import build_attended_embeds

# Confidence gate threshold — if the greedy baseline is more confident than
# this, DCLA routing is bypassed entirely.
CONFIDENCE_GATE: float = 0.65

# Fraction of the top-1 probability below which tokens are masked in APC.
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
    """Run DCLA dual-pass greedy decoding for a single vision-language sample.

    Parameters
    ----------
    model:
        ``LlavaForConditionalGeneration`` (after ``apply_igap_to_llava``).
    processor:
        HF ``AutoProcessor`` for LLaVA-1.5.
    prompt:
        The full text prompt string, e.g.
        ``"USER: <image>\\nWhat is in this image?\\nASSISTANT:"``.
    image:
        A PIL ``Image`` object.
    img_start:
        Start position of image tokens in the merged embedding sequence.
    img_end:
        End position (exclusive) of image tokens.
    max_new_tokens:
        Maximum number of tokens to generate.
    js_threshold:
        JS-divergence boundary separating complementary from contrastive mode.
    alpha1:
        Complementary amplification coefficient
        (used when JS ≤ ``js_threshold``).
    alpha2:
        Contrastive suppression coefficient
        (used when JS > ``js_threshold``).
    lam:
        Fraction of image tokens to retain in ``build_attended_embeds``.
    token_alpha:
        Scale applied to low-attention image tokens in pass-2 embeddings.

    Returns
    -------
    answer : str
        Decoded text output (special tokens stripped).
    trace : List[Tuple[int, float, str]]
        Per-step trace: ``(token_id, js_divergence, routing_mode)``.
        ``routing_mode`` ∈ {``"greedy"``, ``"complement"``, ``"contrast"``}.
    """
    lm_device: torch.device = next(model.language_model.parameters()).device

    # ---- Prepare processor outputs ------------------------------------------
    vision_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    for key, val in vision_inputs.items():
        if torch.is_tensor(val):
            vision_inputs[key] = val.to(
                lm_device,
                dtype=(torch.float16 if key == "pixel_values" else None),
            )

    # ---- Build attended embeddings for pass-2 (done once per sample) --------
    inputs_embeds_att, att_mask_att = build_attended_embeds(
        model,
        vision_inputs,
        img_start,
        img_end,
        lam=lam,
        token_alpha=token_alpha,
    )
    inputs_embeds_att = inputs_embeds_att.to(lm_device, dtype=torch.float16)
    att_mask_att = att_mask_att.to(lm_device)
    s_merged: int = att_mask_att.shape[1]

    # ---- Prefill — Pass 1: IGAP off, standard pixel values ------------------
    _igap_toggle["active"] = False
    out1 = model(
        input_ids=vision_inputs["input_ids"],
        attention_mask=vision_inputs.get("attention_mask"),
        pixel_values=vision_inputs["pixel_values"],
        use_cache=True,
    )
    past_kv_1 = out1.past_key_values
    mask1 = torch.ones((1, s_merged), dtype=torch.long, device=lm_device)

    # ---- Prefill — Pass 2: IGAP on, attended embeddings --------------------
    _igap_toggle["active"] = True
    out2 = model(
        inputs_embeds=inputs_embeds_att,
        attention_mask=att_mask_att,
        use_cache=True,
    )
    past_kv_2 = out2.past_key_values
    mask2 = att_mask_att.clone()
    _igap_toggle["active"] = False

    # ---- Autoregressive decode loop -----------------------------------------
    generated_ids: List[int] = []
    trace: List[Tuple[int, float, str]] = []
    # Initialise next_id so that the step>0 branch never hits UnboundLocalError.
    # The value is overwritten at the end of step=0 before step=1 reads it.
    next_id: torch.Tensor = torch.zeros((1, 1), dtype=torch.long, device=lm_device)

    for step in range(max_new_tokens):

        if step == 0:
            # Reuse last-position logits from the prefill forward passes
            logits_orig: torch.Tensor = out1.logits[0, -1].float()
            logits_igap: torch.Tensor = out2.logits[0, -1].float()
        else:
            # ---- Decode pass 1 (IGAP off) ------------------------------------
            _igap_toggle["active"] = False
            d1 = model(
                input_ids=next_id,
                attention_mask=mask1,
                past_key_values=past_kv_1,
                use_cache=True,
            )
            logits_orig = d1.logits[0, -1].float()
            past_kv_1 = d1.past_key_values

            # ---- Decode pass 2 (IGAP on) -------------------------------------
            _igap_toggle["active"] = True
            d2 = model(
                input_ids=next_id,
                attention_mask=mask2,
                past_key_values=past_kv_2,
                use_cache=True,
            )
            logits_igap = d2.logits[0, -1].float()
            past_kv_2 = d2.past_key_values
            _igap_toggle["active"] = False

        # ---- Jensen-Shannon divergence ---------------------------------------
        eps: float = 1e-8
        p = F.softmax(logits_orig, dim=-1)
        q = F.softmax(logits_igap, dim=-1)
        m = 0.5 * (p + q)
        js: float = 0.5 * (
            torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
            + torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
        ).item()

        # ---- Visual confidence of the baseline distribution ------------------
        # Reuse `p` (already computed above) to avoid a redundant softmax call.
        base_probs = p
        max_prob: float = torch.max(base_probs).item()

        # ---- DCLA Routing ----------------------------------------------------
        if max_prob > CONFIDENCE_GATE:
            # Confidence gate: model is highly certain — trust baseline directly
            final_logits = logits_orig.clone()
            mode = "greedy"

        elif js <= js_threshold:
            # Complementary amplification: both passes agree — boost the signal
            final_logits = logits_orig + alpha1 * logits_igap
            mode = "complement"

        else:
            # Contrastive suppression: passes diverge — amplify orig, dampen igap
            final_logits = (1.0 + alpha2) * logits_orig - alpha2 * logits_igap

            # APC mask: zero out tokens below 10% of the max baseline probability
            apc_threshold: float = APC_FRACTION * max_prob
            plausible_mask = base_probs >= apc_threshold
            final_logits[~plausible_mask] = -float("inf")

            mode = "contrast"

        # ---- Greedy selection -----------------------------------------------
        next_id: torch.Tensor = torch.argmax(final_logits, dim=-1).view(1, 1)
        token_id: int = int(next_id.item())

        generated_ids.append(token_id)
        trace.append((token_id, round(js, 5), mode))

        # ---- Extend attention masks by 1 slot each step ---------------------
        ones = torch.ones((1, 1), dtype=torch.long, device=lm_device)
        mask1 = torch.cat([mask1, ones], dim=1)
        mask2 = torch.cat([mask2, ones], dim=1)

        if token_id == processor.tokenizer.eos_token_id:
            break

    # ---- Decode generated token ids → string --------------------------------
    answer_ids = torch.tensor([generated_ids], device=lm_device)
    answer: str = processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer, trace
