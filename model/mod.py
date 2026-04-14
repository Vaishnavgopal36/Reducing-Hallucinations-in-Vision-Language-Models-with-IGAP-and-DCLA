import torch
import torch.nn.functional as F


def original_mod_routing(logits_standard, logits_reference, js_threshold=0.05, alpha=1.0):
    """
    Original Mixture of Decoding (MoD) logic using JS Divergence.
    Routes between standard decoding and contrastive decoding.
    """
    eps = 1e-8
    p = F.softmax(logits_standard, dim=-1)
    q = F.softmax(logits_reference, dim=-1)
    m = 0.5 * (p + q)

    js = 0.5 * (
        torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
        + torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
    ).item()

    if js <= js_threshold:
        return logits_standard.clone(), "standard"

    final_logits = (1 + alpha) * logits_standard - alpha * logits_reference
    return final_logits, "contrastive"
