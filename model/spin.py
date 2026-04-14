import torch
import torch.nn.functional as F


def apply_spin_attention_mask(
    attn_probs,
    num_heads,
    img_start,
    img_end,
    keep_ratio=0.95,
    suppression_alpha=0.08,
):
    """
    Original SPIN logic: Suppress attention heads that do not focus on the image.
    """
    bsz = attn_probs.size(0)
    num_keep = max(1, int(round(keep_ratio * num_heads)))

    head_scores = attn_probs[:, :, -1, img_start:img_end].sum(dim=-1)
    _, keep_idx = torch.topk(head_scores, k=num_keep, dim=1)

    mask = torch.full(
        (bsz, num_heads),
        fill_value=float(suppression_alpha),
        dtype=attn_probs.dtype,
        device=attn_probs.device,
    )
    mask.scatter_(1, keep_idx, 1.0)
    return mask.view(bsz, 1, num_heads)
