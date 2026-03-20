"""
Evaluation metrics for HNet training.

Metrics implemented:
- Bits-per-Byte (BPB): Primary metric for byte-level language models.
  Computed as total cross-entropy loss (in nats) / (num_bytes * ln(2)).
- Compression ratio metrics (F, G) from routing module outputs.
"""

import torch


def bits_per_byte(ce_loss_sum: torch.Tensor, num_bytes: int) -> float:
    """Compute Bits-per-Byte from total cross-entropy loss.

    BPB = total_nats / (num_bytes * ln(2))

    Args:
        ce_loss_sum: Sum of per-token cross-entropy losses (in nats, NOT averaged).
        num_bytes: Total number of target bytes evaluated.

    Returns:
        BPB as a float.
    """
    return (ce_loss_sum.item()) / (num_bytes * torch.log(torch.tensor(2.0)).item())


def compression_metrics(bpred_outputs: list) -> dict:
    """Compute compression ratio metrics from routing module outputs.

    Args:
        bpred_outputs: List of RoutingModuleOutput from each hierarchical stage.

    Returns:
        Dict with per-stage and averaged metrics:
        - F (fraction selected): proportion of vectors retained after downsampling
        - G (avg boundary prob): mean boundary probability across positions
    """
    if not bpred_outputs:
        return {}

    metrics = {}
    all_f = []
    all_g = []

    all_g_pos = []
    all_g_neg = []

    for i, router_out in enumerate(bpred_outputs):
        boundary_mask = router_out.boundary_mask
        bp = router_out.boundary_prob[..., -1].float()

        f_selected = boundary_mask.float().mean().item()
        g_avg_prob = bp.mean().item()

        pos_mask = boundary_mask.float()
        neg_mask = (~boundary_mask).float()
        g_pos = (bp * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        g_neg = (bp * neg_mask).sum() / neg_mask.sum().clamp(min=1)

        metrics[f"stage_{i}/F_selected"] = f_selected
        metrics[f"stage_{i}/G_avg_boundary_prob"] = g_avg_prob
        metrics[f"stage_{i}/G_boundary_prob_pos"] = g_pos.item()
        metrics[f"stage_{i}/G_boundary_prob_neg"] = g_neg.item()
        all_f.append(f_selected)
        all_g.append(g_avg_prob)
        all_g_pos.append(g_pos.item())
        all_g_neg.append(g_neg.item())

    if all_f:
        metrics["avg/F_selected"] = sum(all_f) / len(all_f)
        metrics["avg/G_avg_boundary_prob"] = sum(all_g) / len(all_g)
        metrics["avg/G_boundary_prob_pos"] = sum(all_g_pos) / len(all_g_pos)
        metrics["avg/G_boundary_prob_neg"] = sum(all_g_neg) / len(all_g_neg)

    return metrics
