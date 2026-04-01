#!/usr/bin/env python3
"""Measure boundary-probability metrics of a pretrained HNet on FineWeb-Edu.

Loads a pretrained checkpoint, runs inference on a subset of the validation
(or training) split, and reports per-stage compression metrics:
  - F (fraction selected as boundary)
  - G (mean boundary probability)
  - G_pos / G_neg (mean boundary prob for selected / non-selected positions)
  - H_b (mean binary entropy of boundary probabilities — certainty)

Usage:
    python experiments/boundary_metrics.py \
        --model-config configs/hnet_1stage_S.json \
        --checkpoint checkpoints/train/train_hnet_1stage_S/latest.pt \
        --num-batches 200

    # With wandb logging
    python experiments/boundary_metrics.py \
        --model-config configs/hnet_1stage_S.json \
        --checkpoint checkpoints/train/train_hnet_1stage_S/latest.pt \
        --wandb-project hnet-boundary-metrics \
        --wandb-name "1stage_S_boundary"
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import ListConfig

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, RoutingConfig, HNetConfig
from hnet.utils.data import MemmapByteDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, config_path: str, device: str) -> HNetForCausalLM:
    """Load a pretrained HNet model from checkpoint."""
    with open(config_path, "r") as f:
        config = json.load(f)

    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    routing_cfg = RoutingConfig(**config.pop("routing_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, routing_cfg=routing_cfg)

    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)
    model.eval()

    # Resolve latest.pt pointer
    if os.path.basename(checkpoint_path) == "latest.pt":
        ptr = torch.load(checkpoint_path, map_location="cpu")
        step = ptr["step"]
        checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), f"model_step{step}.pt")
        print(f"Resolved latest.pt -> {checkpoint_path}")

    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    return model


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    """Compute element-wise binary entropy H_b(p) in bits."""
    p = p.float().clamp(1e-7, 1 - 1e-7)
    return -(p * p.log2() + (1 - p) * (1 - p).log2())


def compute_boundary_metrics(bpred_outputs: list) -> dict:
    """Compute detailed boundary metrics from routing outputs.

    Returns per-stage and averaged metrics including:
      F, G, G_pos, G_neg, H_b (binary entropy / certainty),
      boundary prob histogram stats (p10, p25, p50, p75, p90).
    """
    if not bpred_outputs:
        return {}

    metrics = {}
    all_stage_metrics = defaultdict(list)

    for i, router_out in enumerate(bpred_outputs):
        boundary_mask = router_out.boundary_mask
        p = router_out.boundary_prob[..., -1].float()

        f_selected = boundary_mask.float().mean().item()
        g_avg = p.mean().item()

        pos_mask = boundary_mask.float()
        neg_mask = (~boundary_mask).float()
        g_pos = (p * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        g_neg = (p * neg_mask).sum() / neg_mask.sum().clamp(min=1)

        h_b = binary_entropy(p).mean().item()

        # Percentiles of boundary probability distribution
        p_flat = p.reshape(-1)
        quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=p.device)
        pcts = torch.quantile(p_flat, quantiles)

        stage = {
            "F_selected": f_selected,
            "G_avg_boundary_prob": g_avg,
            "G_boundary_prob_pos": g_pos.item(),
            "G_boundary_prob_neg": g_neg.item(),
            "H_b_certainty": h_b,
            "p10": pcts[0].item(),
            "p25": pcts[1].item(),
            "p50": pcts[2].item(),
            "p75": pcts[3].item(),
            "p90": pcts[4].item(),
        }

        for k, v in stage.items():
            metrics[f"stage_{i}/{k}"] = v
            all_stage_metrics[k].append(v)

    for k, vals in all_stage_metrics.items():
        metrics[f"avg/{k}"] = sum(vals) / len(vals)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Measure boundary-probability metrics of a pretrained HNet")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (or latest.pt)")
    parser.add_argument("--data-dir", type=Path, default=Path("/scratch-shared/mivanov1/hnet/data"))
    parser.add_argument("--dataset", type=str, default="sample-10BT", help="Dataset subset (default: sample-10BT)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, args.model_config, device)
    print(f"Model loaded on {device}")

    # Set up data
    subset_dir = args.data_dir / f"fineweb-edu-{args.dataset}"
    bin_path = subset_dir / f"{args.split}.bin"
    if not bin_path.exists():
        print(f"Error: {bin_path} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    dataset = MemmapByteDataset(
        bin_path=bin_path,
        seq_len=args.seq_len,
        seed=args.seed,
        shuffle=False,
        max_samples=args.num_batches * args.batch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optional wandb
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    # Run inference and collect metrics
    print(f"Running inference on {args.num_batches} batches ({args.split} split) ...")
    all_metrics = defaultdict(list)
    total_ce_loss = 0.0
    total_bytes = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids, mask=None, targets=targets)

            ce_loss_sum = nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )

        total_ce_loss += ce_loss_sum.item()
        total_bytes += targets.numel()

        batch_metrics = compute_boundary_metrics(output.bpred_output)
        for k, v in batch_metrics.items():
            all_metrics[k].append(v)

        if args.wandb_project:
            import wandb
            wandb.log({"batch": batch_idx, **{f"batch/{k}": v for k, v in batch_metrics.items()}})

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            bpb = total_ce_loss / (total_bytes * math.log(2))
            print(f"  batch {batch_idx + 1}/{args.num_batches} | bpb={bpb:.4f}")

    # Aggregate
    bpb = total_ce_loss / (total_bytes * math.log(2))
    agg = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    agg["bpb"] = bpb
    agg["total_bytes"] = total_bytes
    agg["num_batches"] = min(batch_idx + 1, args.num_batches)

    # Print results
    print(f"\n{'='*70}")
    print(f"  Boundary Metrics  ({agg['num_batches']} batches, {total_bytes:,} bytes)")
    print(f"{'='*70}")
    print(f"  BPB: {bpb:.4f}")
    print()

    # Group by stage
    stages = sorted(set(k.split("/")[0] for k in agg if k.startswith("stage_")))
    metric_names = ["F_selected", "G_avg_boundary_prob", "G_boundary_prob_pos",
                    "G_boundary_prob_neg", "H_b_certainty", "p10", "p25", "p50", "p75", "p90"]

    # Header
    header = f"  {'':>12s}"
    for name in metric_names:
        header += f" | {name:>12s}"
    print(header)
    print(f"  {'-'*12}" + (" | " + "-"*12) * len(metric_names))

    for stage in stages:
        row = f"  {stage:>12s}"
        for name in metric_names:
            val = agg.get(f"{stage}/{name}", float("nan"))
            row += f" | {val:>12.4f}"
        print(row)

    if "avg/F_selected" in agg:
        row = f"  {'avg':>12s}"
        for name in metric_names:
            val = agg.get(f"avg/{name}", float("nan"))
            row += f" | {val:>12.4f}"
        print(row)

    print(f"{'='*70}")

    # Log final aggregated metrics to wandb
    if args.wandb_project:
        import wandb
        wandb.log({f"final/{k}": v for k, v in agg.items()})
        wandb.finish()

    # Save to JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
