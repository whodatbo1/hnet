"""Analyze utilization of pad dimensions in multi-stage HNet models.

For each HNet stage that has a pad_dimension, runs validation data through the
model and measures how much the padded dimensions contribute relative to the
original dimensions.

Metrics reported per stage:
  - std ratio: std(padded dims) / std(original dims)
  - mean abs: mean absolute activation in padded vs original dims
  - grad norm: gradient magnitude flowing into pad_dimension parameter

Usage:
    python scripts/analyze_pad_dims.py \
        --model-path checkpoints/latest.pt \
        --config-path configs/model.json \
        --val-bin data/fineweb-edu-sample-10BT/val.bin \
        --seq-len 2048 \
        --n-batches 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained


def find_pad_stages(model):
    """Find all HNet stages that have a pad_dimension parameter.

    Returns list of (name, module, d_outer) tuples.
    """
    stages = []
    for name, module in model.named_modules():
        if hasattr(module, "pad_dimension") and module.pad_dimension is not None:
            d_inner = module.d_model
            d_pad = module.pad_dimension.shape[0]
            d_outer = d_inner - d_pad
            stages.append((name, module, d_outer))
    return stages


def analyze(model_path, config_path, val_bin, seq_len, n_batches):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path} ...")
    model = load_from_pretrained(model_path, config_path)

    pad_stages = find_pad_stages(model)
    if not pad_stages:
        print("No pad_dimension found in this model (single-stage or same d_model across stages).")
        return

    print(f"\nFound {len(pad_stages)} stage(s) with pad_dimension:")
    for name, module, d_outer in pad_stages:
        d_inner = module.d_model
        d_pad = d_inner - d_outer
        print(f"  {name}: d_outer={d_outer}, d_inner={d_inner}, pad={d_pad}")

    # Register hooks to capture hidden states right after pad_dimension concat
    captured = {}

    def make_hook(stage_name, d_outer):
        def hook(module, input, output):
            # HNet.forward: after pad, hidden_states goes into encoder or main_network
            # We hook the encoder to get the padded input to the inner stage
            h = input[0] if isinstance(input, tuple) else input
            if h.dim() >= 2 and h.shape[-1] > d_outer:
                captured[stage_name] = h.detach()
        return hook

    hooks = []
    for name, module, d_outer in pad_stages:
        # Hook the encoder of this stage — its input is the padded hidden states
        if hasattr(module, "encoder"):
            h = module.encoder.register_forward_hook(make_hook(name, d_outer))
            hooks.append(h)
        elif hasattr(module, "main_network"):
            h = module.main_network.register_forward_hook(make_hook(name, d_outer))
            hooks.append(h)

    # Load validation data
    data = np.memmap(val_bin, dtype=np.uint8, mode="r")
    chunk_len = seq_len + 1
    n_chunks = len(data) // chunk_len
    rng = np.random.default_rng(42)
    indices = rng.choice(n_chunks, size=min(n_batches, n_chunks), replace=False)

    # Accumulators
    stats = {name: {"orig_std": [], "pad_std": [], "orig_abs": [], "pad_abs": []}
             for name, _, _ in pad_stages}

    print(f"\nRunning {len(indices)} validation sequences (seq_len={seq_len}) ...")

    # First pass: activation statistics (no grad)
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            start = int(idx) * chunk_len
            ids = torch.from_numpy(
                data[start: start + chunk_len].astype(np.int64)
            ).unsqueeze(0).to(device)
            input_ids = ids[:, :-1]

            captured.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(input_ids, mask=torch.ones_like(input_ids, dtype=torch.bool))

            for name, _, d_outer in pad_stages:
                if name not in captured:
                    continue
                h = captured[name].float()
                orig = h[..., :d_outer]
                pad = h[..., d_outer:]
                stats[name]["orig_std"].append(orig.std().item())
                stats[name]["pad_std"].append(pad.std().item())
                stats[name]["orig_abs"].append(orig.abs().mean().item())
                stats[name]["pad_abs"].append(pad.abs().mean().item())

    # Second pass: gradient magnitude (single batch, with grad)
    grad_norms = {}
    model.zero_grad()
    idx = indices[0]
    start = int(idx) * chunk_len
    ids = torch.from_numpy(
        data[start: start + chunk_len].astype(np.int64)
    ).unsqueeze(0).to(device)
    input_ids = ids[:, :-1]
    targets = ids[:, 1:]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, mask=torch.ones_like(input_ids, dtype=torch.bool))
        loss = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            targets.reshape(-1),
        )
    loss.backward()

    for name, module, d_outer in pad_stages:
        if module.pad_dimension.grad is not None:
            grad_norms[name] = module.pad_dimension.grad.norm().item()
        else:
            grad_norms[name] = 0.0

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Report
    print("\n" + "=" * 80)
    print("PAD DIMENSION ANALYSIS")
    print("=" * 80)

    for name, module, d_outer in pad_stages:
        d_inner = module.d_model
        d_pad = d_inner - d_outer
        s = stats[name]

        orig_std = np.mean(s["orig_std"])
        pad_std = np.mean(s["pad_std"])
        orig_abs = np.mean(s["orig_abs"])
        pad_abs = np.mean(s["pad_abs"])

        print(f"\n  Stage: {name}")
        print(f"  Dimensions: original={d_outer}, padded={d_pad}")
        print(f"  pad_dimension values: {module.pad_dimension.data.tolist()[:8]}{'...' if d_pad > 8 else ''}")
        print(f"  pad_dimension norm: {module.pad_dimension.data.norm():.6f}")
        print(f"")
        print(f"  {'Metric':<25} {'Original dims':>15} {'Padded dims':>15} {'Ratio':>10}")
        print(f"  {'-'*65}")
        print(f"  {'Activation std':<25} {orig_std:>15.6f} {pad_std:>15.6f} {pad_std/(orig_std+1e-10):>10.4f}")
        print(f"  {'Activation mean |x|':<25} {orig_abs:>15.6f} {pad_abs:>15.6f} {pad_abs/(orig_abs+1e-10):>10.4f}")
        print(f"  {'Grad norm (1 batch)':<25} {'':>15} {grad_norms[name]:>15.6f}")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  - Ratio close to 0: padded dimensions are barely used")
    print("  - Ratio close to 1: padded dimensions have similar activation magnitudes")
    print("  - High grad norm: the model is still actively learning pad values")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze utilization of pad dimensions in multi-stage HNet"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to model config (.json file)")
    parser.add_argument("--val-bin", type=str, required=True,
                        help="Path to val.bin from scripts/prepare_data.py")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-batches", type=int, default=50)
    args = parser.parse_args()

    analyze(args.model_path, args.config_path, args.val_bin, args.seq_len, args.n_batches)


if __name__ == "__main__":
    main()
