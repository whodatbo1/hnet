"""Empirically validates the low-rank structure of the routing Q/K projections.

For each validation sequence:
  1. Feed input_ids through embeddings + encoder (forward hook captures output).
  2. Compute boundary predictions with the full Q and K weight matrices.
  3. Project the encoder output to the top-k right-singular subspace of Q (and K).
  4. Recompute boundary predictions with the projected input.
  5. Measure boundary-decision agreement between full and low-rank routing.

This answers: "does the routing module actually need the full d_model representation,
or do the top-k directions contain all the signal?"

Usage:
    python scripts/analyze_routing.py \\
        --model-path /scratch-shared/mivanov1/hnet/checkpoints/hnet_1stage_XL/hnet_1stage_XL.pt \\
        --config-path configs/hnet_1stage_XL.json \\
        --val-bin /scratch-shared/mivanov1/hnet/data/fineweb-edu-sample-10BT/val.bin \\
        --seq-len 2048 \\
        --n-batches 50 \\
        --ranks 1 2 3 5 10 20
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the project root is on sys.path so generate.py and hnet/ are importable
# regardless of which directory the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

def compute_svd(W: torch.Tensor):
    """Return (U, S, Vh) of W in float32."""
    return torch.linalg.svd(W.float())


def project_to_topk(h: torch.Tensor, Vh: torch.Tensor, k: int) -> torch.Tensor:
    """Project h onto the top-k right-singular subspace of a weight matrix.

    Args:
        h:   (..., d_model) encoder hidden states.
        Vh:  (d_model, d_model) right singular vectors (rows = singular vectors).
        k:   number of top singular directions to keep.

    Returns:
        h_proj: (..., d_model) — h with all but the top-k directions zeroed out.
    """
    V_topk = Vh[:k, :].T          # (d_model, k)  — top-k right singular vectors
    h_proj = (h @ V_topk) @ V_topk.T  # project onto the k-dim subspace
    return h_proj


# ---------------------------------------------------------------------------
# Routing computation (mirrors RoutingModule.forward, non-multiheaded)
# ---------------------------------------------------------------------------

def compute_boundary(h: torch.Tensor, Q_weight: torch.Tensor, K_weight: torch.Tensor):
    """Compute cosine-similarity boundary predictions from hidden states.

    Args:
        h:        (T, d_model) sequence of hidden states.
        Q_weight: (d_model, d_model) query projection weight.
        K_weight: (d_model, d_model) key projection weight.

    Returns:
        cos_sim:      (T-1,) cosine similarities between consecutive tokens.
        boundary_mask: (T-1,) bool — True where a boundary is predicted.
    """
    q = F.normalize(h[:-1] @ Q_weight.T, dim=-1)  # (T-1, d)
    k = F.normalize(h[1:]  @ K_weight.T, dim=-1)  # (T-1, d)
    cos_sim = (q * k).sum(dim=-1)                  # (T-1,)
    boundary_prob = (1 - cos_sim) / 2
    boundary_mask = boundary_prob > 0.5
    return cos_sim, boundary_mask


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(model_path, config_path, val_bin, seq_len, n_batches, ranks):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path} ...")
    model = load_from_pretrained(model_path, config_path)
    model.eval()

    # ---- Extract and pre-decompose Q/K weights ----
    Q_weight = model.backbone.routing_module.q_proj_layer.weight.float()  # (d, d)
    K_weight = model.backbone.routing_module.k_proj_layer.weight.float()

    print("Computing SVD of Q and K ...")
    _, S_q, Vh_q = compute_svd(Q_weight)
    _, S_k, Vh_k = compute_svd(K_weight)

    print(f"Q — top-5 singular values: {S_q[:5].tolist()}")
    print(f"K — top-5 singular values: {S_k[:5].tolist()}")

    # Energy table
    cumE_q = torch.cumsum(S_q**2, 0) / (S_q**2).sum()
    cumE_k = torch.cumsum(S_k**2, 0) / (S_k**2).sum()
    print(f"\nQ 90/99/99.9% energy: "
          f"{(cumE_q < 0.90).sum()+1} / {(cumE_q < 0.99).sum()+1} / {(cumE_q < 0.999).sum()+1} dims")
    print(f"K 90/99/99.9% energy: "
          f"{(cumE_k < 0.90).sum()+1} / {(cumE_k < 0.99).sum()+1} / {(cumE_k < 0.999).sum()+1} dims")

    # ---- Hook to capture encoder output ----
    captured = {}

    def _hook(module, input, output):
        # Isotropic.forward returns a tensor; just grab it.
        captured["encoder_out"] = output.detach().float()

    hook_handle = model.backbone.encoder.register_forward_hook(_hook)

    # ---- Load validation sequences from memmap ----
    data = np.memmap(val_bin, dtype=np.uint8, mode="r")
    chunk_len = seq_len + 1
    n_chunks = len(data) // chunk_len
    rng = np.random.default_rng(0)
    indices = rng.choice(n_chunks, size=min(n_batches, n_chunks), replace=False)

    # Accumulators:  rank -> list of per-sequence agreement rates
    agreement   = defaultdict(list)   # fraction of boundaries that agree
    cos_mae     = defaultdict(list)   # mean |cos_sim_full - cos_sim_proj|
    boundary_frac_full = []           # fraction of positions that are boundaries (full)

    print(f"\nRunning {len(indices)} validation sequences (seq_len={seq_len}) ...")
    with torch.no_grad():
        for idx in indices:
            start = int(idx) * chunk_len
            ids = torch.from_numpy(
                data[start : start + chunk_len].astype(np.int64)
            ).unsqueeze(0).to(device)   # (1, seq_len+1)

            input_ids = ids[:, :-1]     # (1, seq_len)

            # Forward pass — hook fires during backbone.encoder
            _ = model(input_ids, mask=torch.ones_like(input_ids, dtype=torch.bool))

            h = captured["encoder_out"].squeeze(0)   # (seq_len, d_model)

            # Full routing prediction
            cos_full, mask_full = compute_boundary(h, Q_weight, K_weight)
            boundary_frac_full.append(mask_full.float().mean().item())

            # Low-rank routing predictions for each rank k.
            # Q reads from h projected onto Q's top-k input subspace;
            # K reads from h projected onto K's top-k input subspace.
            for k in ranks:
                h_q = project_to_topk(h, Vh_q, k)
                h_k = project_to_topk(h, Vh_k, k)
                q_proj = F.normalize(h_q[:-1] @ Q_weight.T, dim=-1)
                k_proj = F.normalize(h_k[1:]  @ K_weight.T, dim=-1)
                cos_proj = (q_proj * k_proj).sum(-1)
                mask_proj = ((1 - cos_proj) / 2) > 0.5

                agree = (mask_full == mask_proj).float().mean().item()
                mae   = (cos_full - cos_proj).abs().mean().item()
                agreement[k].append(agree)
                cos_mae[k].append(mae)

    hook_handle.remove()

    # ---- Report ----
    print(f"\nMean boundary rate (full model): {np.mean(boundary_frac_full):.3f}\n")
    print(f"{'Rank':>6}  {'Boundary agreement':>20}  {'Cos-sim MAE':>12}")
    print("-" * 44)
    for k in ranks:
        ag  = np.mean(agreement[k])
        mae = np.mean(cos_mae[k])
        print(f"{k:>6}  {ag:>19.4f}  {mae:>12.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--val-bin",     required=True,
                        help="Path to val.bin from scripts/prepare_data.py")
    parser.add_argument("--seq-len",  type=int, default=2048)
    parser.add_argument("--n-batches",type=int, default=50)
    parser.add_argument("--ranks",    type=int, nargs="+", default=[1, 2, 3, 5, 10, 20])
    args = parser.parse_args()

    analyze(
        args.model_path,
        args.config_path,
        args.val_bin,
        args.seq_len,
        args.n_batches,
        args.ranks,
    )


if __name__ == "__main__":
    main()
