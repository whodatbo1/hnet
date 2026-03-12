import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch

from generate import load_from_pretrained
from hnet.utils.data import create_dataloaders

torch.set_printoptions(sci_mode=False)


# ---------------------------------------------------------------------------
# Compute functions — return structured dicts, no printing
# ---------------------------------------------------------------------------

def _r(tensor, decimals=3):
    """Round tensor values to `decimals` places and convert to a Python list."""
    return [round(v, decimals) for v in tensor.tolist()]

def _compute_matrix_metrics(A):
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    U, S, V = torch.linalg.svd(A)

    cumulative_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()

    return {
        "singular_values": {
            "min": round(S.min().item(), 3),
            "max": round(S.max().item(), 3),
            "mean": round(S.mean().item(), 3),
            "std": round(S.std().item(), 3),
            "top10": _r(S[:10]),
            "num_deviating_01": int(((S - 1).abs() > 0.1).sum().item()),
            "num_dead_001": int((S < 0.01).sum().item()),
        },
        "frobenius_distance_from_I": round(torch.norm(A - I, p="fro").item(), 3),
        "orthogonality_distance": round(torch.norm(A.T @ A - I, p="fro").item(), 3),
        "effective_rank": {
            "90pct": int((cumulative_energy < 0.90).sum().item() + 1),
            "99pct": int((cumulative_energy < 0.99).sum().item() + 1),
            "999pct": int((cumulative_energy < 0.999).sum().item() + 1),
        },
    }


def _compute_multiheaded_metrics(W_q, W_k, num_heads, head_dim):
    heads = []
    for h in range(num_heads):
        W_q_h = W_q[h * head_dim:(h + 1) * head_dim, :]
        W_k_h = W_k[h * head_dim:(h + 1) * head_dim, :]

        U_q, S_q, Vh_q = torch.linalg.svd(W_q_h)
        U_k, S_k, Vh_k = torch.linalg.svd(W_k_h)

        k_top = min(3, head_dim)
        cos_angles = torch.linalg.svdvals(U_q[:, :k_top].T @ U_k[:, :k_top])
        heads.append({
            "head": h,
            "singular_values_Q": _r(S_q[:5]),
            "singular_values_K": _r(S_k[:5]),
            "QK_principal_angles": _r(cos_angles),
        })

    cross_head_overlap = {}
    k_top = min(3, head_dim)
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            _, _, Vh_i = torch.linalg.svd(W_q[i * head_dim:(i + 1) * head_dim, :])
            _, _, Vh_j = torch.linalg.svd(W_q[j * head_dim:(j + 1) * head_dim, :])
            cos_angles = torch.linalg.svdvals(Vh_i[:k_top, :] @ Vh_j[:k_top, :].T)
            cross_head_overlap[f"{i}-{j}"] = _r(cos_angles)

    return {"heads": heads, "cross_head_overlap": cross_head_overlap}


def _compute_routing_module_metrics(Q, K, routing_cfg):
    k = 20
    U_q, S_q, _ = torch.linalg.svd(Q)
    U_k, S_k, _ = torch.linalg.svd(K)

    cos_angles_top20 = torch.linalg.svdvals(U_q[:, :k].T @ U_k[:, :k])

    k = 5
    alignment = (U_q[:, :k].T @ U_k[:, :k]).abs()

    result = {
        "Q": _compute_matrix_metrics(Q),
        "K": _compute_matrix_metrics(K),
        "QK_principal_angles_top20": _r(cos_angles_top20),
        "QK_pairwise_alignment_top5": [_r(row) for row in alignment],
    }

    if routing_cfg.multiheaded:
        result["multiheaded"] = _compute_multiheaded_metrics(Q, K, routing_cfg.num_heads, routing_cfg.num_heads and Q.shape[0] // routing_cfg.num_heads)

    return result


# ---------------------------------------------------------------------------
# Print functions — call compute functions and pretty-print
# ---------------------------------------------------------------------------

def get_metrics_for_matrix(A):
    m = _compute_matrix_metrics(A)
    sv = m["singular_values"]
    print(f"Singular values — min: {sv['min']:.4f}, max: {sv['max']:.4f}, mean: {sv['mean']:.4f}, std: {sv['std']:.4f}")
    print(f"Singular values deviating >0.1 from 1: {sv['num_deviating_01']} / {A.shape[0]}")
    print(f"Distance from orthogonality (||A^T A - I||_F): {m['orthogonality_distance']:.4f}")
    er = m["effective_rank"]
    print(f"Dimensions for 90% energy: {er['90pct']} / {A.shape[0]}")
    print(f"Dimensions for 99% energy: {er['99pct']} / {A.shape[0]}")
    print(f"Dimensions for 99.9% energy: {er['999pct']} / {A.shape[0]}")
    print(f"Top 10 singular values: {sv['top10']}")
    print(f"Singular values < 0.01: {sv['num_dead_001']} / {A.shape[0]}")
    print(f"Frobenius norm: {m['frobenius_distance_from_I']:.6f}")


def compute_multiheaded_metrics(W_q, W_k, num_heads, head_dim):
    m = _compute_multiheaded_metrics(W_q, W_k, num_heads, head_dim)
    for h_data in m["heads"]:
        h = h_data["head"]
        print(f"Head {h}:")
        print(f"  Q singular values: {h_data['singular_values_Q']}")
        print(f"  K singular values: {h_data['singular_values_K']}")
        print(f"  Q/K principal angles (top-{min(3, head_dim)}): {h_data['QK_principal_angles']}")
    for pair, overlap in m["cross_head_overlap"].items():
        print(f"  Heads {pair} input subspace overlap: {overlap}")


def get_metrics(Q, K):
    k = 20
    U_q, S_q, _ = torch.linalg.svd(Q)
    U_k, S_k, _ = torch.linalg.svd(K)

    cos_angles = torch.linalg.svdvals(U_q[:, :k].T @ U_k[:, :k])
    print(f"Cosine of principal angles (top-{k}): {_r(cos_angles)}")

    k = 5
    alignment = (U_q[:, :k].T @ U_k[:, :k]).abs()
    print("Pairwise alignment (|cos| between individual singular vectors):")
    print(alignment)

    print('*' * 55)
    print("Metrics for Q")
    get_metrics_for_matrix(Q)

    print('*' * 55)
    print("Metrics for K")
    get_metrics_for_matrix(K)


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def collect_routing_modules(hnet):
    """Recursively collect all RoutingModules from an HNet hierarchy, outermost first."""
    if hnet.is_innermost:
        return []
    return [hnet.routing_module] + collect_routing_modules(hnet.main_network)


def _collect_metrics_for_model(model):
    routing_cfg = model.config.routing_cfg
    routing_modules = collect_routing_modules(model.backbone)

    result = []
    for depth, rm in enumerate(routing_modules):
        Q = rm.q_proj_layer.weight.type(torch.float32)
        K = rm.k_proj_layer.weight.type(torch.float32)
        metrics = _compute_routing_module_metrics(Q, K, routing_cfg)
        metrics["depth"] = depth
        result.append(metrics)
    return result


def _print_metrics_for_model(model):
    routing_cfg = model.config.routing_cfg
    routing_modules = collect_routing_modules(model.backbone)

    for depth, rm in enumerate(routing_modules):
        Q = rm.q_proj_layer.weight.type(torch.float32)
        K = rm.k_proj_layer.weight.type(torch.float32)
        print('=' * 55)
        print(f"Metrics for routing module at depth {depth}")
        get_metrics(Q, K)
        if routing_cfg.multiheaded:
            print('*' * 55)
            print(f"Multiheaded metrics for routing module at depth {depth}")
            compute_multiheaded_metrics(Q, K, rm.num_heads, rm.head_dim)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_metrics_for_model(
    model_path: str,
    model_config_path: str,
    train: bool,
    output_path: str = None,
):
    if not train:
        model = load_from_pretrained(
            model_path=model_path,
            model_config_path=model_config_path,
        )
        _print_metrics_for_model(model)

        if output_path is not None:
            data = {
                "model_path": model_path,
                "model_config_path": model_config_path,
                "routing_modules": _collect_metrics_for_model(model),
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(data, indent=2))
            print(f"Metrics written to {output_path}")
    else:
        all_steps = []
        for step in range(500, 10500, 500):
            print(f"Loading train weights for step: {step}")
            try:
                model = load_from_pretrained(
                    model_path=model_path.format(step=step),
                    model_config_path=model_config_path,
                )
            except FileNotFoundError:
                print("Couldn't find file. Aborting and saving")
                break
            print('-' * 50)
            print(f"Metrics for step {step}")
            _print_metrics_for_model(model)

            if output_path is not None:
                all_steps.append({
                    "step": step,
                    "routing_modules": _collect_metrics_for_model(model),
                })

        if output_path is not None:
            data = {
                "model_path_template": model_path,
                "model_config_path": model_config_path,
                "steps": all_steps,
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(data, indent=2))
            print(f"Metrics written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect routing module weights of an HNet model.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint. In train mode, use {step} as a placeholder (e.g. model_step{step}.pt).")
    parser.add_argument("--model-config-path", type=str, required=True,
                        help="Path to model config JSON.")
    parser.add_argument("--train", action="store_true",
                        help="Loop over training checkpoints (steps 500..10000).")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional path to write metrics as JSON.")
    args = parser.parse_args()

    extract_metrics_for_model(
        model_path=args.model_path,
        model_config_path=args.model_config_path,
        train=args.train,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()