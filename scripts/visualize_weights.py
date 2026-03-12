import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_train_mode(data):
    return "steps" in data


def savefig(fig, output_dir, name):
    path = Path(output_dir) / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Single-checkpoint plots
# ---------------------------------------------------------------------------

def plot_singular_value_spectrum(rm, depth, output_dir):
    """Bar chart of top-10 singular values for Q and K."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Singular value spectrum — depth {depth}")

    for ax, key in zip(axes, ["Q", "K"]):
        vals = rm[key]["singular_values"]["top10"]
        ax.bar(range(len(vals)), vals)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="identity baseline")
        ax.set_title(key)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Singular value")
        ax.legend()

    savefig(fig, output_dir, f"depth{depth}_singular_spectrum.png")


def plot_qk_alignment_heatmap(rm, depth, output_dir):
    """Heatmap of the 5×5 QK pairwise alignment matrix."""
    matrix = np.array(rm["QK_pairwise_alignment_top5"])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.suptitle(f"QK pairwise alignment (top-5) — depth {depth}")

    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("K singular vectors")
    ax.set_ylabel("Q singular vectors")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)

    savefig(fig, output_dir, f"depth{depth}_qk_alignment.png")


def plot_multiheaded_single(rm, depth, output_dir):
    """Per-head QK principal angles and cross-head overlap heatmap."""
    mh = rm["multiheaded"]
    heads = mh["heads"]
    num_heads = len(heads)

    # Per-head QK principal angles
    fig, ax = plt.subplots(figsize=(max(6, num_heads * 1.5), 4))
    fig.suptitle(f"Per-head QK principal angles — depth {depth}")

    x = np.arange(num_heads)
    for angle_idx in range(len(heads[0]["QK_principal_angles"])):
        vals = [h["QK_principal_angles"][angle_idx] for h in heads]
        ax.plot(x, vals, marker="o", label=f"angle {angle_idx + 1}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"head {h['head']}" for h in heads])
    ax.set_ylabel("Cosine of principal angle")
    ax.set_ylim(0, 1)
    ax.legend()

    savefig(fig, output_dir, f"depth{depth}_multiheaded_qk_angles.png")

    # Cross-head overlap heatmap
    overlap = mh["cross_head_overlap"]
    matrix = np.zeros((num_heads, num_heads))
    for key, vals in overlap.items():
        i, j = map(int, key.split("-"))
        matrix[i, j] = np.mean(vals)
        matrix[j, i] = np.mean(vals)
    np.fill_diagonal(matrix, 1.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.suptitle(f"Cross-head input subspace overlap (mean) — depth {depth}")
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Head j")
    ax.set_ylabel("Head i")

    for i in range(num_heads):
        for j in range(num_heads):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)

    savefig(fig, output_dir, f"depth{depth}_cross_head_overlap.png")


def visualize_single(data, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for rm in data["routing_modules"]:
        depth = rm["depth"]
        plot_singular_value_spectrum(rm, depth, output_dir)
        plot_qk_alignment_heatmap(rm, depth, output_dir)
        if "multiheaded" in rm:
            plot_multiheaded_single(rm, depth, output_dir)


# ---------------------------------------------------------------------------
# Training-mode plots (metrics over steps)
# ---------------------------------------------------------------------------

def _extract_train_series(data, depth, key_path):
    """Walk key_path (list of keys) into each step's routing_module at `depth`."""
    steps, vals = [], []
    for entry in data["steps"]:
        rm = entry["routing_modules"][depth]
        node = rm
        for k in key_path:
            node = node[k]
        steps.append(entry["step"])
        vals.append(node)
    return steps, vals


def plot_train_distances(data, depth, output_dir):
    """Frobenius and orthogonality distance from I over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Distance from identity over training — depth {depth}")

    for ax, metric, label in [
        (axes[0], "frobenius_distance_from_I", "||W - I||_F"),
        (axes[1], "orthogonality_distance",    "||W^T W - I||_F"),
    ]:
        for matrix in ["Q", "K"]:
            steps, vals = _extract_train_series(data, depth, [matrix, metric])
            ax.plot(steps, vals, marker="o", markersize=3, label=matrix)
        ax.set_title(label)
        ax.set_xlabel("Step")
        ax.legend()

    savefig(fig, output_dir, f"depth{depth}_train_distances.png")


def plot_train_effective_rank(data, depth, output_dir):
    """Effective rank (90/99/99.9%) over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Effective rank over training — depth {depth}")

    for ax, matrix in zip(axes, ["Q", "K"]):
        for pct, key in [("90%", "90pct"), ("99%", "99pct"), ("99.9%", "999pct")]:
            steps, vals = _extract_train_series(data, depth, [matrix, "effective_rank", key])
            ax.plot(steps, vals, marker="o", markersize=3, label=pct)
        ax.set_title(matrix)
        ax.set_xlabel("Step")
        ax.set_ylabel("Dimensions")
        ax.legend()

    savefig(fig, output_dir, f"depth{depth}_train_effective_rank.png")


def plot_train_singular_stats(data, depth, output_dir):
    """Min/mean/max singular values over training steps for Q and K."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Singular value statistics over training — depth {depth}")

    for ax, matrix in zip(axes, ["Q", "K"]):
        for stat in ["min", "mean", "max"]:
            steps, vals = _extract_train_series(data, depth, [matrix, "singular_values", stat])
            ax.plot(steps, vals, marker="o", markersize=3, label=stat)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="identity baseline")
        ax.set_title(matrix)
        ax.set_xlabel("Step")
        ax.set_ylabel("Singular value")
        ax.legend()

    savefig(fig, output_dir, f"depth{depth}_train_singular_stats.png")


def plot_train_qk_alignment(data, depth, output_dir):
    """Mean and min QK principal angle over training steps."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"QK principal angles over training — depth {depth}")

    steps = [e["step"] for e in data["steps"]]
    means, mins, maxs = [], [], []
    for entry in data["steps"]:
        angles = entry["routing_modules"][depth]["QK_principal_angles_top20"]
        means.append(np.mean(angles))
        mins.append(np.min(angles))
        maxs.append(np.max(angles))

    ax.plot(steps, means, marker="o", markersize=3, label="mean")
    ax.fill_between(steps, mins, maxs, alpha=0.2, label="min–max range")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine of principal angle")
    ax.set_ylim(0, 1)
    ax.legend()

    savefig(fig, output_dir, f"depth{depth}_train_qk_angles.png")


def plot_train_multiheaded(data, depth, output_dir):
    """Mean per-head QK alignment and cross-head overlap over training steps."""
    steps = [e["step"] for e in data["steps"]]
    num_heads = len(data["steps"][0]["routing_modules"][depth]["multiheaded"]["heads"])

    # Per-head mean QK angle over steps
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"Per-head mean QK principal angle over training — depth {depth}")

    for h in range(num_heads):
        vals = [
            np.mean(e["routing_modules"][depth]["multiheaded"]["heads"][h]["QK_principal_angles"])
            for e in data["steps"]
        ]
        ax.plot(steps, vals, marker="o", markersize=3, label=f"head {h}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean cosine of principal angle")
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, output_dir, f"depth{depth}_train_multiheaded_qk.png")

    # Cross-head overlap: mean over all pairs per step
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"Mean cross-head overlap over training — depth {depth}")

    overlap_keys = list(data["steps"][0]["routing_modules"][depth]["multiheaded"]["cross_head_overlap"].keys())
    for pair in overlap_keys:
        vals = [
            np.mean(e["routing_modules"][depth]["multiheaded"]["cross_head_overlap"][pair])
            for e in data["steps"]
        ]
        ax.plot(steps, vals, marker="o", markersize=3, label=f"heads {pair}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean cosine overlap")
    ax.set_ylim(0, 1)
    ax.legend()
    savefig(fig, output_dir, f"depth{depth}_train_cross_head_overlap.png")


def visualize_train(data, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_depths = len(data["steps"][0]["routing_modules"])

    for depth in range(num_depths):
        plot_train_distances(data, depth, output_dir)
        plot_train_effective_rank(data, depth, output_dir)
        plot_train_singular_stats(data, depth, output_dir)
        plot_train_qk_alignment(data, depth, output_dir)
        if "multiheaded" in data["steps"][0]["routing_modules"][depth]:
            plot_train_multiheaded(data, depth, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize routing module metrics from a JSON file.")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to the JSON file produced by inspect_weights.py.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write output plots into.")
    args = parser.parse_args()

    data = load_json(args.input_path)

    if is_train_mode(data):
        visualize_train(data, args.output_dir)
    else:
        visualize_single(data, args.output_dir)


if __name__ == "__main__":
    main()
