"""
Visualize next-byte prediction entropy from the encoder's BMHead.

For each byte position in the input, computes the entropy (in bits) of the
BMHead's next-byte distribution after the encoder. High entropy = the model
is uncertain about what comes next (likely a boundary).

Usage:
    python scripts/visualize_entropy.py \
        --model-path checkpoints/latest.pt \
        --config-path checkpoints/config.json \
        --prompt "The capital of Brazil is "

    # Save a matplotlib figure instead of terminal output
    python scripts/visualize_entropy.py \
        --model-path checkpoints/latest.pt \
        --config-path checkpoints/config.json \
        --prompt "The capital of Brazil is " \
        --output-plot entropy.png
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


def compute_entropy(model, input_ids: torch.Tensor):
    """Run the encoder + BMHead and return per-position entropy in bits.

    Args:
        model: HNetForCausalLM with entropy_routing enabled.
        input_ids: (1, L) long tensor of byte token ids.

    Returns:
        entropies: (L,) float tensor — raw entropy in bits at each position.
        boundary_probs: (L,) float tensor — boundary probability after sigmoid.
        top_preds: list of (byte_value, probability) for the top-1 prediction.
        boundary_mask: (L,) bool tensor — which positions are boundaries.
        routing_params: dict with entropy_mean, entropy_std, threshold, temperature.
    """
    device = input_ids.device
    backbone = model.backbone  # HNet (outermost stage)
    rm = backbone.routing_module

    assert hasattr(backbone, "routing_module"), "Model must have a routing module"
    assert rm.entropy_routing, "Routing module must use entropy routing"

    hidden_states = model.embeddings(input_ids)  # (1, L, D)
    B, L, D = hidden_states.shape

    # Run in padded mode (mask-based) for simplicity with batch=1
    mask = torch.ones(B, L, device=device, dtype=torch.bool)

    # Run encoder
    hidden_states = backbone.encoder(
        hidden_states,
        mask=mask,
    )

    # Apply BMHead
    bm_logits = rm.bm_head(hidden_states)  # (1, L, vocab_size)
    log_probs = bm_logits.log_softmax(-1)

    # Entropy in bits: H = -sum(p * log2(p))
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(-1) / math.log(2)  # (1, L)
    entropies = entropies.squeeze(0)  # (L,)

    # Normalized entropy signal + boundary probability (mirrors dc.py routing)
    entropy_mean = rm.entropy_mean.detach()
    entropy_std = rm.entropy_std.detach()
    temperature = rm.log_temperature.exp().detach()
    threshold = rm.entropy_threshold.detach()

    entropy_signal = (entropies - entropy_mean) / (entropy_std + 1e-6)
    pre_sigmoid = (entropy_signal - threshold) / temperature
    boundary_probs = torch.sigmoid(pre_sigmoid)

    # Top-1 predictions
    top_probs, top_ids = probs.squeeze(0).max(dim=-1)  # (L,)
    top_preds = list(zip(top_ids.tolist(), top_probs.tolist()))

    # Also run routing to get the boundary mask
    routing_out = rm(hidden_states, mask=mask)
    boundary_mask = routing_out.boundary_mask.squeeze(0)  # (L,)

    routing_params = {
        "entropy_mean": entropy_mean.item(),
        "entropy_std": entropy_std.item(),
        "threshold": threshold.item(),
        "temperature": temperature.item(),
    }

    return entropies, boundary_probs, pre_sigmoid, top_preds, boundary_mask, routing_params


def byte_to_display(b: int) -> str:
    """Convert a byte value to a displayable string."""
    if 32 <= b <= 126:
        return chr(b)
    elif b == 10:
        return "\\n"
    elif b == 9:
        return "\\t"
    elif b == 13:
        return "\\r"
    elif b == 0:
        return "\\0"
    else:
        return f"x{b:02x}"


def colorize(text: str, entropy: float, max_entropy: float = 8.0) -> str:
    """Color text by entropy using ANSI 256-color: blue (low) -> red (high)."""
    t = min(entropy / max_entropy, 1.0)
    # Interpolate from blue (low entropy) through green to red (high entropy)
    if t < 0.5:
        # blue -> green
        s = t * 2
        r, g, b = int(0 + s * 0), int(100 + s * 155), int(255 - s * 255)
    else:
        # green -> red
        s = (t - 0.5) * 2
        r, g, b = int(0 + s * 255), int(255 - s * 155), int(0)
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def print_colored_sequence(bytes_list, entropies, boundary_mask, top_preds):
    """Print the input sequence with entropy-colored bytes."""
    max_e = 8.0  # log2(256)

    # Header
    print("\n" + "=" * 80)
    print("Entropy-colored byte sequence (blue=low, red=high entropy)")
    print("Boundaries marked with |")
    print("=" * 80 + "\n")

    line = ""
    line_len = 0
    for i, (b, e, is_boundary) in enumerate(zip(bytes_list, entropies, boundary_mask)):
        disp = byte_to_display(b)
        if is_boundary and i > 0:
            line += "\033[90m|\033[0m"
            line_len += 1
        colored = colorize(disp, e, max_e)
        line += colored
        line_len += len(disp)
        if line_len > 100 or disp == "\\n":
            print(line)
            line = ""
            line_len = 0
    if line:
        print(line)

    # Detailed table
    print("\n" + "=" * 80)
    print(f"{'Pos':>4} {'Byte':>5} {'Display':>7} {'Entropy':>8} {'Top Pred':>9} {'Top Prob':>9} {'Boundary':>9}")
    print("-" * 80)
    for i, (b, e, (pred_b, pred_p), is_boundary) in enumerate(
        zip(bytes_list, entropies, top_preds, boundary_mask)
    ):
        disp = byte_to_display(b)
        pred_disp = byte_to_display(pred_b)
        marker = " *" if is_boundary else ""
        correct = " <-" if pred_b == b else ""
        # Only mark "correct" for positions > 0 where the prediction matches the
        # NEXT byte (since bm_head predicts next byte)
        if i < len(bytes_list) - 1 and pred_b == bytes_list[i + 1]:
            correct = " <-next"
        else:
            correct = ""
        print(
            f"{i:4d} {b:5d} {disp:>7s} {e:8.3f} {pred_disp:>9s} {pred_p:9.4f}{marker}{correct}"
        )

    print(f"\nMean entropy: {sum(entropies) / len(entropies):.3f} bits")
    print(f"Boundaries: {sum(boundary_mask)}/{len(boundary_mask)} ({sum(boundary_mask)/len(boundary_mask)*100:.1f}%)")


def _add_byte_labels(ax, bytes_list, positions):
    """Add byte character labels on the x-axis."""
    labels = [byte_to_display(b) for b in bytes_list]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=max(4, min(8, 800 // len(bytes_list))),
                       rotation=0, fontfamily="monospace")


def save_plot(bytes_list, entropies, boundary_mask, routing_params, output_path):
    """Save a matplotlib figure of raw entropy over the sequence."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(12, len(bytes_list) * 0.15), 5))

    positions = np.arange(len(entropies))
    colors = ["red" if bm else "steelblue" for bm in boundary_mask]
    ax.bar(positions, entropies, color=colors, width=1.0, edgecolor="none")

    _add_byte_labels(ax, bytes_list, positions)

    ax.set_ylabel("Entropy (bits)")
    ax.set_xlabel("Byte position")
    ax.set_title("Next-byte prediction entropy from BMHead")
    ax.set_ylim(0, 8.5)
    ax.axhline(y=entropies.mean(), color="gray", linestyle="--", alpha=0.5,
               label=f"mean={entropies.mean():.2f}")
    ax.axhline(y=routing_params["entropy_mean"], color="orange", linestyle=":",
               alpha=0.7, label=f"EMA \u03bc={routing_params['entropy_mean']:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def save_boundary_prob_plot(bytes_list, boundary_probs, boundary_mask, routing_params, output_path):
    """Save a matplotlib figure of the normalized entropy signal / boundary probability."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(12, len(bytes_list) * 0.15), 5))

    positions = np.arange(len(boundary_probs))
    colors = ["red" if bm else "steelblue" for bm in boundary_mask]
    ax.bar(positions, boundary_probs, color=colors, width=1.0, edgecolor="none")

    _add_byte_labels(ax, bytes_list, positions)

    ax.set_ylabel("Boundary probability")
    ax.set_xlabel("Byte position")
    ax.set_title(
        f"Boundary probability  "
        f"(EMA \u03bc={routing_params['entropy_mean']:.2f}, "
        f"\u03c3={routing_params['entropy_std']:.2f}, "
        f"threshold={routing_params['threshold']:.2f}, "
        f"temperature={routing_params['temperature']:.3f})"
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="decision boundary (0.5)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def save_pre_sigmoid_plot(bytes_list, pre_sigmoid, boundary_mask, routing_params, output_path):
    """Save a matplotlib figure of the pre-sigmoid logits (before applying sigmoid)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(12, len(bytes_list) * 0.15), 5))

    positions = np.arange(len(pre_sigmoid))
    colors = ["red" if bm else "steelblue" for bm in boundary_mask]
    ax.bar(positions, pre_sigmoid, color=colors, width=1.0, edgecolor="none")

    _add_byte_labels(ax, bytes_list, positions)

    ax.set_ylabel("(z - threshold) / temperature")
    ax.set_xlabel("Byte position")
    ax.set_title("Pre-sigmoid boundary logit")
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5,
               label=f"decision boundary (0.0)")
    ax.legend([
        f"threshold={routing_params['threshold']:.3f}",
        f"temperature={routing_params['temperature']:.4f}",
        "decision boundary (0.0)",
    ], fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize next-byte prediction entropy from the encoder's BMHead"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config-path", type=str, required=True,
        help="Path to the model configuration (.json file)",
    )
    parser.add_argument(
        "--prompt", type=str, default="The capital of Brazil is ",
        help="Input text to analyze",
    )
    parser.add_argument(
        "--output-plot", type=str, default=None,
        help="If set, save a matplotlib bar chart to this path (e.g. entropy.png)",
    )

    args = parser.parse_args()

    print("Loading model...")
    model = load_from_pretrained(args.model_path, args.config_path)
    print("Model loaded.\n")

    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode([args.prompt], add_bos=True)[0]
    input_ids = torch.tensor(
        encoded["input_ids"], dtype=torch.long, device=next(model.parameters()).device
    ).unsqueeze(0)

    bytes_list = encoded["input_ids"].tolist()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        entropies, boundary_probs, pre_sigmoid, top_preds, boundary_mask, routing_params = compute_entropy(model, input_ids)

    entropies = entropies.float().cpu()
    boundary_probs = boundary_probs.float().cpu()
    pre_sigmoid = pre_sigmoid.float().cpu()
    boundary_mask = boundary_mask.cpu().tolist()

    print_colored_sequence(bytes_list, entropies.tolist(), boundary_mask, top_preds)

    print(f"\nRouting params: EMA mean={routing_params['entropy_mean']:.3f}, "
          f"std={routing_params['entropy_std']:.3f}, "
          f"threshold={routing_params['threshold']:.3f}, "
          f"temperature={routing_params['temperature']:.4f}")

    if args.output_plot:
        p = Path(args.output_plot)
        save_plot(bytes_list, entropies.numpy(), boundary_mask, routing_params,
                  args.output_plot)
        save_boundary_prob_plot(bytes_list, boundary_probs.numpy(), boundary_mask,
                                routing_params,
                                str(p.with_name(p.stem + "_boundary_prob" + p.suffix)))
        save_pre_sigmoid_plot(bytes_list, pre_sigmoid.numpy(), boundary_mask,
                              routing_params,
                              str(p.with_name(p.stem + "_pre_sigmoid" + p.suffix)))


if __name__ == "__main__":
    main()
