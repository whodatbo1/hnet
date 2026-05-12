"""
Visualize predicted patch boundaries for any H-Net routing implementation.

Runs the full model forward and reads boundary predictions from the
`RoutingModuleOutput` returned by each stage's routing module. Because every
routing variant (cosine-similarity, entropy, multiheaded, random, space-like,
identity, single-projection, bm-head-cos) produces the same output shape, this
script is router-agnostic — it does not look at routing-mode-specific internals
like BMHead logits or pre-sigmoid signals.

Usage:
    python scripts/visualize_boundaries.py \\
        --model-path checkpoints/latest.pt \\
        --config-path checkpoints/config.json \\
        --prompt "The capital of Brazil is "

    # Save a matplotlib figure
    python scripts/visualize_boundaries.py \\
        --model-path checkpoints/latest.pt \\
        --config-path checkpoints/config.json \\
        --prompt "The capital of Brazil is " \\
        --output-plot boundaries.png

    # For hierarchical (multi-stage) models, the outermost (byte-level) stage
    # is plotted by default. Use --stage to inspect deeper stages.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate import load_from_pretrained
from hnet.utils.byte_tokenizer import ByteTokenizer


def compute_boundaries(model, input_ids: torch.Tensor):
    """Run the full model and return per-stage boundary predictions.

    Args:
        model: HNetForCausalLM.
        input_ids: (1, L) long tensor of byte token ids.

    Returns:
        stages: list of dicts, one per hierarchical stage (outer -> inner). Each
            dict has:
              - boundary_prob: (T,) float tensor — P(position is a boundary).
              - boundary_mask: (T,) bool tensor — which positions were selected.
              - stage_idx: int.
    """
    device = input_ids.device
    B, L = input_ids.shape
    mask = torch.ones(B, L, device=device, dtype=torch.bool)

    out = model(input_ids, mask=mask)
    bpred_list = out.bpred_output  # outer-most stage first

    stages = []
    for bp in bpred_list:
        # boundary_prob: (B, T, 2) -> P(boundary) at last dim index 1
        p = bp.boundary_prob[..., 1].squeeze(0).float().cpu()
        m = bp.boundary_mask.squeeze(0).cpu()
        stages.append({
            "boundary_prob": p,
            "boundary_mask": m,
            "stage_idx": bp.stage_idx,
        })
    return stages


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


def decode_utf8_labels(bytes_list):
    """Group bytes into UTF-8 characters for display.

    Returns a list `labels` of the same length as `bytes_list`. The lead byte
    of each UTF-8 character carries the decoded character; continuation bytes
    carry the empty string. Bytes that don't form a valid UTF-8 sequence fall
    back to per-byte hex.
    """
    labels = [""] * len(bytes_list)
    i = 0
    n = len(bytes_list)
    while i < n:
        b = bytes_list[i]
        if b < 0x80:
            labels[i] = byte_to_display(b)
            i += 1
            continue
        if b < 0xC2:
            labels[i] = f"x{b:02x}"
            i += 1
            continue
        if b < 0xE0:
            need = 2
        elif b < 0xF0:
            need = 3
        elif b < 0xF8:
            need = 4
        else:
            labels[i] = f"x{b:02x}"
            i += 1
            continue
        if i + need <= n:
            chunk = bytes(bytes_list[i:i + need])
            try:
                labels[i] = chunk.decode("utf-8")
                i += need
                continue
            except UnicodeDecodeError:
                pass
        labels[i] = f"x{b:02x}"
        i += 1
    return labels


def char_starts(labels):
    """Indices of byte positions that begin a (multi-byte) character."""
    return [i for i, s in enumerate(labels) if s != ""]


def colorize(text: str, intensity: float) -> str:
    """Color text by a [0, 1] intensity. Low = steel blue, high = red."""
    t = max(0.0, min(intensity, 1.0))
    if t < 0.5:
        s = t * 2
        r, g, b = int(0 + s * 0), int(100 + s * 155), int(255 - s * 255)
    else:
        s = (t - 0.5) * 2
        r, g, b = int(0 + s * 255), int(255 - s * 155), int(0)
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def print_colored_sequence(bytes_list, boundary_probs, boundary_mask):
    """Print the input sequence with characters colored by boundary probability.

    Multi-byte UTF-8 characters are printed once (at the lead byte). The
    character's color is taken from the lead byte's P(boundary). Per-byte
    boundary marks (`|`) still appear between bytes — a boundary inside a
    multi-byte character will show as a `|` mid-character.
    """
    labels = decode_utf8_labels(bytes_list)

    print("\n" + "=" * 80)
    print("Byte sequence colored by P(boundary) (blue=low, red=high)")
    print("Selected boundaries marked with |")
    print("=" * 80 + "\n")

    line = ""
    line_len = 0
    for i, (disp, p, is_boundary) in enumerate(zip(labels, boundary_probs, boundary_mask)):
        if is_boundary and i > 0:
            line += "\033[90m|\033[0m"
            line_len += 1
        if disp:  # lead byte: emit the (possibly multi-byte) character
            line += colorize(disp, p)
            line_len += len(disp)
        if line_len > 100 or disp == "\\n":
            print(line)
            line = ""
            line_len = 0
    if line:
        print(line)

    print("\n" + "=" * 80)
    print(f"{'Pos':>4} {'Byte':>5} {'Char':>5} {'Hex':>5} {'P(bnd)':>8} {'Boundary':>9}")
    print("-" * 80)
    for i, (b, p, is_boundary) in enumerate(zip(bytes_list, boundary_probs, boundary_mask)):
        char = labels[i] if labels[i] else "·"
        hex_repr = f"x{b:02x}"
        marker = " *" if is_boundary else ""
        print(f"{i:4d} {b:5d} {char:>5s} {hex_repr:>5s} {p:8.4f}{marker}")

    n_bnd = int(sum(boundary_mask))
    n_tot = len(boundary_mask)
    print(f"\nMean P(boundary): {sum(boundary_probs) / n_tot:.4f}")
    print(f"Boundaries: {n_bnd}/{n_tot} ({100 * n_bnd / n_tot:.1f}%)")
    if n_bnd > 0:
        print(f"Effective compression ratio: {n_tot / n_bnd:.2f}x")


def _add_byte_labels(ax, bytes_list, positions):
    """Two-row x-axis labels: decoded UTF-8 character on top (empty for
    continuation bytes), and an L/C marker on the bottom indicating whether
    the byte is the lead of a character or a continuation. Faint grey
    vertical lines mark character boundaries."""
    labels = decode_utf8_labels(bytes_list)
    n_chars = sum(1 for s in labels if s)
    has_continuation = any(s == "" for s in labels)

    if has_continuation:
        tick_labels = [f"{s}\nL" if s else "\nC" for s in labels]
    else:
        tick_labels = labels

    ax.set_xticks(positions)
    ax.set_xticklabels(
        tick_labels,
        fontsize=max(5, min(9, 800 // max(1, n_chars))),
        rotation=0,
        fontfamily="monospace",
    )
    for i in char_starts(labels)[1:]:
        ax.axvline(i - 0.5, color="lightgrey", linewidth=0.5, alpha=0.6, zorder=0)


def save_boundary_prob_plot(bytes_list, boundary_probs, boundary_mask, output_path, stage_idx):
    """Bar chart of P(boundary) per position, with selected boundaries highlighted."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(12, len(bytes_list) * 0.15), 5))
    positions = np.arange(len(boundary_probs))
    colors = ["red" if bm else "steelblue" for bm in boundary_mask]
    ax.bar(positions, boundary_probs, color=colors, width=1.0, edgecolor="none")

    _add_byte_labels(ax, bytes_list, positions)
    ax.set_ylabel("P(boundary)")
    ax.set_xlabel("Byte position")
    ax.set_title(f"Predicted boundary probability (stage {stage_idx})")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
               label="decision threshold (0.5)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def save_boundary_mask_plot(bytes_list, boundary_mask, output_path, stage_idx):
    """Binary plot showing which positions were selected as boundaries."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(12, len(bytes_list) * 0.15), 2.5))
    positions = np.arange(len(boundary_mask))
    values = np.asarray(boundary_mask, dtype=float)
    ax.bar(positions, values, color="red", width=1.0, edgecolor="none")

    _add_byte_labels(ax, bytes_list, positions)
    ax.set_ylabel("boundary")
    ax.set_xlabel("Byte position")
    ax.set_yticks([0, 1])
    ax.set_ylim(0, 1.1)
    n_bnd = int(values.sum())
    n_tot = len(values)
    ratio = n_tot / n_bnd if n_bnd > 0 else float("inf")
    ax.set_title(
        f"Selected boundaries (stage {stage_idx}) — "
        f"{n_bnd}/{n_tot} = {100 * n_bnd / n_tot:.1f}%, "
        f"compression {ratio:.2f}x"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predicted patch boundaries (router-agnostic)."
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the model configuration (.json file)")
    parser.add_argument("--prompt", type=str, default="The capital of Brazil is ",
                        help="Input text to analyze")
    parser.add_argument("--stage", type=int, default=0,
                        help="Hierarchy stage to visualize (0 = outermost = byte-level)")
    parser.add_argument("--output-plot", type=str, default=None,
                        help="If set, save matplotlib plots starting from this path")

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
        stages = compute_boundaries(model, input_ids)

    print(f"Model produced {len(stages)} routing stage(s).")
    for s in stages:
        n_bnd = int(s["boundary_mask"].sum())
        n_tot = len(s["boundary_mask"])
        ratio = n_tot / n_bnd if n_bnd > 0 else float("inf")
        print(f"  stage {s['stage_idx']}: T={n_tot}, boundaries={n_bnd} "
              f"({100 * n_bnd / n_tot:.1f}%), compression={ratio:.2f}x")

    if args.stage < 0 or args.stage >= len(stages):
        raise SystemExit(
            f"--stage {args.stage} out of range; model has {len(stages)} stage(s)."
        )
    stage = stages[args.stage]
    boundary_probs = stage["boundary_prob"].tolist()
    boundary_mask = stage["boundary_mask"].tolist()
    stage_idx = stage["stage_idx"]

    # Only the outermost stage's predictions align 1:1 with input bytes. Deeper
    # stages are over patches; we still render but warn the user.
    if args.stage != 0:
        print(f"\nNote: stage {args.stage} operates over patches from the previous "
              f"stage; x-axis labels are still the input bytes but boundaries "
              f"are over patch positions and won't align with byte positions.")

    print_colored_sequence(bytes_list, boundary_probs, boundary_mask)

    if args.output_plot:
        p = Path(args.output_plot)
        save_boundary_prob_plot(bytes_list, boundary_probs, boundary_mask,
                                args.output_plot, stage_idx)
        save_boundary_mask_plot(bytes_list, boundary_mask,
                                str(p.with_name(p.stem + "_mask" + p.suffix)),
                                stage_idx)


if __name__ == "__main__":
    main()
