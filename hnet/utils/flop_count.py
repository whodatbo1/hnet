"""
Forward-pass FLOP counter for HNet model configurations.

Uses analytical formulas from the H-Net paper:

  Embeddings:             2 × L × V × D
  Attention (per layer):  QKV projections + logits + softmax + score@V + out_proj
  Mamba-2 (per layer):    XZ + BCΔt + SSD + depthwise conv + gating + out_proj
  SwiGLU MLP (per layer): in/gate/out projections + gating
  LM head:                2 × L × V × D

Usage:
    python -m hnet.utils.flop_count --config configs/hnet_1stage_XL.json --seq-len 8192
    python -m hnet.utils.flop_count --config configs/hnet_2stage_XL.json \\
        --seq-len 8192 --downsample-n 4
"""

from __future__ import annotations

import re
import json
import argparse
import importlib.util
import os
from collections import OrderedDict


def _import_config_hnet():
    """Import config_hnet directly from its file to avoid triggering hnet/__init__.py,
    which eagerly loads flash_attn (requiring CUDA) via HNetForCausalLM."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(this_dir, "..", "models", "config_hnet.py")
    spec = importlib.util.spec_from_file_location("config_hnet", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Primitive FLOP formulas from the H-Net paper
# ---------------------------------------------------------------------------

def _attn_flops(seq_len: float, d_model: int, num_heads: int, window_size: int) -> float:
    """Attention FLOPs for one layer.

    window_size < 0 means full causal attention (context = seq_len).
    window_size > 0 means sliding-window attention (context = min(seq_len, window_size)).
    """
    head_dim = d_model // num_heads
    kv_dim   = num_heads * head_dim          # == d_model for standard MHA
    ctx      = seq_len if window_size < 0 else min(seq_len, window_size)

    qkv_proj    = 2 * 3 * seq_len * d_model * kv_dim
    attn_logits = 2 * seq_len * ctx * kv_dim
    softmax     = 3 * num_heads * seq_len * ctx
    score_v     = 2 * seq_len * ctx * kv_dim
    out_proj    = 2 * seq_len * kv_dim * d_model
    return qkv_proj + attn_logits + softmax + score_v + out_proj


def _ssm_flops(
    seq_len: float,
    d_model: int,
    expand: int,
    d_state: int,
    d_conv: int,
    ssm_headdim: int,
) -> float:
    """Mamba-2 FLOPs for one layer."""
    num_heads_ssm = max(1, (d_model * expand) // ssm_headdim)

    xz_proj      = 2 * seq_len * d_model * (2 * expand * d_model)
    bcdelta_proj = 2 * seq_len * d_model * (2 * d_state + num_heads_ssm)
    ssd          = 2 * 3 * seq_len * (expand * d_model) * d_state
    conv         = 2 * seq_len * d_model * d_conv
    gating       = 5 * seq_len * d_model
    out_proj     = 2 * seq_len * d_model * d_model
    return xz_proj + bcdelta_proj + ssd + conv + gating + out_proj


def _mlp_flops(seq_len: float, d_model: int, d_intermediate: int) -> float:
    """SwiGLU (gated MLP) FLOPs for one layer."""
    projections = 2 * seq_len * 3 * d_model * d_intermediate
    gating      = 5 * seq_len * d_model
    return projections + gating


# ---------------------------------------------------------------------------
# Isotropic stack FLOP counter
# ---------------------------------------------------------------------------

def _isotropic_flops(
    layout_str: str,
    seq_len: float,
    d_model: int,
    d_intermediate: int,
    num_heads: int,
    window_size: int,
    ssm_cfg: SSMConfig,
    ssm_headdim: int,
) -> dict[str, float]:
    """Return {'attn': ..., 'ssm': ..., 'mlp': ...} for one Isotropic module."""
    counts: dict[str, float] = {"attn": 0.0, "ssm": 0.0, "mlp": 0.0}
    for arch, n_str in re.findall(r"([mMtT])(\d+)", layout_str):
        for _ in range(int(n_str)):
            if arch in ("t", "T"):
                counts["attn"] += _attn_flops(seq_len, d_model, num_heads, window_size)
            else:  # m / M
                counts["ssm"] += _ssm_flops(
                    seq_len, d_model,
                    ssm_cfg.expand, ssm_cfg.d_state, ssm_cfg.d_conv, ssm_headdim,
                )
            if arch.isupper() and d_intermediate > 0:
                counts["mlp"] += _mlp_flops(seq_len, d_model, d_intermediate)
    return counts


# ---------------------------------------------------------------------------
# Recursive FLOP accumulator
# ---------------------------------------------------------------------------

def _accumulate(
    config: HNetConfig,
    stage_idx: int,
    seq_len: float,
    downsample_n: float,
    ssm_headdim: int,
    result: OrderedDict,
) -> None:
    """Walk the HNet hierarchy recursively and accumulate FLOPs per component."""
    arch_layout = config.arch_layout
    for _ in range(stage_idx):
        arch_layout = arch_layout[1]

    d_model        = config.d_model[stage_idx]
    d_intermediate = config.d_intermediate[stage_idx]
    num_heads      = config.attn_cfg.num_heads[stage_idx]
    window_size    = config.attn_cfg.window_size[stage_idx]

    is_innermost = (len(arch_layout) == 1)

    if is_innermost:
        result["main_network"] = _isotropic_flops(
            arch_layout[0], seq_len, d_model, d_intermediate,
            num_heads, window_size, config.ssm_cfg, ssm_headdim,
        )
    else:
        # Encoder at this stage
        result[f"encoder_{stage_idx}"] = _isotropic_flops(
            arch_layout[0], seq_len, d_model, d_intermediate,
            num_heads, window_size, config.ssm_cfg, ssm_headdim,
        )
        # Recurse into inner stage with compressed sequence length
        _accumulate(config, stage_idx + 1, seq_len / downsample_n,
                    downsample_n, ssm_headdim, result)
        # Decoder at this stage
        result[f"decoder_{stage_idx}"] = _isotropic_flops(
            arch_layout[2], seq_len, d_model, d_intermediate,
            num_heads, window_size, config.ssm_cfg, ssm_headdim,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_flops(
    config_path: str,
    seq_len: int,
    downsample_n: float = 4.0,
    ssm_headdim: int = 64,
) -> OrderedDict:
    """Compute forward-pass FLOPs for an HNet config.

    Args:
        config_path:  Path to the model JSON config file.
        seq_len:      Input sequence length (bytes) at the outermost stage.
        downsample_n: Expected average token compression at each routing stage.
                      E.g. 4 means the inner stage processes seq_len/4 tokens.
        ssm_headdim:  Mamba-2 SSM head dimension used to derive num_heads_ssm.
                      Default 64 matches the mamba_ssm library default.

    Returns:
        OrderedDict mapping component name to a breakdown dict with keys
        'attn', 'ssm', 'mlp', and 'total' (all in raw FLOPs).
        The final entry 'TOTAL' holds the grand total.
    """
    _m = _import_config_hnet()
    AttnConfig, SSMConfig, RoutingConfig, HNetConfig = (
        _m.AttnConfig, _m.SSMConfig, _m.RoutingConfig, _m.HNetConfig
    )

    with open(config_path) as f:
        cfg = json.load(f)

    attn_cfg    = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg_obj = SSMConfig(**cfg.pop("ssm_cfg"))
    routing_cfg = RoutingConfig(**cfg.pop("routing_cfg"))
    hnet_cfg    = HNetConfig(
        **cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg_obj, routing_cfg=routing_cfg
    )

    result: OrderedDict = OrderedDict()

    # Embeddings
    d_embed = hnet_cfg.d_model[0]
    emb_raw = 2.0 * seq_len * hnet_cfg.vocab_size * d_embed
    result["embeddings"] = {"attn": 0.0, "ssm": 0.0, "mlp": 0.0, "other": emb_raw}

    # Hierarchical backbone
    _accumulate(hnet_cfg, stage_idx=0, seq_len=float(seq_len),
                downsample_n=downsample_n, ssm_headdim=ssm_headdim, result=result)

    # LM head
    head_raw = 2.0 * seq_len * hnet_cfg.vocab_size * d_embed
    result["lm_head"] = {"attn": 0.0, "ssm": 0.0, "mlp": 0.0, "other": head_raw}

    # Compute per-component totals and grand total
    grand_total = 0.0
    for key, breakdown in result.items():
        component_total = sum(breakdown.values())
        breakdown["total"] = component_total
        grand_total += component_total

    result["TOTAL"] = {"attn": 0.0, "ssm": 0.0, "mlp": 0.0, "other": 0.0, "total": grand_total}
    return result


def print_flop_table(counts: OrderedDict, config_path: str = "", seq_len: int = 0) -> None:
    """Print a formatted FLOP breakdown table."""
    header = f"FLOPs for: {config_path}"
    if seq_len:
        header += f"  (seq_len={seq_len:,})"
    print(f"\n{header}")
    W = 76
    print("-" * W)
    print(f"{'Component':<22}  {'Attn (G)':>10}  {'SSM (G)':>10}  {'MLP (G)':>10}  {'Other (G)':>10}  {'Total (G)':>10}  {'Share':>6}")
    print("-" * W)

    grand = counts["TOTAL"]["total"]

    def _g(v):
        return f"{v / 1e9:>10.2f}" if v > 0 else f"{'—':>10}"

    for component, breakdown in counts.items():
        if component == "TOTAL":
            print("-" * W)
            total_g = breakdown["total"] / 1e9
            print(f"{'TOTAL':<22}  {'':>10}  {'':>10}  {'':>10}  {'':>10}  {total_g:>10.2f}")
            continue
        total  = breakdown["total"]
        share  = 100.0 * total / grand if grand > 0 else 0.0
        attn   = breakdown.get("attn",  0.0)
        ssm    = breakdown.get("ssm",   0.0)
        mlp    = breakdown.get("mlp",   0.0)
        other  = breakdown.get("other", 0.0)
        print(
            f"{component:<22}  {_g(attn)}  {_g(ssm)}  {_g(mlp)}  {_g(other)}  "
            f"{total/1e9:>10.2f}  {share:>5.1f}%"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Count HNet forward-pass FLOPs by component")
    parser.add_argument("--config",       type=str,   required=True,
                        help="Path to the model JSON config file")
    parser.add_argument("--seq-len",      type=int,   default=8192,
                        help="Input sequence length in bytes (default: 8192)")
    parser.add_argument("--downsample-n", type=float, default=4.0,
                        help="Expected token compression per routing stage (default: 4)")
    parser.add_argument("--ssm-headdim",  type=int,   default=64,
                        help="Mamba-2 SSM head dimension for num_heads_ssm derivation (default: 64)")
    args = parser.parse_args()

    counts = count_flops(args.config, args.seq_len, args.downsample_n, args.ssm_headdim)
    print_flop_table(counts, args.config, args.seq_len)


if __name__ == "__main__":
    main()
