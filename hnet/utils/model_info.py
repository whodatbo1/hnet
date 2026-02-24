"""Utility for counting model parameters broken down by component."""

from __future__ import annotations

import json
import argparse
from collections import OrderedDict

import torch

from hnet.models.config_hnet import AttnConfig, SSMConfig, RoutingConfig, HNetConfig
from hnet.models.mixer_seq import HNetForCausalLM


def _get_num_stages(arch_layout) -> int:
    """Count the number of hierarchical encoder/decoder stages."""
    n = 0
    while isinstance(arch_layout, list) and len(arch_layout) == 3:
        n += 1
        arch_layout = arch_layout[1]
    return n


def _build_prefix_map(n_stages: int) -> list[tuple[str, str]]:
    """Build an ordered list of (prefix, component_name) pairs for parameter classification.

    For n_stages=1, the parameter groups are:
        embeddings, encoder_0, routing_0, main_network, decoder_0, lm_head, other

    For n_stages=2:
        embeddings, encoder_0, routing_0, encoder_1, routing_1,
        main_network, decoder_1, decoder_0, lm_head, other
    """
    entries: list[tuple[str, str]] = [("embeddings", "embeddings")]

    for s in range(n_stages):
        base = "backbone" + ".main_network" * s
        entries.append((f"{base}.encoder", f"encoder_{s}"))
        entries.append((f"{base}.routing_module", f"routing_{s}"))
        entries.append((f"{base}.residual_proj", f"routing_{s}"))
        entries.append((f"{base}.dechunk_layer", f"routing_{s}"))

    # Innermost Isotropic: backbone + .main_network * n_stages + .main_network
    innermost_base = "backbone" + ".main_network" * n_stages
    entries.append((f"{innermost_base}.main_network", "main_network"))

    # Decoders from inner to outer
    for s in range(n_stages - 1, -1, -1):
        base = "backbone" + ".main_network" * s
        entries.append((f"{base}.decoder", f"decoder_{s}"))

    entries.append(("lm_head", "lm_head"))
    return entries


def _classify_param(name: str, prefix_map: list[tuple[str, str]]) -> str:
    for prefix, component in prefix_map:
        if name == prefix or name.startswith(prefix + "."):
            return component
    return "other"


def count_params(config_path: str) -> OrderedDict:
    """Count model parameters broken down by component.

    Args:
        config_path: Path to the model configuration JSON file.

    Returns:
        OrderedDict mapping component name to parameter count. Components appear
        in architecture order: embeddings, encoder_0, [encoder_1, ...], routing_0,
        [routing_1, ...], main_network, [decoder_n, ...], decoder_0, lm_head, other,
        total.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    routing_cfg = RoutingConfig(**cfg.pop("routing_cfg"))
    hnet_cfg = HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, routing_cfg=routing_cfg)

    n_stages = _get_num_stages(hnet_cfg.arch_layout)
    prefix_map = _build_prefix_map(n_stages)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)

    # Initialize counts for all expected components in order
    counts: dict[str, int] = {}
    seen_components: list[str] = []
    for _, component in prefix_map:
        if component not in counts:
            counts[component] = 0
            seen_components.append(component)
    counts["other"] = 0
    seen_components.append("other")

    for name, param in model.named_parameters():
        component = _classify_param(name, prefix_map)
        counts[component] += param.numel()

    total = sum(v for k, v in counts.items())
    counts["total"] = total

    ordered = OrderedDict()
    for comp in seen_components:
        ordered[comp] = counts[comp]
    ordered["total"] = total

    return ordered


def print_param_table(counts: OrderedDict, config_path: str = "") -> None:
    """Print a formatted parameter count table."""
    if config_path:
        print(f"\nParameter counts for: {config_path}")
    print("-" * 45)
    print(f"{'Component':<25} {'Parameters':>12}  {'Share':>6}")
    print("-" * 45)

    total = counts.get("total", 1)
    for component, count in counts.items():
        if component == "total":
            print("-" * 45)
            print(f"{'TOTAL':<25} {count:>12,}")
        else:
            share = 100.0 * count / total if total > 0 else 0.0
            print(f"{component:<25} {count:>12,}  {share:>5.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Count HNet model parameters by component"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration JSON file",
    )
    args = parser.parse_args()

    counts = count_params(args.config)
    print_param_table(counts, args.config)


if __name__ == "__main__":
    main()
