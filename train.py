"""
Training script for HNet language models.

Usage:
    # Single GPU with default config
    python train.py --config configs/train/default.yaml

    # Override specific fields via CLI
    python train.py --config configs/train/default.yaml --max-steps 10 --batch-size 2 --seq-len 512

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 train.py --config configs/train/default.yaml

    # Resume from checkpoint
    python train.py --config configs/train/default.yaml --resume
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import AttnConfig, SSMConfig, RoutingConfig, HNetConfig
from hnet.modules.block import Block
from hnet.utils.data import create_dataloaders, create_sft_dataloaders
from hnet.utils.train import load_balancing_loss, certainty_loss, group_params, orthogonality_regularization_soft, get_compression_ratio
from hnet.utils.eval import compression_metrics
from hnet.modules.dc import RoutingModule

# Learning rate schedule: Warmup-Stable-Decay (WSD)
def wsd_schedule(step, max_steps, warmup_steps, decay_fraction, base_lr):
    """WSD learning rate schedule with linear warmup, stable phase, and 1/sqrt decay."""
    decay_steps = int(max_steps * decay_fraction)
    stable_end = max_steps - decay_steps

    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        # Stable phase
        return base_lr
    else:
        # Inverse square root decay
        decay_progress = (step - stable_end) / max(decay_steps, 1)
        return base_lr / math.sqrt(1.0 + decay_progress * 9.0)
        # At the end (decay_progress=1), LR = base_lr / sqrt(10) ~ 0.316 * base_lr


# Learning rate schedule: linear warmup + cosine decay (Megatron/Llama style)
def cosine_schedule(step, max_steps, warmup_steps, min_lr_ratio, base_lr):
    """Linear warmup, then cosine decay from base_lr to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    decay_progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)

def log_gradient_norms(model, distributed=False, device=None, prefix="grad_norm"):
    """Compute per-module gradient L2 norms, grouped by the first two name segments.

    With FSDP (distributed=True), squared norms are all-reduced across ranks before
    taking the sqrt so that logged values reflect the full gradient, not a shard.
    Should be called after loss.backward() and before optimizer.step().
    """
    module_sq_norms = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = ".".join(name.split(".")[:3])
        sq = param.grad.detach().float().norm() ** 2
        module_sq_norms[key] = module_sq_norms.get(key, 0.0) + sq

    if distributed and device is not None:
        keys = sorted(module_sq_norms.keys())
        tensor = torch.tensor(
            [module_sq_norms[k].item() if isinstance(module_sq_norms[k], torch.Tensor) else module_sq_norms[k] for k in keys],
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        module_sq_norms = {k: tensor[i] for i, k in enumerate(keys)}

    return {
        f"{prefix}/{k}": (v.sqrt().item() if isinstance(v, torch.Tensor) else v ** 0.5)
        for k, v in module_sq_norms.items()
    }


# Precompute LR per step
def build_lr_schedule(cfg):
    # warmup_steps (absolute) takes precedence over warmup_fraction
    if cfg.get("warmup_steps") is not None:
        warmup_steps = int(cfg.warmup_steps)
    else:
        warmup_steps = int(cfg.max_steps * cfg.warmup_fraction)

    schedule = cfg.get("lr_schedule", "wsd")
    if schedule == "wsd":
        return {
            step: wsd_schedule(step, cfg.max_steps, warmup_steps, cfg.decay_fraction, 1.0)
            for step in range(cfg.max_steps)
        }
    elif schedule == "cosine":
        # min_lr is absolute (like Megatron's --min-lr); defaults to 10% of peak
        min_lr_ratio = cfg.get("min_lr", 0.1 * cfg.lr) / cfg.lr
        return {
            step: cosine_schedule(step, cfg.max_steps, warmup_steps, min_lr_ratio, 1.0)
            for step in range(cfg.max_steps)
        }
    else:
        raise ValueError(f"Unknown lr_schedule: {schedule!r} (expected 'wsd' or 'cosine')")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_num_stages(config):
    """Count the number of hierarchical stages from arch_layout."""
    layout = config.arch_layout
    n = 0
    while isinstance(layout, list) and len(layout) == 3:
        n += 1
        layout = layout[1]
    return n


def collect_entropy_metrics(model, prefix=""):
    """Collect entropy routing stats and learned params from all RoutingModules."""
    metrics = {}
    for module in model.modules():
        if isinstance(module, RoutingModule) and module.entropy_routing:
            s = module.stage_idx
            metrics[f"{prefix}entropy/stage_{s}/running_mean"]  = module.entropy_mean.item()
            metrics[f"{prefix}entropy/stage_{s}/running_std"]   = module.entropy_std.item()
            metrics[f"{prefix}entropy/stage_{s}/threshold"]     = module.entropy_threshold.item()
            metrics[f"{prefix}entropy/stage_{s}/temperature"]   = module.log_temperature.exp().item()
    return metrics


def print_rank0(msg, rank=None):
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(msg, flush=True)


def save_checkpoint(model, optimizer, step, cfg, checkpoint_dir, elapsed_time_since_training_start):
    """Save model and training state checkpoint."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full model state dict (gathered from FSDP)
    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(model_state, os.path.join(checkpoint_dir, f"model_step{step}.pt"))
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(),
         "config": OmegaConf.to_container(cfg),
         "time_since_start": int(elapsed_time_since_training_start)},
        os.path.join(checkpoint_dir, f"train_state_step{step}.pt"),
    )
    # Save a "latest" pointer
    torch.save({"step": step}, os.path.join(checkpoint_dir, "latest.pt"))
    print_rank0(f"Checkpoint saved at step {step} to {checkpoint_dir}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load model and training state from checkpoint. Returns the step to resume from."""
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest_path):
        return 0, 0

    latest = torch.load(latest_path, map_location="cpu")
    step = latest["step"]

    model_path = os.path.join(checkpoint_dir, f"model_step{step}.pt")
    train_state_path = os.path.join(checkpoint_dir, f"train_state_step{step}.pt")

    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    time_since_start = 0
    if os.path.exists(train_state_path):
        train_state = torch.load(train_state_path, map_location="cpu")
        optimizer.load_state_dict(train_state["optimizer"])
        time_since_start = train_state["time_since_start"]

    print_rank0(f"Resumed from checkpoint at step {step}")
    return step, time_since_start


def load_weights_only(model, init_from, device):
    """Warm-start: load ONLY model weights from a checkpoint; fresh optimizer/step.

    Used for continual pretraining and SFT initialization (cfg.init_from), as
    opposed to cfg.resume (which also restores the optimizer state and step
    counter to continue the same run). The LR schedule therefore restarts from
    step 0 with the new run's max_steps.

    `init_from` may be a checkpoint directory (resolves latest.pt -> model_step{N}.pt),
    a path to a latest.pt pointer, or a direct model_step*.pt file. Mirrors
    generate.load_from_pretrained's resolution.
    """
    init_from = str(init_from)
    if os.path.isdir(init_from):
        latest_path = os.path.join(init_from, "latest.pt")
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"init_from dir has no latest.pt: {init_from}")
        step = torch.load(latest_path, map_location="cpu")["step"]
        model_path = os.path.join(init_from, f"model_step{step}.pt")
    elif os.path.basename(init_from) == "latest.pt":
        step = torch.load(init_from, map_location="cpu")["step"]
        model_path = os.path.join(os.path.dirname(init_from), f"model_step{step}.pt")
    else:
        model_path = init_from
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"init_from checkpoint not found: {model_path}")

    if isinstance(model, FSDP):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    print_rank0(f"Warm-started model weights from {model_path} (fresh optimizer, step 0)")


def split_batch(batch, sft):
    """Derive (input_ids, targets_ce, targets_model) for one batch.

    Pretraining: `batch` is (B, seq_len+1); plain next-token shift, and the same
    shifted targets are used for both the outer CE and the model's `targets=`.

    SFT: `batch` is (B, 2, seq_len+1) — row 0 token ids, row 1 labels (-100 on
    BOS+prompt, byte id on target+EOS). The model conditions on real bytes
    (input_ids), the outer cross-entropy is supervised only on the target region
    (targets_ce, -100 elsewhere), and the DENSE token row (targets_model) is
    passed to model(targets=) so the router's boundary-matching aux loss stays
    healthy over the whole sequence.
    """
    if sft:
        tokens = batch[:, 0, :]
        labels = batch[:, 1, :]
        return tokens[:, :-1], labels[:, 1:], tokens[:, 1:]
    shifted = batch[:, 1:]
    return batch[:, :-1], shifted, shifted


def unpack_batch(batch, device):
    """Split a dataloader batch into (chunks, source_ids).

    Mixture batches arrive as [chunks (B, L+1), source_ids (B,)] from
    default_collate (MultilingualByteDataset yields (chunk, source_idx));
    single-source and SFT batches are plain tensors -> source_ids is None.
    """
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)
    return batch.to(device), None


class PerSourceMetrics:
    """Accumulate per-source (language) metric sums for dataset mixtures.

    All state lives in fixed-size fp32 device tensors (sums and counts), so a
    single all_reduce(SUM) at log/val time yields count-weighted global means.
    Per-source lb/cert values are per-group statistics over that source's
    positions; at inner stages they do not linearly recombine into the
    aggregate scalars (rows contribute unequal patch counts) — that is the
    intended semantics of a per-language split, not a decomposition.
    """

    def __init__(self, source_names, n_route_stages, device):
        self.source_names = list(source_names)
        self.n_src = len(self.source_names)
        self.n_route = n_route_stages
        self.device = device
        # Cumulative supervised-token totals across the whole run (survives
        # reset(); like the aggregate total_tokens, restarts at 0 on resume).
        # fp64 so multi-billion-token counts accumulate without rounding.
        self.total_tok_count = torch.zeros(self.n_src, device=device, dtype=torch.float64)
        self._q_points = torch.tensor([0.5, 0.75, 0.9, 0.99], device=device)
        self.reset()

    def reset(self):
        def z(*shape):
            return torch.zeros(*shape, device=self.device)
        self.loss_sum = z(self.n_src)        # per-token CE sum
        self.tok_count = z(self.n_src)       # supervised target tokens
        self.pos_count = z(self.n_route, self.n_src)  # routing positions
        self.mask_sum = z(self.n_route, self.n_src)   # selected boundaries
        self.prob_sum = z(self.n_route, self.n_src)   # boundary probs
        self.cert_sum = z(self.n_route, self.n_src)   # binary entropies
        self.lb_wsum = z(self.n_route, self.n_src)    # position-weighted lb values
        self.bm_sum = z(self.n_route, self.n_src)
        self.bm_count = z(self.n_route, self.n_src)
        self.ent_sum = z(self.n_route, self.n_src)
        self.ent_count = z(self.n_route, self.n_src)
        # Boundary-statistic detail sums
        self.prob_sq_sum = z(self.n_route, self.n_src)   # G_var
        self.sel_prob_sum = z(self.n_route, self.n_src)  # G_pos / G_neg
        self.gq_sum = z(self.n_route, self.n_src, 4)     # per-batch G quantiles
        self.gq_batch_count = z(self.n_route, self.n_src)

    def _state(self):
        return [self.loss_sum, self.tok_count, self.pos_count, self.mask_sum,
                self.prob_sum, self.cert_sum, self.lb_wsum, self.bm_sum,
                self.bm_count, self.ent_sum, self.ent_count,
                self.prob_sq_sum, self.sel_prob_sum, self.gq_sum, self.gq_batch_count]

    def flat_state(self):
        return torch.cat([t.reshape(-1) for t in self._state()])

    def load_flat_state(self, flat):
        offset = 0
        for t in self._state():
            t.copy_(flat[offset:offset + t.numel()].view_as(t))
            offset += t.numel()

    def all_reduce(self):
        flat = self.flat_state()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        self.load_flat_state(flat)

    def accumulate_totals(self):
        """Fold this window's token counts into the cumulative per-source totals.

        Call once per log window, after all_reduce() and before reset(), so the
        totals are global across ranks (every rank ends up with the same value).
        """
        self.total_tok_count += self.tok_count.double()

    def total_metrics(self, prefix="total_tokens/"):
        return {
            f"{prefix}{name}": int(self.total_tok_count[j].item())
            for j, name in enumerate(self.source_names)
        }

    @torch.no_grad()
    def update(self, logits, targets, bpred_output, source_ids, downsample_n):
        """Bucket one micro-batch's metrics by source. Detached throughout."""
        tok_loss = nn.functional.cross_entropy(
            logits.detach().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(targets.shape)
        self.loss_sum.index_add_(0, source_ids, tok_loss.sum(dim=1).float())
        self.tok_count.index_add_(0, source_ids, (targets != -100).sum(dim=1).float())

        if not bpred_output:
            return
        # Map each packed routing position back to its row's source: bpred_output
        # is ordered outermost-first, and each stage's boundary_mask selects, in
        # order, the positions forwarded to the next stage.
        row_ids = torch.arange(
            source_ids.shape[0], device=source_ids.device
        ).repeat_interleave(targets.shape[1])
        for s, router_out in enumerate(bpred_output):
            lang = source_ids[row_ids]
            bmask = router_out.boundary_mask.reshape(-1)
            p = router_out.boundary_prob[..., -1].float().reshape(-1)

            cnt = torch.zeros(self.n_src, device=self.device).index_add_(0, lang, torch.ones_like(p))
            mask_sum = torch.zeros(self.n_src, device=self.device).index_add_(0, lang, bmask.float())
            prob_sum = torch.zeros(self.n_src, device=self.device).index_add_(0, lang, p)
            self.pos_count[s] += cnt
            self.mask_sum[s] += mask_sum
            self.prob_sum[s] += prob_sum

            # Same eps and formula as certainty_loss
            pc = p.clamp(1e-7, 1 - 1e-7)
            self.cert_sum[s].index_add_(0, lang, -(pc * pc.log() + (1 - pc) * (1 - pc).log()))

            # Same combiner as load_balancing_loss, applied to this source's
            # means for this micro-batch, weighted by its position count (uses
            # the correct per-step N under compression_schedule and skips
            # sources absent from the micro-batch).
            safe_cnt = cnt.clamp(min=1)
            true_ratio = mask_sum / safe_cnt
            average_prob = prob_sum / safe_cnt
            lb = (
                (1 - true_ratio) * (1 - average_prob)
                + true_ratio * average_prob * (downsample_n - 1)
            ) * downsample_n / (downsample_n - 1)
            self.lb_wsum[s] += lb * cnt

            if router_out.bm_loss_per_pos is not None:
                self.bm_sum[s].index_add_(0, lang, router_out.bm_loss_per_pos.reshape(-1))
                self.bm_count[s].index_add_(0, lang, router_out.bm_valid_mask.reshape(-1).float())
            if router_out.entropy_per_pos is not None:
                self.ent_sum[s].index_add_(0, lang, router_out.entropy_per_pos.reshape(-1))
                self.ent_count[s] += cnt

            # Boundary-statistic details: exact position sums for G_var and
            # G_pos/G_neg; quantiles can't be accumulated exactly, so store
            # per-micro-batch quantiles and average them at log time (same
            # semantics as the aggregate accum_comp_metrics).
            self.prob_sq_sum[s].index_add_(0, lang, p * p)
            self.sel_prob_sum[s].index_add_(0, lang, p * bmask.float())
            for j in lang.unique().tolist():
                self.gq_sum[s, j] += torch.quantile(p[lang == j], self._q_points)
                self.gq_batch_count[s, j] += 1

            row_ids = row_ids[bmask]

    def metrics(self, prefix="", comp_prefix=None):
        """Per-source metric dict from the accumulated sums.

        prefix is used for loss-like keys ("" for train, "val/" for val);
        comp_prefix for the per-stage compression keys ("train/" or "val/",
        defaults to prefix). Sources with zero counts are omitted.
        """
        if comp_prefix is None:
            comp_prefix = prefix
        out = {}
        for j, name in enumerate(self.source_names):
            if self.tok_count[j] > 0:
                out[f"{prefix}loss/{name}"] = (self.loss_sum[j] / self.tok_count[j]).item()
            stage_has = self.pos_count[:, j] > 0
            if stage_has.any():
                cnt = self.pos_count[stage_has, j]
                # Mean over stages, mirroring the aggregate losses' average
                # over bpred_output
                out[f"{prefix}lb_loss/{name}"] = (self.lb_wsum[stage_has, j] / cnt).mean().item()
                out[f"{prefix}cert_loss/{name}"] = (self.cert_sum[stage_has, j] / cnt).mean().item()
            bm_has = self.bm_count[:, j] > 0
            if bm_has.any():
                out[f"{prefix}bm_loss/{name}"] = (
                    self.bm_sum[bm_has, j] / self.bm_count[bm_has, j]
                ).mean().item()
            for s in range(self.n_route):
                if self.pos_count[s, j] > 0:
                    n = self.pos_count[s, j]
                    sel = self.mask_sum[s, j]
                    ps = self.prob_sum[s, j]
                    out[f"{comp_prefix}stage_{s}/F_selected/{name}"] = (sel / n).item()
                    out[f"{comp_prefix}stage_{s}/G_avg_boundary_prob/{name}"] = (ps / n).item()
                    # Unbiased variance over the window's positions from sums
                    # (boundary_mask is 0/1, so its sum of squares == sel)
                    denom = (n - 1).clamp(min=1)
                    out[f"{comp_prefix}stage_{s}/G_var/{name}"] = (
                        (self.prob_sq_sum[s, j] - ps * ps / n) / denom
                    ).item()
                    out[f"{comp_prefix}stage_{s}/F_var/{name}"] = ((sel - sel * sel / n) / denom).item()
                    if sel > 0:
                        out[f"{comp_prefix}stage_{s}/G_boundary_prob_pos/{name}"] = (
                            self.sel_prob_sum[s, j] / sel
                        ).item()
                    if n > sel:
                        out[f"{comp_prefix}stage_{s}/G_boundary_prob_neg/{name}"] = (
                            (ps - self.sel_prob_sum[s, j]) / (n - sel)
                        ).item()
                if self.gq_batch_count[s, j] > 0:
                    for qi, qname in enumerate(("G_p50", "G_p75", "G_p90", "G_p99")):
                        out[f"{comp_prefix}stage_{s}/{qname}/{name}"] = (
                            self.gq_sum[s, j, qi] / self.gq_batch_count[s, j]
                        ).item()
                if self.ent_count[s, j] > 0:
                    out[f"{prefix}entropy/stage_{s}/batch_mean/{name}"] = (
                        self.ent_sum[s, j] / self.ent_count[s, j]
                    ).item()
        return out


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(argv=None):
    """Load training config from YAML file with CLI overrides.

    Precedence: CLI flags > YAML file > defaults (in default.yaml).

    Model config fields (from the JSON) can be overridden using the ``--model.``
    prefix, e.g.::

        --model.d_model=[512,1024]
        --model.routing_cfg.multiheaded=true
        --model.attn_cfg.num_heads=[8,8]

    All other ``--key value`` flags override the training YAML config.
    """
    parser = argparse.ArgumentParser(description="Train an HNet language model")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml",
                        help="Path to training config YAML")
    # Allow arbitrary overrides via dotlist (e.g. --lr 1e-3 --seq_len 512)
    args, remaining = parser.parse_known_args(argv)

    # Load YAML
    yaml_cfg = OmegaConf.load(args.config)

    # Parse remaining CLI args as dotlist overrides.
    # Args prefixed with --model. go to the model config; all others to train config.
    # Convert --kebab-case to snake_case for OmegaConf compatibility.
    train_overrides = []
    model_overrides = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                val = remaining[i + 1]
                entry = f"{key}={val}"
                i += 2
            else:
                entry = f"{key}=true"
                i += 1
            if key.startswith("model."):
                model_overrides.append(entry[len("model."):])
            elif key.startswith("train."):
                train_overrides.append(entry[len("train."):])
            else:
                train_overrides.append(entry)
        else:
            i += 1

    cli_cfg = OmegaConf.from_dotlist(train_overrides)
    cfg = OmegaConf.merge(yaml_cfg, cli_cfg)

    # Stash model overrides so main() can apply them to the JSON config.
    if model_overrides:
        cfg.model_overrides = OmegaConf.from_dotlist(model_overrides)

    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_dataloader, cfg, step, device, downsample_n=None):
    """Run validation and return metrics dict.

    val_dataloader is either a single DataLoader (single-source / SFT runs) or
    a list of (source_name, DataLoader) pairs (mixture runs; see
    create_dataloaders). In the list case each loader is single-source and is
    evaluated for cfg.val_batches batches — i.e. val_batches is PER SOURCE —
    producing val/.../<name> metrics, while the unsuffixed val/* metrics pool
    the batches of all sources.

    Computes (pooled, and per source with a /<name> suffix for mixtures):
    - val/loss: average cross-entropy loss
    - val/bpb: bits-per-byte
    - val/lb_loss, val/cert_loss, val/bm_loss: average aux losses
    - val/stage_*/...: compression / boundary statistics
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed = dist.is_initialized()

    was_training = model.training
    model.eval()

    loaders = val_dataloader if isinstance(val_dataloader, list) else [(None, val_dataloader)]
    n_loaders = len(loaders)

    val_batches = cfg.get("val_batches", 50)
    if downsample_n is None:
        downsample_n = cfg.downsample_n

    ortho_reg_lambda = cfg.get("ortho_reg_lambda", 0.0)
    if not ortho_reg_lambda:
        ortho_reg_lambda = 0.0

    sft = cfg.get("sft", False)

    # Per-loader sums; columns: [ce, lb, cert, ortho, bm, bytes, batches]
    stats = torch.zeros(n_loaders, 7, device=device)
    # Per-loader {metric_key: [per-batch values]}; rank-local (as before)
    comp_lists = [{} for _ in loaders]

    for li, (_, loader) in enumerate(loaders):
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= val_batches:
                break

            batch, _ = unpack_batch(batch, device)
            input_ids, targets, targets_model = split_batch(batch, sft)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, mask=None, targets=targets_model)
                logits = output.logits

                # Sum (not mean) CE loss so we can compute BPB correctly. ignore_index
                # drops the -100 prompt positions in SFT (no-op in pretraining).
                ce_loss_sum = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )

                ortho_loss = 0.0
                if ortho_reg_lambda > 0:
                    for module in model.modules():
                        if isinstance(module, RoutingModule):
                            ortho_loss += orthogonality_regularization_soft(module.q_proj_layer.weight)
                            ortho_loss += orthogonality_regularization_soft(module.k_proj_layer.weight)

                lb_loss = torch.tensor(0.0, device=device)
                cert_loss = torch.tensor(0.0, device=device)
                bm_loss = torch.tensor(0.0, device=device)

                if output.bpred_output:
                    for router_out in output.bpred_output:
                        lb_loss = lb_loss + load_balancing_loss(
                            router_out, N=downsample_n
                        )
                        cert_loss = cert_loss + certainty_loss(router_out)
                        if router_out.bm_loss is not None:
                            bm_loss += router_out.bm_loss

                    lb_loss = lb_loss / len(output.bpred_output)
                    cert_loss = cert_loss / len(output.bpred_output)
                    bm_loss = bm_loss / len(output.bpred_output)

                    # Accumulate compression / boundary metrics
                    batch_comp = compression_metrics(output.bpred_output)
                    for k, v in batch_comp.items():
                        comp_lists[li].setdefault(k, []).append(v)

            stats[li, 0] += ce_loss_sum
            stats[li, 1] += lb_loss
            stats[li, 2] += cert_loss
            stats[li, 3] += ortho_loss
            stats[li, 4] += bm_loss
            # Count only supervised positions so val/loss and bpb are per-target-byte
            # in SFT. In pretraining no position is -100, so this equals targets.numel().
            stats[li, 5] += (targets != -100).sum()
            stats[li, 6] += 1

    if stats[:, 6].sum().item() == 0:
        model.train(was_training)
        return {}

    # Aggregate across ranks (single collective over all loaders' sums)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    def derive(row, suffix=""):
        """Loss metrics from one summed stats row (pooled or per-source)."""
        ce, lb, cert, _, bm, nbytes, nbatches = row.tolist()
        if nbatches == 0 or nbytes == 0:
            return {}
        sfx = f"/{suffix}" if suffix else ""
        return {
            f"val/loss{sfx}": ce / nbytes,
            f"val/bpb{sfx}": ce / (nbytes * math.log(2)),
            f"val/lb_loss{sfx}": lb / nbatches,
            f"val/cert_loss{sfx}": cert / nbatches,
            f"val/bm_loss{sfx}": bm / nbatches,
        }

    pooled_row = stats.sum(dim=0)
    metrics = {"step": step}
    metrics.update(derive(pooled_row))
    metrics["val/ortho_loss"] = (pooled_row[3] / pooled_row[6]).item()

    # Compression / boundary metrics: pooled over all batches, plus per source
    pooled_comp = {}
    for li, (name, _) in enumerate(loaders):
        for k, vals in comp_lists[li].items():
            pooled_comp.setdefault(k, []).extend(vals)
            if name is not None and vals:
                metrics[f"val/{k}/{name}"] = sum(vals) / len(vals)
    for k, vals in pooled_comp.items():
        metrics[f"val/{k}"] = sum(vals) / len(vals)

    # Per-source loss metrics
    for li, (name, _) in enumerate(loaders):
        if name is not None:
            metrics.update(derive(stats[li], suffix=name))

    # Entropy routing stats (running EMA + learned params)
    metrics.update(collect_entropy_metrics(model, prefix="val/"))

    print_rank0(
        f"[val] step={step:>6d} | bpb={metrics['val/bpb']:.4f} | loss={metrics['val/loss']:.4f} | "
        f"lb_loss={metrics['val/lb_loss']:.4f} | cert_loss={metrics['val/cert_loss']:.4f} | "
        f"ortho_loss={metrics['val/ortho_loss']:.4f} | bm_loss={metrics['val/bm_loss']:.4f}",
        rank,
    )
    for li, (name, _) in enumerate(loaders):
        if name is not None and f"val/loss/{name}" in metrics:
            print_rank0(
                f"[val]   {name}: loss={metrics[f'val/loss/{name}']:.4f} | bpb={metrics[f'val/bpb/{name}']:.4f}",
                rank,
            )

    model.train(was_training)
    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # torch.autograd.set_detect_anomaly(True)
    cfg = load_config()

    # ---- Distributed setup ----
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    torch.manual_seed(cfg.seed + rank)

    # ---- Validate cfg ----
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * world_size * cfg.seq_len

    if cfg.get("total_tokens"):
        max_steps = cfg.total_tokens // tokens_per_step
        cfg.max_steps = max_steps
    elif cfg.get("max_steps"):
        max_steps = cfg.max_steps
    else:
        raise ValueError("Specify either total_tokens or max_steps")

    total_tokens_actual = max_steps * tokens_per_step
    print_rank0(f"Training for {max_steps} steps = {total_tokens_actual:,} tokens")

    # ---- Load model config & create model ----
    with open(cfg.model_config, "r") as f:
        model_config = json.load(f)

    # Apply any --model.* CLI overrides (deep-merge via OmegaConf)
    if cfg.get("model_overrides"):
        model_oc = OmegaConf.merge(OmegaConf.create(model_config), cfg.model_overrides)
        model_config = OmegaConf.to_container(model_oc, resolve=True)
        print_rank0(f"Model config overrides applied: {OmegaConf.to_yaml(cfg.model_overrides)}", rank)

    model_config_raw = dict(model_config)  # snapshot before pops, for wandb logging

    attn_cfg = AttnConfig(**model_config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**model_config.pop("ssm_cfg"))
    routing_cfg = RoutingConfig(**model_config.pop("routing_cfg"))
    hnet_cfg = HNetConfig(**model_config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg, routing_cfg=routing_cfg)

    print_rank0(f"Model config: {hnet_cfg}", rank)
    print_rank0(f"Train config: {OmegaConf.to_yaml(cfg)}", rank)

    # Create model on CPU first, then move to device (for FSDP)
    model = HNetForCausalLM(hnet_cfg, device="cpu", dtype=torch.float32)
    model.init_weights()

    num_stages = get_num_stages(hnet_cfg) + 1  # +1 for innermost
    lr_multiplier = cfg.get("lr_multiplier", None)
    if lr_multiplier is not None:
        lr_multiplier = list(lr_multiplier) if not isinstance(lr_multiplier, list) else lr_multiplier
        assert len(lr_multiplier) == num_stages, \
            f"Expected {num_stages} LR multipliers, got {len(lr_multiplier)}"
        model.apply_lr_multiplier(lr_multiplier)

    # Ensure every parameter has _optim (group_params requires it)
    for param in model.parameters():
        if not hasattr(param, "_optim"):
            param._optim = {}

    # Create param groups before FSDP wrapping
    param_groups = group_params(model)
    # Apply base LR and weight decay defaults
    for pg in param_groups:
        pg.setdefault("weight_decay", cfg.weight_decay)
        if "lr_multiplier" in pg:
            pg["lr"] = cfg.lr * pg.pop("lr_multiplier")
        else:
            pg["lr"] = cfg.lr

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"Parameters: {n_params:,} total, {n_trainable:,} trainable", rank)

    # ---- FSDP wrapping ----
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    if distributed:
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={Block},
        )
        model = FSDP(
            model.to(device),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=bf16_policy,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            limit_all_gathers=True,
            use_orig_params=True,  # Required for per-parameter LR multipliers
        )
    else:
        model = model.to(device)

    # Activation checkpointing
    if cfg.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            apply_activation_checkpointing,
            CheckpointImpl,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda module: isinstance(module, Block),
        )
        print_rank0("Activation checkpointing enabled on Block modules", rank)
    
    # Compile the model
    if cfg.get("compile", False):
        model = torch.compile(model)
        print_rank0("torch.compile enabled", rank)

    # ---- Optimizer ----
    # use_orig_params=True in FSDP allows per-parameter LR/WD from group_params
    betas = tuple(cfg.betas) if cfg.get("betas") else (0.9, 0.95)
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    # ---- LR Schedule ----
    lr_per_step = build_lr_schedule(cfg)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    # ---- Data ----
    sft = cfg.get("sft", False)
    if sft:
        dataloader, val_dataloader = create_sft_dataloaders(
            Path(cfg.data_dir),
            cfg.sft_name,
            cfg.seq_len,
            cfg.seed,
            cfg.get("val_batches", 50),
            cfg.batch_size,
            cfg.num_workers,
        )
    else:
        dataloader, val_dataloader = create_dataloaders(
            Path(cfg.data_dir),
            cfg.get("dataset_config", "sample-10BT"),
            cfg.get("dataset_mixture", None),
            cfg.seq_len,
            cfg.seed,
            cfg.get("val_batches", 50),
            cfg.batch_size,
            cfg.num_workers
        )

    # Per-language metrics only apply to real multi-source mixtures; for a
    # single-entry mixture _build_dataset returns a bare MemmapByteDataset
    # (plain-tensor batches, no source ids), so the >1 guard must match.
    mixture = cfg.get("dataset_mixture", None)
    source_names = None
    if not sft and mixture is not None and len(mixture) > 1:
        source_names = [s["name"] for s in mixture]

    # ---- Resume / warm-start ----
    # resume: continue this run (weights + optimizer + step). init_from: start a
    # fresh run from another checkpoint's weights only (continual PT / SFT).
    assert not (cfg.resume and cfg.get("init_from")), (
        "Set either `resume: true` (continue this run) or `init_from` (warm-start "
        "a new run from another checkpoint), not both."
    )
    start_step = 0
    time_since_start = 0
    if cfg.resume:
        start_step, time_since_start = load_checkpoint(model, optimizer, cfg.checkpoint_dir, device)
    elif cfg.get("init_from"):
        load_weights_only(model, cfg.init_from, device)

    # ---- Wandb ----
    wandb_project = cfg.get("wandb_project", None)
    wandb_run_name = cfg.get("wandb_run_name", "train_hnet_unknown") + "_" + datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
    if wandb_project and rank == 0:
        import wandb
        wandb.init(
            name=wandb_run_name,
            project=wandb_project,
            entity="marko-ivanovv",
            tags=["train"],
            config={
                "train": OmegaConf.to_container(cfg),
                "model": model_config_raw,
            },
        )

    # ---- Training loop ----
    effective_batch = cfg.batch_size * cfg.grad_accum_steps * world_size
    print_rank0(
        f"Training: {max_steps} steps, "
        f"micro_batch={cfg.batch_size}, grad_accum={cfg.grad_accum_steps}, "
        f"world_size={world_size}, effective_batch={effective_batch}, "
        f"seq_len={cfg.seq_len}",
        rank,
    )

    model.train()
    optimizer.zero_grad()

    step = start_step
    accum_loss = 0.0
    accum_lb_loss = 0.0
    accum_cert_loss = 0.0
    accum_ortho_loss = 0.0
    accum_bm_loss = 0.0
    accum_comp_metrics = defaultdict(float)
    accum_tokens = 0
    total_tokens = 0
    epoch = 0
    # num_stages includes the innermost stage, which has no router
    lang_metrics_acc = (
        PerSourceMetrics(source_names, num_stages - 1, device)
        if source_names is not None else None
    )

    # Timing
    train_start = time.time()
    elapsed_time_since_last_log = 0
    elapsed_time_since_training_start = time_since_start
    print_rank0(f"Starting training at {train_start}")

    data_iter = iter(dataloader)
    ortho_reg_lambda = cfg.get("ortho_reg_lambda", 0)
    if not ortho_reg_lambda:
        ortho_reg_lambda = 0

    while step < max_steps:
        step_start_time = time.time()
        if step % cfg.log_every == 0:
            print_rank0(f"Starting step: {step}")

        lr_scale = lr_per_step[step]
        if step % cfg.log_every == 0:
            print_rank0(f"Setting lr_scale: {lr_scale}")
        for base_lr, pg in zip(base_lrs, optimizer.param_groups):
            pg["lr"] = base_lr * lr_scale

        compression_schedule = cfg.get("compression_schedule", None)
        current_downsample_n = (
            get_compression_ratio(step, max_steps, compression_schedule)
            if compression_schedule is not None
            else cfg.downsample_n
        )

        batch_entropy_means = torch.zeros((num_stages, cfg.grad_accum_steps), device=device)
        batch_entropy_stds = torch.zeros((num_stages, cfg.grad_accum_steps), device=device)
        batch_entropy_element_counts = torch.zeros((num_stages, cfg.grad_accum_steps), device=device)
        for micro_step in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                print_rank0(f"Epoch {epoch} started at step {step}", rank)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch, source_ids = unpack_batch(batch, device)
            # SFT: targets is the -100-masked label row (loss on target only);
            # targets_model is the dense token row for the router aux loss.
            input_ids, targets, targets_model = split_batch(batch, sft)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # mask=None triggers packed mode in the model
                output = model(input_ids, mask=None, targets=targets_model)

                # AR cross-entropy loss
                logits = output.logits  # (B, seq_len, vocab_size)
                ce_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                            
                ortho_loss = 0.0
            
                if ortho_reg_lambda > 0:
                    for module in model.modules():
                        if isinstance(module, RoutingModule):
                            ortho_loss += orthogonality_regularization_soft(module.q_proj_layer.weight)
                            ortho_loss += orthogonality_regularization_soft(module.k_proj_layer.weight)

                # Load balancing loss across routing stages
                lb_loss = torch.tensor(0.0, device=device)
                cert_loss = torch.tensor(0.0, device=device)
                bm_loss = torch.tensor(0.0, device=device)

                if output.bpred_output:
                    for router_out in output.bpred_output:
                        lb_loss = lb_loss + load_balancing_loss(
                            router_out, N=current_downsample_n
                        )
                        cert_loss = cert_loss + certainty_loss(router_out)
                        if router_out.entropy_mean is not None and router_out.entropy_std is not None:
                            batch_entropy_means[router_out.stage_idx, micro_step] = router_out.entropy_mean
                            batch_entropy_stds[router_out.stage_idx, micro_step] = router_out.entropy_std
                            batch_entropy_element_counts[router_out.stage_idx, micro_step] = router_out.boundary_prob.numel()
                        if router_out.bm_loss is not None:
                            bm_loss += router_out.bm_loss
                    lb_loss = lb_loss / len(output.bpred_output)
                    cert_loss = cert_loss / len(output.bpred_output)
                    bm_loss = bm_loss / len(output.bpred_output)

                bm_loss_weight = cfg.get("bm_loss_weight", 0.1)
                cert_loss_weight = cfg.get("cert_loss_weight", 0.0)
                loss = ce_loss + cfg.alpha * lb_loss + cert_loss_weight * cert_loss + ortho_reg_lambda * ortho_loss + bm_loss_weight * bm_loss
                loss = loss / cfg.grad_accum_steps

            loss.backward()

            accum_loss += ce_loss.detach().item()
            accum_lb_loss += lb_loss.detach().item()
            accum_cert_loss += cert_loss.detach().item()
            accum_ortho_loss += ortho_loss.detach().item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
            accum_bm_loss += bm_loss.detach().item()
            accum_tokens += targets.numel()
            for k, v in compression_metrics(output.bpred_output).items():
                accum_comp_metrics[k] += v
            if lang_metrics_acc is not None and source_ids is not None:
                lang_metrics_acc.update(
                    logits, targets, output.bpred_output, source_ids, current_downsample_n
                )

        # clamp(min=1) prevents 0/0 NaN for stages without entropy routing;
        # those stages have zero counts so their weights are 0/1=0, which is harmless
        # since the module.entropy_routing guard below skips their EMA update anyway.
        global_entropy_weights = batch_entropy_element_counts / batch_entropy_element_counts.sum(dim=-1, keepdim=True).clamp(min=1)
        global_entropy_means = (batch_entropy_means * global_entropy_weights).sum(dim=-1)
        global_entropy_var = (
            (global_entropy_weights * batch_entropy_stds ** 2).sum(dim=-1) +
            (global_entropy_weights * (batch_entropy_means - global_entropy_means.unsqueeze(-1)) ** 2).sum(dim=-1)
        )
        for module in model.modules():
            if isinstance(module, RoutingModule) and module.entropy_routing:
                stage_idx = module.stage_idx
                module.entropy_mean.copy_((1 - cfg.entropy_decay) * module.entropy_mean + cfg.entropy_decay * global_entropy_means[stage_idx])
                module.entropy_std.copy_((1 - cfg.entropy_decay) * module.entropy_std + cfg.entropy_decay * global_entropy_var[stage_idx].sqrt())

        # Per-module gradient norms (pre-clip, only on log steps to limit overhead)
        grad_norm_metrics = {}
        if (step + 1) % cfg.log_every == 0:
            grad_norm_metrics = log_gradient_norms(model, distributed=distributed, device=device)

        # Gradient clipping
        if cfg.max_grad_norm > 0:
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(cfg.max_grad_norm).item()
            else:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                ).item()
        else:
            grad_norm = 0.0

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        elapsed_time_since_last_log += step_time
        elapsed_time_since_training_start += step_time

        # Logging
        if step % cfg.log_every == 0:
            avg_loss = accum_loss / (cfg.grad_accum_steps * cfg.log_every)
            avg_lb = accum_lb_loss / (cfg.grad_accum_steps * cfg.log_every)
            avg_cert = accum_cert_loss / (cfg.grad_accum_steps * cfg.log_every)
            avg_ortho = accum_ortho_loss / (cfg.grad_accum_steps * cfg.log_every)
            avg_bm = accum_bm_loss / (cfg.grad_accum_steps * cfg.log_every)

            tokens_per_sec = accum_tokens / elapsed_time_since_last_log
            total_tokens += accum_tokens
            current_lr = optimizer.param_groups[0]["lr"]

            if distributed:
                # Average loss across ranks
                loss_tensor = torch.tensor([avg_loss, avg_lb, avg_cert, avg_ortho, avg_bm], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss, avg_lb, avg_cert, avg_ortho, avg_bm = (
                    loss_tensor[0].item(), loss_tensor[1].item(), loss_tensor[2].item(),
                    loss_tensor[3].item(), loss_tensor[4].item()
                )

            print_rank0(
                f"step={step:>6d} | epoch={epoch} | loss={avg_loss:.4f} | lb_loss={avg_lb:.4f} | cert_loss={avg_cert:.4f} | ortho_loss={avg_ortho:.4f} | bm_loss={avg_bm:.4f} | "
                f"grad_norm={grad_norm:.3f} | lr={current_lr:.2e} | "
                f"tok/s={tokens_per_sec:.0f} | time_since_start={elapsed_time_since_training_start}",
                rank,
            )

            n_accum = cfg.grad_accum_steps * cfg.log_every
            avg_comp_metrics = {f"train/{k}": v / n_accum for k, v in accum_comp_metrics.items()}

            # Per-language metrics: reduce sums on ALL ranks (collective), then
            # compute count-weighted means for this log window
            lang_metrics = {}
            if lang_metrics_acc is not None:
                if distributed:
                    lang_metrics_acc.all_reduce()
                lang_metrics_acc.accumulate_totals()
                lang_metrics = lang_metrics_acc.metrics(prefix="", comp_prefix="train/")
                lang_metrics.update(lang_metrics_acc.total_metrics())
                lang_metrics_acc.reset()

            if wandb_project and rank == 0:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lb_loss": avg_lb,
                    "cert_loss": avg_cert,
                    "ortho_loss": avg_ortho,
                    "bm_loss": avg_bm,
                    "grad_norm": grad_norm,
                    "lr": current_lr,
                    "compression_ratio": current_downsample_n,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step,
                    "epoch": epoch,
                    "total_tokens": total_tokens,
                    "time_since_start": elapsed_time_since_training_start,
                    **grad_norm_metrics,
                    **avg_comp_metrics,
                    **collect_entropy_metrics(model),
                    **lang_metrics,
                }, step=step)

            accum_loss = 0.0
            accum_lb_loss = 0.0
            accum_cert_loss = 0.0
            accum_ortho_loss = 0.0
            accum_bm_loss = 0.0
            accum_comp_metrics = defaultdict(float)
            accum_tokens = 0
            elapsed_time_since_last_log = 0

        if cfg.get("validate_every", 0) > 0 and step % cfg.validate_every == 0:
            val_metrics = validate(model, val_dataloader, cfg, step, device,
                                   downsample_n=current_downsample_n)
            if val_metrics and wandb_project and rank == 0:
                import wandb
                wandb.log(val_metrics)
        # Checkpointing
        if step % cfg.save_every == 0:
            save_checkpoint(model, optimizer, step, cfg, cfg.checkpoint_dir, elapsed_time_since_training_start)

        if distributed:
            dist.barrier()
        
        post_step_time = time.time() - step_end_time
        elapsed_time_since_last_log += post_step_time
        elapsed_time_since_training_start += post_step_time


    # Final checkpoint
    save_checkpoint(model, optimizer, step, cfg, cfg.checkpoint_dir, elapsed_time_since_training_start)
    print_rank0("Training complete.", rank)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
