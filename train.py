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
from hnet.utils.data import create_dataloaders
from hnet.utils.train import load_balancing_loss, group_params, orthogonality_regularization_soft
from hnet.utils.eval import bits_per_byte, compression_metrics
from hnet.modules.dc import RoutingModule

# Learning rate schedule: Warmup-Stable-Decay (WSD)
def wsd_schedule(step, max_steps, warmup_fraction, decay_fraction, base_lr):
    """WSD learning rate schedule with linear warmup, stable phase, and 1/sqrt decay."""
    warmup_steps = int(max_steps * warmup_fraction)
    decay_steps = int(max_steps * decay_fraction)
    stable_end = max_steps - decay_steps

    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step + 1) / warmup_steps
    elif step < stable_end:
        # Stable phase
        return base_lr
    else:
        # Inverse square root decay
        decay_progress = (step - stable_end) / max(decay_steps, 1)
        return base_lr / math.sqrt(1.0 + decay_progress * 9.0)
        # At the end (decay_progress=1), LR = base_lr / sqrt(10) ~ 0.316 * base_lr

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
    return {
        step: wsd_schedule(step, cfg.max_steps, cfg.warmup_fraction, cfg.decay_fraction, 1.0)
        for step in range(cfg.max_steps)
    }

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
def validate(model, val_dataloader, cfg, step, device):
    """Run validation and return metrics dict.

    Computes:
    - val/loss: average cross-entropy loss
    - val/bpb: bits-per-byte
    - val/lb_loss: average load-balancing loss
    - val/stage_*/F_selected, val/stage_*/G_avg_boundary_prob: compression metrics
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    distributed = dist.is_initialized()

    was_training = model.training
    model.eval()

    total_ce_loss = torch.tensor(0.0, device=device)
    total_lb_loss = torch.tensor(0.0, device=device)
    total_bm_loss = torch.tensor(0.0, device=device)
    total_ortho_loss = torch.tensor(0.0, device=device)
    total_bytes = 0
    num_batches = 0
    all_compression_metrics = {}

    val_batches = cfg.get("val_batches", 50)

    ortho_reg_lambda = cfg.get("ortho_reg_lambda", 0.0)
    if not ortho_reg_lambda:
        ortho_reg_lambda = 0.0

    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= val_batches:
            break

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids, mask=None, targets=targets)
            logits = output.logits

            # Sum (not mean) CE loss so we can compute BPB correctly
            ce_loss_sum = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            
            ortho_loss = 0.0
            if ortho_reg_lambda > 0:
                for module in model.modules():
                    if isinstance(module, RoutingModule):
                        ortho_loss += orthogonality_regularization_soft(module.q_proj_layer.weight)
                        ortho_loss += orthogonality_regularization_soft(module.k_proj_layer.weight)

            lb_loss = torch.tensor(0.0, device=device)
            bm_loss = torch.tensor(0.0, device=device)

            if output.bpred_output:
                for router_out in output.bpred_output:
                    lb_loss = lb_loss + load_balancing_loss(
                        router_out, N=cfg.downsample_n
                    )
                    if router_out.bm_loss is not None:
                        bm_loss += router_out.bm_loss
                
                lb_loss = lb_loss / len(output.bpred_output)
                bm_loss = bm_loss / len(output.bpred_output)

                # Accumulate compression metrics
                batch_comp = compression_metrics(output.bpred_output)
                for k, v in batch_comp.items():
                    all_compression_metrics.setdefault(k, []).append(v)

        total_ce_loss += ce_loss_sum
        total_lb_loss += lb_loss
        total_bm_loss += bm_loss
        total_ortho_loss += ortho_loss
        total_bytes += targets.numel()
        num_batches += 1

    if num_batches == 0:
        model.train(was_training)
        return {}

    # Aggregate across ranks
    if distributed:
        stats = torch.tensor([total_ce_loss, total_lb_loss, total_ortho_loss, total_bm_loss, float(total_bytes), float(num_batches)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_ce_loss = stats[0]
        total_lb_loss_sum = stats[1]
        total_ortho_loss = stats[2]
        total_bm_loss = stats[3]
        total_bytes = int(stats[4].item())
        num_batches = int(stats[5].item())
    else:
        total_lb_loss_sum = total_lb_loss

    bpb = bits_per_byte(total_ce_loss, total_bytes)
    avg_loss = (total_ce_loss / total_bytes).item()
    avg_lb = (total_lb_loss_sum / num_batches).item()
    avg_ortho = (total_ortho_loss / num_batches).item()
    avg_bm_loss = (total_bm_loss / num_batches).item()

    metrics = {
        "val/loss": avg_loss,
        "val/bpb": bpb,
        "val/lb_loss": avg_lb,
        "val/ortho_loss": avg_ortho,
        "val/bm_loss": avg_bm_loss,
        "step": step,
    }

    # Average compression metrics
    for k, vals in all_compression_metrics.items():
        metrics[f"val/{k}"] = sum(vals) / len(vals)

    # Entropy routing stats (running EMA + learned params)
    metrics.update(collect_entropy_metrics(model, prefix="val/"))

    print_rank0(
        f"[val] step={step:>6d} | bpb={bpb:.4f} | loss={avg_loss:.4f} | lb_loss={avg_lb:.4f} | ortho_loss={avg_ortho:.4f} | bm_loss={avg_bm_loss:.4f}",
        rank,
    )

    model.train(was_training)
    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.autograd.set_detect_anomaly(True)
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
    dataloader, val_dataloader = create_dataloaders(
        Path(cfg.data_dir),
        cfg.get("dataset_config", "sample-10BT"),
        cfg.seq_len,
        cfg.seed,
        cfg.get("val_batches", 50),
        cfg.batch_size,
        cfg.num_workers
    )

    # ---- Resume ----
    start_step = 0
    time_since_start = 0
    if cfg.resume:
        start_step, time_since_start = load_checkpoint(model, optimizer, cfg.checkpoint_dir, device)

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
    accum_ortho_loss = 0.0
    accum_bm_loss = 0.0
    accum_comp_metrics = defaultdict(float)
    accum_tokens = 0
    total_tokens = 0
    epoch = 0

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

            batch = batch.to(device)
            input_ids = batch[:, :-1]  # (B, seq_len)
            targets = batch[:, 1:]     # (B, seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # mask=None triggers packed mode in the model
                output = model(input_ids, mask=None, targets=targets)

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
                bm_loss = torch.tensor(0.0, device=device)

                if output.bpred_output:
                    for router_out in output.bpred_output:
                        lb_loss = lb_loss + load_balancing_loss(
                            router_out, N=cfg.downsample_n
                        )
                        if router_out.entropy_mean is not None and router_out.entropy_std is not None:
                            batch_entropy_means[router_out.stage_idx, micro_step] = router_out.entropy_mean
                            batch_entropy_stds[router_out.stage_idx, micro_step] = router_out.entropy_std
                            batch_entropy_element_counts[router_out.stage_idx, micro_step] = router_out.boundary_prob.numel()
                        if router_out.bm_loss is not None:
                            bm_loss += router_out.bm_loss
                    lb_loss = lb_loss / len(output.bpred_output)
                    bm_loss = bm_loss / len(output.bpred_output)

                bm_loss_weight = cfg.get("bm_loss_weight", 0.1)
                loss = ce_loss + cfg.alpha * lb_loss + ortho_reg_lambda * ortho_loss + bm_loss_weight * bm_loss
                loss = loss / cfg.grad_accum_steps

            loss.backward()

            accum_loss += ce_loss.detach().item()
            accum_lb_loss += lb_loss.detach().item()
            accum_ortho_loss += ortho_loss.detach().item() if isinstance(ortho_loss, torch.Tensor) else ortho_loss
            accum_bm_loss += bm_loss.detach().item()
            accum_tokens += targets.numel()
            for k, v in compression_metrics(output.bpred_output).items():
                accum_comp_metrics[k] += v

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
            avg_ortho = accum_ortho_loss / (cfg.grad_accum_steps * cfg.log_every)
            avg_bm = accum_bm_loss / (cfg.grad_accum_steps * cfg.log_every)
            
            tokens_per_sec = accum_tokens / elapsed_time_since_last_log
            total_tokens += accum_tokens
            current_lr = optimizer.param_groups[0]["lr"]

            if distributed:
                # Average loss across ranks
                loss_tensor = torch.tensor([avg_loss, avg_lb, avg_ortho, avg_bm], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss, avg_lb, avg_ortho, avg_bm = loss_tensor[0].item(), loss_tensor[1].item(), loss_tensor[2].item(), loss_tensor[3].item()

            print_rank0(
                f"step={step:>6d} | epoch={epoch} | loss={avg_loss:.4f} | lb_loss={avg_lb:.4f} | ortho_loss={avg_ortho:.4f} | bm_loss={avg_bm:.4f} | "
                f"grad_norm={grad_norm:.3f} | lr={current_lr:.2e} | "
                f"tok/s={tokens_per_sec:.0f} | time_since_start={elapsed_time_since_training_start}",
                rank,
            )

            n_accum = cfg.grad_accum_steps * cfg.log_every
            avg_comp_metrics = {f"train/{k}": v / n_accum for k, v in accum_comp_metrics.items()}

            if wandb_project and rank == 0:
                import wandb
                wandb.log({
                    "loss": avg_loss,
                    "lb_loss": avg_lb,
                    "ortho_loss": avg_ortho,
                    "bm_loss": avg_bm,
                    "grad_norm": grad_norm,
                    "lr": current_lr,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step,
                    "epoch": epoch,
                    "total_tokens": total_tokens,
                    "time_since_start": elapsed_time_since_training_start,
                    **grad_norm_metrics,
                    **avg_comp_metrics,
                    **collect_entropy_metrics(model),
                }, step=step)

            accum_loss = 0.0
            accum_lb_loss = 0.0
            accum_ortho_loss = 0.0
            accum_bm_loss = 0.0
            accum_comp_metrics = defaultdict(float)
            accum_tokens = 0
            elapsed_time_since_last_log = 0

        if cfg.get("validate_every", 0) > 0 and step % cfg.validate_every == 0:
            val_metrics = validate(model, val_dataloader, cfg, step, device)
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
