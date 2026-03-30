"""
This file contains utility functions for training.

NOTE: This file is not used inside the HNet package, but contains useful utilities for training the model itself.
"""

import torch

from hnet.modules.dc import RoutingModuleOutput
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.modules.utils import apply_optimization_params


def get_compression_ratio(step: int, max_steps: int, schedule: list) -> float:
    """Return the current compression ratio by piecewise-linear interpolation.

    Args:
        step:      Current training step (0-indexed).
        max_steps: Total number of training steps.
        schedule:  List of [fraction, ratio] waypoints, e.g.
                   [[0.0, 2], [0.15, 2], [0.30, 4], [0.45, 4], [0.60, 6], [1.0, 6]]
                   Fractions must be in [0, 1] and strictly non-decreasing.

    Returns:
        The interpolated compression ratio as a float.
    """
    frac = step / max(max_steps, 1)
    # Clamp to the defined range
    if frac <= schedule[0][0]:
        return float(schedule[0][1])
    if frac >= schedule[-1][0]:
        return float(schedule[-1][1])
    # Find the bracketing waypoints and interpolate
    for i in range(len(schedule) - 1):
        f0, r0 = schedule[i]
        f1, r1 = schedule[i + 1]
        if f0 <= frac <= f1:
            if f1 == f0:
                return float(r1)
            t = (frac - f0) / (f1 - f0)
            return float(r0 + t * (r1 - r0))
    return float(schedule[-1][1])

def load_balancing_loss(
    router_output: RoutingModuleOutput,
    N: float,
) -> torch.Tensor:
    """
    Compute the load balancing loss.
    
    NOTE: This is the loss we used for all experiments. It computes the loss on each device/minibatch, and then averages the loss over all devices/minibatches.
    It is possible that computing the loss on each example is better, or that computing the statistics over the entire (global) batch would have been better.

    Args:
        router_output: The output of the routing module.
        N: The number of "experts", i.e. the downsampling factor. Can be a float (e.g. 2.5) or an integer (e.g. 3), but must be greater than 1.

    Returns:
        A single tensor, the load balancing loss.
    """
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask

    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()

    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N-1)
    ) * N / (N-1)

def certainty_loss(router_output: RoutingModuleOutput) -> torch.Tensor:
    """
    Compute the binary entropy (certainty) loss over boundary probabilities.

    Minimising this encourages the router to make confident (near-0 or near-1)
    boundary decisions without imposing a target compression ratio.

    Args:
        router_output: The output of the routing module.

    Returns:
        A scalar tensor: mean binary entropy H_b(p_t) over all positions.
    """
    p = router_output.boundary_prob[..., 1].float().clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


# Keeping the effective rank of BP matrices high
def orthogonality_regularization_soft(W):
    """Penalizes singular values drifting from 1."""
    WTW = W.T @ W
    I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    return torch.norm(WTW - I, p='fro') ** 2

def group_params(
    model: HNetForCausalLM,
) -> list[dict[str, list[torch.Tensor] | float]]:
    """
    Creates parameter groups for the optimizer, based on the learning rate multiplier and weight decay.

    Each parameter group has the following form: 
    {
        "params": [list of parameters],
        "lr": learning rate
        "weight_decay": weight decay,
    }

    Inputs:
        model: The model to group parameters for.
        lr_multiplier: A list of learning rate multipliers, one for each stage of the hierarchy, with the outer stages first (e.g. [3.0, 1.7, 0.9]).
        weight_decay: The weight decay to apply to all parameters (except bias + norms)

    Returns:
        A list of parameter groups, each with the above form.
    """
    param_groups = []
    all_keys = set()

    for name, param in model.named_parameters():
        if name.endswith(".bias") or ".norm." in name:
            apply_optimization_params(param, weight_decay=0.0)
        
        all_keys.update(param._optim.keys())
    
    all_keys = list(all_keys)
    all_tuples = []
    param_groups = []

    for name, param in model.named_parameters():
        current_tuple = tuple(param._optim.get(key, None) for key in all_keys)
        if current_tuple not in all_tuples:
            all_tuples.append(current_tuple)
            param_groups.append({
                "params": [param],
                **param._optim,
            })
        else:
            idx = all_tuples.index(current_tuple)
            param_groups[idx]["params"].append(param)
    
    return param_groups

