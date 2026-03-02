"""Residual stream probing: extract z_H, z_H*, z_L, and input embeddings via hooks.

Provides hook callbacks that capture residual streams at each ACT step for
training a probing MLP to measure H-level information gain.

Usage:
    from probe_train import create_probe_hooks, ProbeData

    hooks, storage = create_probe_hooks(skip_steps=2, to_cpu=True)
    remove_l = model.register_hook_L(hooks['L'])
    remove_h = model.register_hook_H(hooks['H'])

    # Run ACT loop...
    # storage now contains ProbeData per ACT step
"""
from typing import Dict, List
from dataclasses import dataclass

import torch

from models.hrm.hrm_act_v2 import HookContext, HookCallback


@dataclass
class ProbeData:
    """Collected residual stream data for a single ACT step."""
    act_step: int
    z_H_before: torch.Tensor   # [batch, seq+puzzle_emb_len, hidden_size]
    z_H_after: torch.Tensor    # [batch, seq+puzzle_emb_len, hidden_size]
    z_L: torch.Tensor          # [batch, seq+puzzle_emb_len, hidden_size]
    input_embeddings: torch.Tensor  # [batch, seq+puzzle_emb_len, hidden_size]


def create_probe_hooks(
    skip_steps: int = 2,
    to_cpu: bool = True,
) -> tuple[Dict[str, HookCallback], List[ProbeData]]:
    """Create L and H hook callbacks for residual stream extraction.

    The L-hook captures z_H_before (carry-in z_H before H-level) and z_L
    at the grad-step (last L/H cycle). The H-hook captures z_H_after
    (z_H* after H-level) at the grad-step.

    Args:
        skip_steps: Skip ACT steps < skip_steps (early steps are unstable).
        to_cpu: If True, detach and move tensors to CPU to avoid OOM.

    Returns:
        (hooks_dict, storage) where hooks_dict has 'L' and 'H' keys,
        and storage accumulates ProbeData per qualifying ACT step.
    """
    storage: List[ProbeData] = []
    # L-hook fires before H-hook within the same ACT step; buffer partial data
    _pending: Dict[int, Dict[str, torch.Tensor]] = {}

    def _detach(t: torch.Tensor) -> torch.Tensor:
        d = t.detach()
        return d.cpu() if to_cpu else d

    def l_hook(ctx: HookContext) -> None:
        """Capture z_H_before and z_L at the grad-step L-hook."""
        if not ctx.is_grad_step:
            return
        if ctx.act_step < skip_steps:
            return
        # ctx.z_H here is z_H BEFORE H-level (the carry-in)
        # ctx.z_L here is z_L AFTER the final L-level call
        _pending[ctx.act_step] = {
            'z_H_before': _detach(ctx.z_H),
            'z_L': _detach(ctx.z_L),
            'input_embeddings': _detach(ctx.input_embeddings) if ctx.input_embeddings is not None else torch.zeros_like(ctx.z_H.detach()),
        }

    def h_hook(ctx: HookContext) -> None:
        """Capture z_H_after at the grad-step H-hook and assemble ProbeData."""
        if not ctx.is_grad_step:
            return
        if ctx.act_step < skip_steps:
            return
        # ctx.z_H here is z_H AFTER H-level (z_H*)
        pending = _pending.pop(ctx.act_step, None)
        if pending is None:
            return
        storage.append(ProbeData(
            act_step=ctx.act_step,
            z_H_before=pending['z_H_before'],
            z_H_after=_detach(ctx.z_H),
            z_L=pending['z_L'],
            input_embeddings=pending['input_embeddings'],
        ))

    hooks: Dict[str, HookCallback] = {'L': l_hook, 'H': h_hook}
    return hooks, storage
