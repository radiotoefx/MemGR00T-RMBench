# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful test-time-training memory for GR00T action tokens.

This is a small, explicit implementation of the key-value-binding (KVB)
mechanism described in RoboTTT.  Slow projections produce queries, keys and
values.  A per-environment two-layer MLP is updated on ``K -> V`` with one
gradient step, then queried with ``Q``.  The updated MLP parameters are the
recurrent state carried between robot timesteps.

Runtime fast weights deliberately are not parameters or buffers: checkpoints
contain the meta-learned initialization W0, but never an in-progress episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TTTFastState:
    """Batched fast weights belonging to one TTT layer."""

    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    step: int = 0

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.w1, self.b1, self.w2, self.b2

    def detached(self) -> "TTTFastState":
        return TTTFastState(*(value.detach() for value in self.tensors()), step=self.step)


class RoboTTTLayer(nn.Module):
    """RoboTTT KVB layer with a two-layer GeLU fast model.

    The inner loss is averaged over all tokens from one robot timestep and one
    update is committed per call.  This treats the state/action token set as an
    inner-loop minibatch while keeping the recurrent boundary at environment
    timesteps.
    """

    def __init__(
        self,
        model_dim: int,
        *,
        ttt_dim: int = 256,
        fast_hidden_dim: int = 1024,
        base_lr: float = 0.1,
        gate_init: float = 0.001,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if ttt_dim <= 0 or fast_hidden_dim <= 0:
            raise ValueError("ttt_dim and fast_hidden_dim must be positive")

        self.model_dim = model_dim
        self.ttt_dim = ttt_dim
        self.fast_hidden_dim = fast_hidden_dim
        self.base_lr = base_lr
        self.gate_init = gate_init

        self.norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.q_proj = nn.Linear(model_dim, ttt_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, ttt_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, ttt_dim, bias=False)
        self.out_proj = nn.Linear(ttt_dim, model_dim, bias=False)

        # Meta-learned fast-weight initialization W0.
        self.fast_w1_init = nn.Parameter(torch.empty(ttt_dim, fast_hidden_dim))
        self.fast_b1_init = nn.Parameter(torch.zeros(fast_hidden_dim))
        self.fast_w2_init = nn.Parameter(torch.empty(fast_hidden_dim, ttt_dim))
        self.fast_b2_init = nn.Parameter(torch.zeros(ttt_dim))
        # A learned positive multiplier on the paper's constant base LR.
        self.log_lr_multiplier = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.full((model_dim,), float(gate_init)))

        self._fast_state: TTTFastState | None = None
        self._last_update_norm: float = 0.0
        self._last_ttt_output_norm: float = 0.0
        self._last_gated_residual_norm: float = 0.0
        self._last_effective_residual_norm: float = 0.0
        self._last_effective_residual_fraction: float = 0.0
        self._last_effective_nonzero_fraction: float = 0.0
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize slow projections and W0, including after meta-device loading."""
        self.norm.reset_parameters()
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()
        nn.init.normal_(self.fast_w1_init, mean=0.0, std=0.02)
        nn.init.zeros_(self.fast_b1_init)
        nn.init.normal_(self.fast_w2_init, mean=0.0, std=0.02)
        nn.init.zeros_(self.fast_b2_init)
        nn.init.zeros_(self.log_lr_multiplier)
        nn.init.constant_(self.gate, float(self.gate_init))
        self.reset_state()

    @property
    def fast_state(self) -> TTTFastState | None:
        return self._fast_state

    @property
    def state_step(self) -> int:
        return 0 if self._fast_state is None else self._fast_state.step

    @property
    def last_update_norm(self) -> float:
        return self._last_update_norm

    @property
    def last_ttt_output_norm(self) -> float:
        return self._last_ttt_output_norm

    @property
    def last_gated_residual_norm(self) -> float:
        return self._last_gated_residual_norm

    @property
    def last_effective_residual_norm(self) -> float:
        return self._last_effective_residual_norm

    @property
    def last_effective_residual_fraction(self) -> float:
        return self._last_effective_residual_fraction

    @property
    def last_effective_nonzero_fraction(self) -> float:
        return self._last_effective_nonzero_fraction

    def reset_state(self, batch_size: int | None = None) -> None:
        """Reset to W0, or clear state for lazy initialization."""
        self._fast_state = None if batch_size is None else self._initial_state(batch_size)
        self._last_update_norm = 0.0
        self._last_ttt_output_norm = 0.0
        self._last_gated_residual_norm = 0.0
        self._last_effective_residual_norm = 0.0
        self._last_effective_residual_fraction = 0.0
        self._last_effective_nonzero_fraction = 0.0

    def detach_state(self) -> None:
        """Truncate outer gradients while preserving the fast-weight values."""
        if self._fast_state is not None:
            self._fast_state = self._fast_state.detached()

    def clone_state(self) -> TTTFastState | None:
        """Return an independent detached snapshot for paired diagnostics."""
        if self._fast_state is None:
            return None
        return TTTFastState(
            *(value.detach().clone() for value in self._fast_state.tensors()),
            step=self._fast_state.step,
        )

    def load_state(self, state: TTTFastState) -> None:
        """Restore an independent detached fast-state snapshot."""
        self._fast_state = TTTFastState(
            *(value.detach().clone() for value in state.tensors()),
            step=state.step,
        )

    def _initial_state(self, batch_size: int) -> TTTFastState:
        # Inner optimization is kept in fp32 even when the VLA runs in bf16.
        def batched(value: torch.Tensor) -> torch.Tensor:
            return value.float().unsqueeze(0).expand(batch_size, *value.shape)

        return TTTFastState(
            w1=batched(self.fast_w1_init),
            b1=batched(self.fast_b1_init),
            w2=batched(self.fast_w2_init),
            b2=batched(self.fast_b2_init),
        )

    def _state_for(self, batch_size: int) -> TTTFastState:
        if self._fast_state is None or self._fast_state.w1.shape[0] != batch_size:
            self._fast_state = self._initial_state(batch_size)
        return self._fast_state

    @staticmethod
    def _fast_model(x: torch.Tensor, state: TTTFastState) -> torch.Tensor:
        hidden = torch.einsum("bnd,bdh->bnh", x, state.w1) + state.b1[:, None]
        hidden = F.gelu(hidden)
        return torch.einsum("bnh,bhd->bnd", hidden, state.w2) + state.b2[:, None]

    def _updated_state(
        self, keys: torch.Tensor, values: torch.Tensor, state: TTTFastState
    ) -> TTTFastState:
        outer_grad_enabled = torch.is_grad_enabled()

        # get_action() is no_grad, but a real TTT inner gradient must still run.
        # In inference the local leaves are detached again before being stored.
        if not outer_grad_enabled:
            state = TTTFastState(
                *(value.detach().requires_grad_(True) for value in state.tensors()),
                step=state.step,
            )
        elif not all(value.requires_grad for value in state.tensors()):
            state = TTTFastState(
                *(value.detach().requires_grad_(True) for value in state.tensors()),
                step=state.step,
            )

        with torch.enable_grad():
            prediction = self._fast_model(keys.float(), state)
            per_environment_loss = (prediction - values.float()).square().mean(dim=(1, 2))
            gradients = torch.autograd.grad(
                per_environment_loss.sum(),
                state.tensors(),
                create_graph=outer_grad_enabled,
            )
            inner_lr = self.base_lr * self.log_lr_multiplier.exp()
            updated_tensors = tuple(
                value - inner_lr * gradient for value, gradient in zip(state.tensors(), gradients)
            )

        update_sq_norm = sum((new - old).float().square().sum() for new, old in zip(updated_tensors, state.tensors()))
        self._last_update_norm = float(update_sq_norm.detach().sqrt().cpu())
        updated = TTTFastState(*updated_tensors, step=state.step + 1)
        return updated if outer_grad_enabled else updated.detached()

    def forward(self, hidden_states: torch.Tensor, *, update_state: bool = True) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, dim], got {tuple(hidden_states.shape)}")

        normalized = self.norm(hidden_states)
        queries = self.q_proj(normalized)
        keys = self.k_proj(normalized)
        values = self.v_proj(normalized)
        state = self._state_for(hidden_states.shape[0])

        if update_state:
            state = self._updated_state(keys, values, state)
            self._fast_state = state

        ttt_output = self._fast_model(queries.float(), state).to(hidden_states.dtype)
        ttt_output = self.out_proj(ttt_output)
        gated_residual = torch.tanh(self.gate).view(1, 1, -1) * ttt_output
        merged = hidden_states + gated_residual

        # These diagnostics deliberately observe the production dtype path. In
        # BF16, a non-zero mathematical residual can round away at the add.
        with torch.no_grad():
            effective_residual = merged.float() - hidden_states.float()
            hidden_norm = hidden_states.float().norm().clamp_min(1e-12)
            self._last_ttt_output_norm = float(ttt_output.float().norm().cpu())
            self._last_gated_residual_norm = float(gated_residual.float().norm().cpu())
            self._last_effective_residual_norm = float(effective_residual.norm().cpu())
            self._last_effective_residual_fraction = float(
                (effective_residual.norm() / hidden_norm).cpu()
            )
            self._last_effective_nonzero_fraction = float(
                effective_residual.ne(0).float().mean().cpu()
            )
        return merged
