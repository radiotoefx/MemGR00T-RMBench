# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RoboTTT fast-weight memory and GR00T DiT integration."""

from gr00t.model.modules.dit import DiT
from gr00t.model.modules.robottt import RoboTTTLayer
import torch


def _layer(gate_init: float = 0.2) -> RoboTTTLayer:
    torch.manual_seed(7)
    return RoboTTTLayer(
        model_dim=16,
        ttt_dim=8,
        fast_hidden_dim=12,
        base_lr=0.1,
        gate_init=gate_init,
    )


def test_zero_gate_is_exact_identity() -> None:
    layer = _layer(gate_init=0.0)
    inputs = torch.randn(2, 5, 16)
    outputs = layer(inputs)
    assert torch.equal(outputs, inputs)
    assert layer.state_step == 1


def test_state_persists_and_reset_is_reproducible() -> None:
    layer = _layer()
    inputs = torch.randn(2, 5, 16)

    first = layer(inputs)
    first_state = tuple(value.clone() for value in layer.fast_state.tensors())
    second = layer(inputs)

    assert layer.state_step == 2
    assert not torch.equal(first, second)
    assert any(
        not torch.equal(old, new) for old, new in zip(first_state, layer.fast_state.tensors())
    )

    layer.reset_state()
    replay = layer(inputs)
    torch.testing.assert_close(first, replay)
    assert layer.state_step == 1


def test_update_can_be_disabled_without_mutating_state() -> None:
    layer = _layer()
    inputs = torch.randn(1, 4, 16)
    layer(inputs)
    before = tuple(value.clone() for value in layer.fast_state.tensors())

    layer(inputs, update_state=False)

    assert layer.state_step == 1
    assert all(torch.equal(old, new) for old, new in zip(before, layer.fast_state.tensors()))


def test_inner_update_runs_inside_outer_no_grad() -> None:
    layer = _layer()
    with torch.no_grad():
        output = layer(torch.randn(1, 4, 16))

    assert output.grad_fn is None
    assert layer.state_step == 1


def test_residual_diagnostics_observe_effective_dtype_path() -> None:
    layer = _layer(gate_init=0.001).to(dtype=torch.bfloat16)
    inputs = torch.randn(1, 4, 16, dtype=torch.bfloat16)

    outputs = layer(inputs)

    torch.testing.assert_close(
        (outputs.float() - inputs.float()).norm(),
        torch.tensor(layer.last_effective_residual_norm),
    )
    assert layer.last_ttt_output_norm > 0
    assert layer.last_gated_residual_norm > 0
    assert 0 <= layer.last_effective_nonzero_fraction <= 1


def test_explicit_initialization_repairs_uninitialized_checkpoint_additions() -> None:
    layer = _layer(gate_init=0.0)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(float("nan"))

    layer.reset_parameters()

    assert all(torch.isfinite(parameter).all() for parameter in layer.parameters())
    assert torch.equal(layer.gate, torch.zeros_like(layer.gate))


def test_outer_gradient_meta_learns_initialization_and_projections() -> None:
    layer = _layer()
    loss = layer(torch.randn(2, 3, 16)).square().mean()
    loss.backward()

    assert layer.fast_w1_init.grad is not None
    assert layer.q_proj.weight.grad is not None
    assert layer.k_proj.weight.grad is not None
    assert layer.v_proj.weight.grad is not None
    assert layer.gate.grad is not None


def test_detach_state_truncates_graph_without_resetting_values() -> None:
    layer = _layer()
    layer(torch.randn(1, 3, 16, requires_grad=True))
    before = tuple(value.clone() for value in layer.fast_state.tensors())
    layer.detach_state()

    assert layer.state_step == 1
    assert all(value.grad_fn is None for value in layer.fast_state.tensors())
    assert all(torch.equal(old, new) for old, new in zip(before, layer.fast_state.tensors()))


def test_fast_state_snapshot_restore_is_independent_and_reproducible() -> None:
    layer = _layer()
    inputs = torch.randn(1, 3, 16)
    first = layer(inputs)
    snapshot = layer.clone_state()
    assert snapshot is not None

    second = layer(inputs)
    layer.load_state(snapshot)
    replay_second = layer(inputs)

    torch.testing.assert_close(second, replay_second)
    assert layer.state_step == 2
    assert not torch.equal(first, second)


def test_batched_fast_states_match_independent_episode_chains() -> None:
    """Batching sequences must not mix memory between episode items."""
    batched = _layer()
    single_a = _layer()
    single_b = _layer()
    hidden = torch.randn(2, 5, 16)

    batched_output = batched(hidden)
    output_a = single_a(hidden[:1])
    output_b = single_b(hidden[1:])

    torch.testing.assert_close(batched_output[:1], output_a)
    torch.testing.assert_close(batched_output[1:], output_b)
    assert batched.fast_state is not None
    assert single_a.fast_state is not None
    assert single_b.fast_state is not None
    for batched_value, value_a, value_b in zip(
        batched.fast_state.tensors(),
        single_a.fast_state.tensors(),
        single_b.fast_state.tensors(),
    ):
        torch.testing.assert_close(batched_value[:1], value_a)
        torch.testing.assert_close(batched_value[1:], value_b)


def test_dit_updates_ttt_once_when_requested() -> None:
    model = DiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=16,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        positional_embeddings=None,
        interleave_self_attention=False,
        cross_attention_dim=16,
        ttt_config={
            "enabled": True,
            "num_layers": 1,
            "dim": 8,
            "hidden_dim": 12,
            "base_lr": 0.1,
            "gate_init": 0.0,
        },
    )
    hidden = torch.randn(1, 4, 16)
    encoder = torch.randn(1, 5, 16)
    timestep = torch.zeros(1, dtype=torch.long)

    model(hidden, encoder, timestep, ttt_update_enabled=True)
    assert len(model.ttt_layers()) == 1
    assert model.ttt_layers()[0].state_step == 1

    model(hidden, encoder, timestep, ttt_update_enabled=False)
    assert model.ttt_layers()[0].state_step == 1

    model.reset_ttt_state()
    assert model.ttt_layers()[0].state_step == 0
