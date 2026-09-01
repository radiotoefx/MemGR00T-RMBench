# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ordered RoboTTT collation and TBPTT lifecycle."""

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
import numpy as np
import pytest
import torch
from torch import nn
from transformers import BatchFeature


def test_sequence_collator_transposes_batch_into_timesteps() -> None:
    collator = object.__new__(Gr00tN1d7DataCollator)
    features = [
        {
            "robottt_sequence": [
                {"state": np.array([batch * 10 + timestep], dtype=np.float32)}
                for timestep in range(3)
            ]
        }
        for batch in range(2)
    ]

    result = collator(features)["inputs"]["robottt_sequence"]

    assert len(result) == 3
    torch.testing.assert_close(result[0]["state"].squeeze(-1), torch.tensor([0.0, 10.0]))
    torch.testing.assert_close(result[2]["state"].squeeze(-1), torch.tensor([2.0, 12.0]))


def test_sequence_collator_rejects_mixed_sequence_and_single_step() -> None:
    collator = object.__new__(Gr00tN1d7DataCollator)
    features = [
        {"robottt_sequence": [{"state": np.array([0.0], dtype=np.float32)}]},
        {"state": np.array([1.0], dtype=np.float32)},
    ]

    with pytest.raises(ValueError, match="Cannot mix RoboTTT sequences and single steps"):
        collator(features)


def test_sequence_collator_rejects_unequal_lengths() -> None:
    collator = object.__new__(Gr00tN1d7DataCollator)
    features = [
        {"robottt_sequence": [{"state": np.array([0.0], dtype=np.float32)}]},
        {
            "robottt_sequence": [
                {"state": np.array([1.0], dtype=np.float32)},
                {"state": np.array([2.0], dtype=np.float32)},
            ]
        },
    ]

    with pytest.raises(ValueError, match="RoboTTT sequences must have equal length"):
        collator(features)


class _ActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, backbone_output, action_input):
        self.calls += 1
        value = action_input["value"] * self.scale
        return {
            "loss": value.mean(),
            "action_loss": value[:, None, None],
            "action_mask": torch.ones_like(value[:, None, None]),
        }


class _SequenceHarness(nn.Module):
    forward_robottt_sequence = Gr00tN1d7.forward_robottt_sequence

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_ttt=True,
            ttt_sequence_training=True,
            ttt_tbptt_segment_length=2,
            ttt_decision_loss_start_fraction=1.0,
            ttt_decision_loss_weight=1.0,
        )
        self.backbone = nn.Identity()
        self.action_head = _ActionHead()
        self.reset_calls = 0
        self.detach_calls = 0

    def reset_ttt_state(self):
        self.reset_calls += 1

    def detach_ttt_state(self):
        self.detach_calls += 1

    def prepare_input(self, inputs):
        return BatchFeature({"features": inputs["value"]}), BatchFeature(inputs)


def test_sequence_forward_resets_once_and_detaches_at_tbptt_boundaries() -> None:
    model = _SequenceHarness()
    sequence = [{"value": torch.tensor([float(t), float(t + 1)])} for t in range(5)]

    output = model.forward_robottt_sequence(sequence)
    output.loss.backward()

    assert model.reset_calls == 1
    assert model.detach_calls == 2
    assert model.action_head.calls == 5
    torch.testing.assert_close(output.timestep_loss, torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5]))
    assert model.action_head.scale.grad is not None


def test_sequence_forward_continues_detached_state_across_trainer_segments() -> None:
    model = _SequenceHarness()
    sequence = [{"value": torch.tensor([float(t), float(t + 1)])} for t in range(4)]

    first = model.forward_robottt_sequence(sequence[:2], reset_state=True)
    first.loss.backward()
    second = model.forward_robottt_sequence(sequence[2:], reset_state=False)
    second.loss.backward()

    assert model.reset_calls == 1
    assert model.detach_calls == 1
    assert model.action_head.calls == 4
    torch.testing.assert_close(first.timestep_loss, torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(second.timestep_loss, torch.tensor([2.5, 3.5]))


def test_sequence_forward_batches_frozen_backbone_over_time() -> None:
    model = _SequenceHarness()
    model.config.ttt_backbone_chunk_size = 2
    sequence = [{"value": torch.tensor([float(t), float(t + 1)])} for t in range(5)]
    backbone_batch_sizes = []
    model.backbone.register_forward_hook(
        lambda _module, inputs, _output: backbone_batch_sizes.append(inputs[0]["features"].shape[0])
    )

    output = model.forward_robottt_sequence(sequence)

    torch.testing.assert_close(output.timestep_loss, torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5]))
    assert model.action_head.calls == 5
    assert backbone_batch_sizes == [4, 4, 2]


def test_sequence_forward_applies_normalized_decision_weight() -> None:
    model = _SequenceHarness()
    model.config.ttt_decision_loss_start_fraction = 0.5
    model.config.ttt_decision_loss_weight = 3.0
    sequence = [{"value": torch.tensor([float(t)])} for t in range(4)]

    output = model.forward_robottt_sequence(sequence)

    # Weighted mean of [0, 1, 2, 3] with weights [1, 1, 3, 3].
    torch.testing.assert_close(output.loss, torch.tensor(2.0))
    torch.testing.assert_close(output.loss_weights, torch.tensor([1.0, 1.0, 3.0, 3.0]))


def test_sequence_forward_can_focus_decision_loss_on_selected_action_dims() -> None:
    model = _SequenceHarness()
    model.config.ttt_decision_loss_start_fraction = 0.5
    model.config.ttt_decision_loss_weight = 2.0
    model.config.ttt_decision_action_indices = [0]

    class PerDimensionActionHead(_ActionHead):
        def forward(self, backbone_output, action_input):
            self.calls += 1
            values = action_input["value"] * self.scale
            action_loss = values.square()[:, None, :]
            action_mask = torch.ones_like(action_loss)
            return {
                "loss": action_loss.mean(),
                "action_loss": action_loss,
                "action_mask": action_mask,
            }

    model.action_head = PerDimensionActionHead()
    sequence = [
        {"value": torch.tensor([[1.0, 10.0]])},
        {"value": torch.tensor([[2.0, 20.0]])},
    ]

    output = model.forward_robottt_sequence(sequence)

    # Full loss at t0 is (1^2 + 10^2) / 2 = 50.5. The decision loss at t1
    # selects dimension 0, so it is 2^2 = 4. Weighted mean: (50.5 + 2*4) / 3.
    torch.testing.assert_close(output.loss, torch.tensor(19.5))
    torch.testing.assert_close(output.timestep_loss, torch.tensor([50.5, 4.0]))
    torch.testing.assert_close(output.full_timestep_loss, torch.tensor([50.5, 202.0]))
