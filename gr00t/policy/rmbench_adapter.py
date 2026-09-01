# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMBench-specific policy protocol adapters."""

from typing import Any

import numpy as np

from .policy import BasePolicy, PolicyWrapper


def _validate_current_actual_arm(current_actual_arm: np.ndarray) -> None:
    if not isinstance(current_actual_arm, np.ndarray):
        raise TypeError(
            f"current actual arm state must be an ndarray, got {type(current_actual_arm)}"
        )
    if current_actual_arm.ndim != 3 or current_actual_arm.shape[1] != 1:
        raise ValueError(
            "current actual arm state must have shape [B,1,D], "
            f"got {current_actual_arm.shape}"
        )
    if not np.isfinite(current_actual_arm).all():
        raise FloatingPointError("current actual arm state contains NaN or Inf")


def _validate_decoded_absolute_arm(
    decoded_absolute_arm: np.ndarray, current_actual_arm: np.ndarray
) -> None:
    if not isinstance(decoded_absolute_arm, np.ndarray):
        raise TypeError(
            "processor decoded absolute arm action must be an ndarray, "
            f"got {type(decoded_absolute_arm)}"
        )
    if decoded_absolute_arm.ndim != 3:
        raise ValueError(
            "processor decoded absolute arm action must have shape [B,H,D], "
            f"got {decoded_absolute_arm.shape}"
        )
    if (
        decoded_absolute_arm.shape[0] != current_actual_arm.shape[0]
        or decoded_absolute_arm.shape[2] != current_actual_arm.shape[2]
    ):
        raise ValueError(
            "processor decoded arm action and current actual arm state must match "
            f"in batch and dimension, got {decoded_absolute_arm.shape} and "
            f"{current_actual_arm.shape}"
        )
    if not np.isfinite(decoded_absolute_arm).all():
        raise FloatingPointError("processor decoded absolute arm action contains NaN or Inf")


class RMBenchDecodedActionPolicyWrapper(PolicyWrapper):
    """Validate and expose processor-decoded absolute actions without conversion."""

    _ARM_STATE_KEY = "joint_position"
    _ARM_ACTION_KEY = "joint_position"
    _ARM_DIM = 12

    def __init__(self, policy: BasePolicy, *, strict: bool = True):
        super().__init__(policy, strict=strict)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            current_actual_arm = observation["state"][self._ARM_STATE_KEY]
        except KeyError as exc:
            raise KeyError(
                f"observation is missing state.{self._ARM_STATE_KEY} actual qpos"
            ) from exc
        _validate_current_actual_arm(current_actual_arm)
        if current_actual_arm.shape[2] != self._ARM_DIM:
            raise ValueError(
                f"RMBench actual arm state must be 12D, got {current_actual_arm.shape}"
            )

        decoded_action, diagnostics = self.policy.get_action(observation, options)
        if self._ARM_ACTION_KEY not in decoded_action:
            raise KeyError(f"processor decoded action is missing {self._ARM_ACTION_KEY!r}")

        decoded_absolute_arm = decoded_action[self._ARM_ACTION_KEY]
        _validate_decoded_absolute_arm(decoded_absolute_arm, current_actual_arm)

        diagnostics = dict(diagnostics or {})
        diagnostics["arm_boundary_adapter"] = {
            "state_source": f"observation.state.{self._ARM_STATE_KEY}",
            "processor_output_semantics": "absolute",
            "boundary_conversion": "none",
            "external_semantics": "absolute",
            "current_actual_arm": current_actual_arm[:, 0, :].tolist(),
            "processor_decoded_arm_first": decoded_absolute_arm[:, 0, :].tolist(),
            "external_returned_arm_first": decoded_absolute_arm[:, 0, :].tolist(),
        }
        return decoded_action, diagnostics

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self):
        return getattr(self.policy, "get_modality_config", lambda: {})()
