# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMBench decoded-absolute action boundary tests."""

import socket
import threading

from gr00t.eval.run_gr00t_server import (
    RMBENCH_ADAPTER_VERSION,
    ServerConfig,
    _build_server_metadata,
)
from gr00t.policy.rmbench_adapter import RMBenchDecodedActionPolicyWrapper
from gr00t.policy.server_client import MsgSerializer, PolicyClient, PolicyServer
import numpy as np
import pytest


def _current_actual_arm() -> np.ndarray:
    return np.full((1, 1, 12), 0.3, dtype=np.float32)


def _decoded_absolute_horizon() -> np.ndarray:
    return np.stack(
        (
            np.full(12, 0.4, dtype=np.float32),
            np.full(12, 0.5, dtype=np.float32),
        ),
        axis=0,
    )[None, :, :]


class _DecodedAbsolutePolicy:
    strict = False

    def __init__(self, *, nonfinite_action: bool = False):
        self.calls = 0
        self.nonfinite_action = nonfinite_action
        self.gripper = np.array([[[0.25, 0.75], [0.5, 1.0]]], dtype=np.float32)
        self.last_decoded_arm = None

    def get_action(self, observation, options=None):
        self.calls += 1
        decoded_arm = _decoded_absolute_horizon() + (self.calls - 1)
        if self.nonfinite_action:
            decoded_arm[0, 0, 0] = np.inf
        self.last_decoded_arm = decoded_arm
        return {
            "joint_position": decoded_arm,
            "gripper_close": self.gripper,
        }, {"inference_count": self.calls}

    def reset(self, options=None):
        return {}

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def get_metadata(self):
        return {
            "adapter_version": "gr00t_policy_adapter_v4",
            "arm_action_semantics": "absolute",
            "gripper_action_semantics": "absolute",
        }


def _observation() -> dict:
    return {"state": {"joint_position": _current_actual_arm()}}


def _options() -> dict:
    return {
        "episode_id": "semantic-episode",
        "request_id": "semantic-request",
        "source_step_id": 4,
        "dry_run": True,
    }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_decoded_absolute_action_is_external_action_without_second_state_addition():
    raw_policy = _DecodedAbsolutePolicy()
    adapter = RMBenchDecodedActionPolicyWrapper(raw_policy, strict=False)

    action, diagnostics = adapter.get_action(_observation())

    assert action["joint_position"] is raw_policy.last_decoded_arm
    np.testing.assert_array_equal(action["joint_position"], _decoded_absolute_horizon())
    np.testing.assert_allclose(action["joint_position"][:, 0], 0.4)
    assert not np.allclose(action["joint_position"][:, 0], 0.7)
    boundary = diagnostics["arm_boundary_adapter"]
    assert boundary["processor_decoded_arm_first"] == boundary["external_returned_arm_first"]


def test_decoded_absolute_action_preserves_shape():
    adapter = RMBenchDecodedActionPolicyWrapper(_DecodedAbsolutePolicy(), strict=False)
    action, _ = adapter.get_action(_observation())
    assert action["joint_position"].shape == (1, 2, 12)


def test_boundary_adapter_leaves_gripper_untouched_and_reports_no_conversion():
    raw_policy = _DecodedAbsolutePolicy()
    adapter = RMBenchDecodedActionPolicyWrapper(raw_policy, strict=False)

    action, diagnostics = adapter.get_action(_observation())

    assert action["gripper_close"] is raw_policy.gripper
    np.testing.assert_array_equal(action["gripper_close"], raw_policy.gripper)
    boundary = diagnostics["arm_boundary_adapter"]
    assert boundary["processor_output_semantics"] == "absolute"
    assert boundary["boundary_conversion"] == "none"
    assert "conversion_formula" not in boundary
    assert "raw_model_semantics" not in boundary


@pytest.mark.parametrize("invalid_source", ["state", "action"])
def test_boundary_rejects_nonfinite_state_or_decoded_action(invalid_source):
    raw_policy = _DecodedAbsolutePolicy(nonfinite_action=invalid_source == "action")
    adapter = RMBenchDecodedActionPolicyWrapper(raw_policy, strict=False)
    observation = _observation()
    if invalid_source == "state":
        observation["state"]["joint_position"][0, 0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        adapter.get_action(observation)

    assert raw_policy.calls == (0 if invalid_source == "state" else 1)


def test_decoded_response_is_idempotent_and_echoes_freshness_on_wire():
    raw_policy = _DecodedAbsolutePolicy()
    adapter = RMBenchDecodedActionPolicyWrapper(raw_policy, strict=False)
    port = _free_port()
    options = _options()

    with PolicyServer(adapter, host="127.0.0.1", port=port) as server:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        with PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000) as client:
            first = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            retry = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            client.kill_server()
        thread.join(timeout=2)

    assert raw_policy.calls == 1
    assert MsgSerializer.to_bytes(first) == MsgSerializer.to_bytes(retry)
    np.testing.assert_array_equal(first[0]["joint_position"], retry[0]["joint_position"])
    for field in ("episode_id", "request_id", "source_step_id"):
        assert first[1][field] == options[field]


def test_reset_clears_cached_decoded_response():
    raw_policy = _DecodedAbsolutePolicy()
    adapter = RMBenchDecodedActionPolicyWrapper(raw_policy, strict=False)
    port = _free_port()
    options = _options()

    with PolicyServer(adapter, host="127.0.0.1", port=port) as server:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        with PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000) as client:
            first = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            client.reset({"episode_id": options["episode_id"]})
            after_reset = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            client.kill_server()
        thread.join(timeout=2)

    assert raw_policy.calls == 2
    assert not np.array_equal(first[0]["joint_position"], after_reset[0]["joint_position"])


def test_adapter_v4_identity_declares_decoded_absolute_external_actions():
    config = ServerConfig(
        state_arm_semantics="actual_qpos",
        state_gripper_semantics="physical",
        arm_action_semantics="absolute",
        gripper_action_semantics="absolute",
        decoded_absolute_action_boundary=True,
    )

    metadata = _build_server_metadata(config)

    assert RMBENCH_ADAPTER_VERSION == "gr00t_policy_adapter_v4"
    assert metadata["adapter_version"] == "gr00t_policy_adapter_v4"
    assert metadata["state_arm_semantics"] == "actual_qpos"
    assert metadata["state_gripper_semantics"] == "physical"
    assert metadata["arm_action_semantics"] == "absolute"
    assert metadata["gripper_action_semantics"] == "absolute"
    assert metadata["processor_decode_output_semantics"] == "absolute"
    assert metadata["boundary_conversion"] == "none"
    assert metadata["decoded_absolute_action_boundary"] is True
