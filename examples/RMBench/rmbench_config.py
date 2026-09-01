# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 modality configuration for the RMBench LeRobot conversion."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


rmbench_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view", "left_wrist_view", "right_wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["joint_position", "gripper_close"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(rmbench_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
