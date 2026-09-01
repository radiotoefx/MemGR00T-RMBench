# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

import hashlib
import json
from pathlib import Path
import secrets
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import FINETUNE_ONLY_TAGS, POSTTRAIN_TAGS, EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData
from gr00t.utils.compact_checkpoint import (
    resolve_compact_checkpoint_chain,
    validate_finite_state_dict,
)

from .policy import BasePolicy, PolicyWrapper


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


def _sim_language_batch_to_sequence(value: Any) -> Any:
    """Normalize sim language batches while preserving validation semantics."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, str):
        return [value]
    return value


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        model_config_overrides: dict[str, Any] | None = None,
        base_model_path: str | None = None,
        processor_path: str | None = None,
        policy_seed: int | None = None,
        noise_mode: str = "independent",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type.
                Accepts an EmbodimentTag enum or a string (resolved case-insensitively).
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            processor_path: Optional checkpoint directory containing the processor.
                This is useful for evaluating an unadapted base model with exactly
                the same embodiment mapping and normalization as a finetuned model.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        model_dir = Path(model_path)
        compact_weights = model_dir / "trainable_model.safetensors"
        compact_manifest = model_dir / "trainable_model.json"

        # Load the pretrained model and move to target device with bfloat16 precision
        model_config = None
        checkpoint_load_report: list[dict[str, Any]] = []

        def unpack_pretrained(result: Any) -> tuple[Any, dict[str, Any]]:
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], dict(result[1])
            return result, {}

        def record_load(
            *,
            source: Path | str,
            model: Any,
            loading_info: dict[str, Any],
            scope: str,
            loaded_keys: list[str] | None = None,
        ) -> None:
            missing = sorted(loading_info.get("missing_keys", []))
            unexpected = sorted(loading_info.get("unexpected_keys", []))
            if loaded_keys is None:
                loaded_keys = sorted(set(model.state_dict()) - set(missing))
            report = {
                "source": str(Path(source).resolve()),
                "scope": scope,
                "loaded": loaded_keys,
                "missing": missing,
                "unexpected": unexpected,
            }
            checkpoint_load_report.append(report)
            print("Checkpoint load report: " + repr(report), flush=True)

        if compact_weights.is_file():
            if model_config_overrides:
                raise ValueError(
                    "Do not use model_config_overrides with a compact trainable checkpoint; "
                    "its saved config defines the trained TTT architecture."
                )
            if not compact_manifest.is_file():
                raise FileNotFoundError(
                    f"Compact checkpoint is missing manifest: {compact_manifest}"
                )
            compact_base_path, delta_chain = resolve_compact_checkpoint_chain(model_dir)
            compact_base_path = base_model_path or compact_base_path
            if not compact_base_path:
                raise ValueError(
                    "Compact checkpoint does not identify a base model; pass base_model_path"
                )
            model_config = AutoConfig.from_pretrained(model_dir)
            # Public N1.7 weights and policy execution are BF16. Constructing
            # the 3B base in FP32 only to cast it after applying a compact delta
            # doubles host traffic and made every offline probe spend minutes in
            # CPU conversion. Loading directly in the production dtype is
            # numerically equivalent to the former final ``model.to(bfloat16)``.
            model, base_loading_info = unpack_pretrained(
                AutoModel.from_pretrained(
                    compact_base_path,
                    config=model_config,
                    dtype=torch.bfloat16,
                    output_loading_info=True,
                )
            )
            record_load(
                source=compact_base_path,
                model=model,
                loading_info=base_loading_info,
                scope="base_model",
            )
            for delta_dir in delta_chain:
                delta = load_file(delta_dir / "trainable_model.safetensors", device="cpu")
                validate_finite_state_dict(delta, source=str(delta_dir))
                unexpected = sorted(set(delta) - set(model.state_dict()))
                if unexpected:
                    raise RuntimeError(
                        f"Compact checkpoint {delta_dir} has {len(unexpected)} unknown "
                        f"parameters: {unexpected}"
                    )
                load_result = model.load_state_dict(delta, strict=False)
                record_load(
                    source=delta_dir,
                    model=model,
                    loading_info={
                        "missing_keys": load_result.missing_keys,
                        "unexpected_keys": load_result.unexpected_keys,
                    },
                    scope="compact_delta",
                    loaded_keys=sorted(delta),
                )
        elif model_config_overrides:
            model_config = AutoConfig.from_pretrained(model_dir)
            unknown = [key for key in model_config_overrides if not hasattr(model_config, key)]
            if unknown:
                raise ValueError(f"Unknown model config override(s): {unknown}")
            for key, value in model_config_overrides.items():
                setattr(model_config, key, value)
        if compact_weights.is_file():
            pass
        elif model_config_overrides:
            model, loading_info = unpack_pretrained(
                AutoModel.from_pretrained(
                    model_dir, config=model_config, output_loading_info=True
                )
            )
            record_load(
                source=model_dir,
                model=model,
                loading_info=loading_info,
                scope="full_checkpoint_with_overrides",
            )
            ttt_missing = [
                key
                for key in loading_info.get("missing_keys", [])
                if ".ttt." in key or key.endswith("register_tokens")
            ]
            # Only a vanilla -> TTT architecture expansion needs initialization.
            # Never overwrite TTT tensors that were loaded from a trained checkpoint.
            if model_config_overrides.get("use_ttt", False) and ttt_missing:
                model.action_head.initialize_ttt_parameters()
        else:
            model, loading_info = unpack_pretrained(
                AutoModel.from_pretrained(model_dir, output_loading_info=True)
            )
            record_load(
                source=model_dir,
                model=model,
                loading_info=loading_info,
                scope="full_checkpoint",
            )
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model
        self.use_ttt = getattr(getattr(model, "config", None), "use_ttt", False) is True
        self.checkpoint_load_report = checkpoint_load_report

        if noise_mode not in {"independent", "episode_common"}:
            raise ValueError(
                "noise_mode must be 'independent' or 'episode_common', "
                f"got {noise_mode!r}"
            )
        if policy_seed is None:
            policy_seed = secrets.randbits(63)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, (int, np.integer)):
            raise TypeError("policy_seed must be an integer")
        if int(policy_seed) < 0:
            raise ValueError("policy_seed must be non-negative")
        self.noise_mode = noise_mode
        self.policy_seed = int(policy_seed)
        self._policy_cpu_generator = torch.Generator(device="cpu")
        self._policy_device_generator = torch.Generator(device=self.model.device)
        self._reseed_policy_generators(self.policy_seed)
        self._noise_episode = -1
        self._noise_query = 0
        self._episode_noise: torch.Tensor | None = None
        self._current_noise_id = ""
        self._metadata = dict(metadata or {})
        self._begin_noise_episode()

        # Load the processor for input/output transformation.
        # Training saves processor files under a "processor/" subdirectory, but
        # AutoProcessor expects them at the model root.  Fall back to the
        # subdirectory when the root lacks a processor_config.json.
        processor_model_dir = Path(processor_path) if processor_path else model_dir
        processor_dir = (
            processor_model_dir / "processor"
            if (processor_model_dir / "processor").is_dir()
            and not (processor_model_dir / "processor_config.json").exists()
            else processor_model_dir
        )
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(processor_dir)
        self.processor.eval()
        statistics_path = self._metadata.get("statistics_path")
        if statistics_path is not None:
            with Path(statistics_path).open("r") as stream:
                source_statistics = json.load(stream)
            self._metadata["stats_source_content_sha256"] = _canonical_json_sha256(
                source_statistics
            )
            self._metadata["stats_runtime_content_sha256"] = _canonical_json_sha256(
                self.processor.state_action_processor.statistics
            )

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        all_modality_configs = self.processor.get_modality_configs()
        if self.embodiment_tag.value not in all_modality_configs:
            # Map raw checkpoint tag values to user-friendly enum names where possible.
            supported_lines = []
            for tag_value in sorted(all_modality_configs.keys()):
                enum_name = EmbodimentTag.reverse_lookup(tag_value)
                if enum_name != tag_value:
                    supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
                else:
                    supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
            supported_str = "\n".join(supported_lines)

            hint = ""
            if self.embodiment_tag in POSTTRAIN_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is a posttrain tag that requires "
                    f"a finetuned checkpoint, not the base model. "
                    f"See the example READMEs for how to finetune and download checkpoints."
                )
            elif self.embodiment_tag in FINETUNE_ONLY_TAGS:
                hint = (
                    f"\n\nHint: '{self.embodiment_tag.name}' is for finetuning custom robots. "
                    f"Use it with launch_finetune.py, not with the base model directly."
                )

            raise ValueError(
                f"Embodiment tag '{self.embodiment_tag.name}' "
                f"(value='{self.embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{supported_str}"
                f"{hint}"
            )
        self.modality_configs = {
            k: v
            for k, v in all_modality_configs[self.embodiment_tag.value].items()
            if k != "rl_info"
        }
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Some embodiments (e.g. OXE_DROID) define multiple language keys for
        # training-time augmentation (paraphrases). At inference we only use the first key.
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) >= 1, "At least one language key is required"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Check that the expected state key exists in the observation
            # (must happen before indexing — see video validation above)
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Check that the expected language key exists in the observation
            # (must happen before indexing — see video validation above)
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        # inference_mode cannot be locally re-enabled for the real inner
        # gradient required by TTT. no_grad can, and RoboTTTLayer does so only
        # around its small fast model.
        inference_context = torch.no_grad() if self.use_ttt else torch.inference_mode()
        with inference_context:
            model_options = dict(options or {})
            model_options["generator"] = self._policy_device_generator
            noise_id = self._claim_noise_id()
            if self.noise_mode == "episode_common" and self._episode_noise is not None:
                model_options["initial_noise"] = self._episode_noise
            model_pred = self.model.get_action(**collated_inputs, options=model_options)
            if self.noise_mode == "episode_common" and self._episode_noise is None:
                initial_noise = model_pred.get("initial_noise")
                if initial_noise is None:
                    raise RuntimeError(
                        "episode_common requires the model to return its initial_noise"
                    )
                self._episode_noise = initial_noise.detach().clone()
        normalized_action = model_pred["action_pred"].float()
        ttt_diagnostics = self.model.ttt_diagnostics() if self.use_ttt else None
        if not torch.isfinite(normalized_action).all():
            raise FloatingPointError(
                f"Policy produced non-finite normalized actions; TTT diagnostics: {ttt_diagnostics}"
            )

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        action_diagnostics: dict[str, Any] = {}
        offset = 0
        action_params = self.processor.state_action_processor.norm_params[
            self.embodiment_tag.value
        ]["action"]
        for key in self.modality_configs["action"].modality_keys:
            dimension = int(action_params[key]["dim"].item())
            group = normalized_action[..., offset : offset + dimension]
            group_abs = group.abs()
            decoded_group = casted_action[key]
            group_metrics: dict[str, Any] = {
                "normalized_out_of_range_fraction": float((group_abs > 1.0).float().mean()),
                "normalized_boundary_fraction": float((group_abs >= 0.999).float().mean()),
                "normalized_abs_p95": float(torch.quantile(group_abs, 0.95)),
                "normalized_abs_max": float(group_abs.max()),
                # Preserve the signed, pre-decode first action.  Aggregate OOR
                # rates cannot distinguish a small persistent bias from
                # normalization clipping, which is essential for diagnosing
                # relative-action drift in closed loop.
                "normalized_first_action": group[:, 0].detach().cpu().tolist(),
                "normalized_first_action_clipped": (
                    group[:, 0].clamp(-1.0, 1.0).detach().cpu().tolist()
                ),
                "normalized_first_clip_abs_delta_max": float(
                    (group[:, 0] - group[:, 0].clamp(-1.0, 1.0)).abs().max()
                ),
            }
            if decoded_group.shape[-2] > 1:
                adjacent = np.abs(np.diff(decoded_group, axis=-2))
                group_metrics.update(
                    {
                        "decoded_adjacent_abs_mean": float(adjacent.mean()),
                        "decoded_adjacent_abs_p95": float(np.quantile(adjacent, 0.95)),
                        "decoded_adjacent_abs_max": float(adjacent.max()),
                    }
                )
                if "gripper" in key:
                    binary = (decoded_group >= 0.5).astype(np.int8)
                    group_metrics["decoded_binary_flip_count"] = float(
                        np.count_nonzero(np.diff(binary, axis=-2))
                    )
            if key in batched_states:
                current = batched_states[key][:, -1]
                signed_first_delta = decoded_group[:, 0] - current
                first_delta = np.abs(signed_first_delta)
                group_metrics["decoded_first_state_delta_mean"] = float(first_delta.mean())
                group_metrics["decoded_first_state_delta_max"] = float(first_delta.max())
                # Request-level provenance for diagnosing slow relative-action drift.
                # Lists are intentionally limited to the first action and current
                # state, so the wire/log overhead stays tiny even for long chunks.
                group_metrics["reference_state"] = current.tolist()
                group_metrics["decoded_first_target"] = decoded_group[:, 0].tolist()
                group_metrics["decoded_first_state_delta_signed"] = (
                    signed_first_delta.tolist()
                )

                state_params = self.processor.state_action_processor.norm_params[
                    self.embodiment_tag.value
                ]["state"].get(key)
                if state_params is not None and {"min", "max"} <= set(state_params):
                    state_min = np.asarray(state_params["min"])
                    state_max = np.asarray(state_params["max"])
                    state_range = np.maximum(state_max - state_min, 1e-8)
                    preclip_normalized = 2.0 * (current - state_min) / state_range - 1.0
                    preclip_oor = np.abs(preclip_normalized) > 1.0
                    group_metrics["state_preclip_out_of_range_fraction"] = float(
                        np.mean(preclip_oor)
                    )
                    group_metrics["state_preclip_normalized_abs_max"] = float(
                        np.max(np.abs(preclip_normalized))
                    )
                    # Twelve joints are cheap to send and make the first OOR
                    # event attributable to a concrete joint instead of only
                    # exposing an aggregate fraction/max.
                    group_metrics["state_preclip_normalized"] = preclip_normalized.tolist()
                    group_metrics["state_preclip_out_of_range_mask"] = preclip_oor.tolist()
                    group_metrics["state_normalization_lower"] = state_min.tolist()
                    group_metrics["state_normalization_upper"] = state_max.tolist()
                    # Backward-compatible aliases.  With use_percentiles=True
                    # these are q01/q99 normalization bounds, not raw extrema.
                    group_metrics["state_training_min"] = state_min.tolist()
                    group_metrics["state_training_max"] = state_max.tolist()
                    raw_state_stats = self.processor.state_action_processor.statistics[
                        self.embodiment_tag.value
                    ]["state"].get(key)
                    if raw_state_stats is not None and {"min", "max"} <= set(
                        raw_state_stats
                    ):
                        observed_min = np.asarray(raw_state_stats["min"])
                        observed_max = np.asarray(raw_state_stats["max"])
                        observed_oor = (current < observed_min) | (current > observed_max)
                        group_metrics["state_observed_min"] = observed_min.tolist()
                        group_metrics["state_observed_max"] = observed_max.tolist()
                        group_metrics["state_observed_out_of_range_mask"] = (
                            observed_oor.tolist()
                        )
                        group_metrics["state_observed_out_of_range_fraction"] = float(
                            np.mean(observed_oor)
                        )
            action_diagnostics[key] = group_metrics
            offset += dimension

        info: dict[str, Any] = {
            "action_diagnostics": action_diagnostics,
            "noise_id": noise_id,
            "noise_mode": self.noise_mode,
        }
        if ttt_diagnostics is not None:
            info["ttt"] = ttt_diagnostics
        return casted_action, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def _reseed_policy_generators(self, seed: int) -> None:
        self._policy_cpu_generator.manual_seed(seed)
        self._policy_device_generator.manual_seed(seed)

    def _new_noise_id(self, query: int | str) -> str:
        nonce = int(
            torch.randint(
                0,
                torch.iinfo(torch.int64).max,
                (),
                generator=self._policy_cpu_generator,
                dtype=torch.int64,
            ).item()
        )
        identity = f"{self.policy_seed}:{query}:{nonce}"
        return "noise-" + hashlib.sha256(identity.encode("ascii")).hexdigest()[:20]

    def _begin_noise_episode(self) -> None:
        self._noise_episode += 1
        self._noise_query = 0
        self._episode_noise = None
        query: int | str = "common" if self.noise_mode == "episode_common" else 0
        self._current_noise_id = self._new_noise_id(query)

    def _claim_noise_id(self) -> str:
        noise_id = self._current_noise_id
        if self.noise_mode == "independent":
            self._noise_query += 1
            self._current_noise_id = self._new_noise_id(self._noise_query)
        return noise_id

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata)
        metadata.update(
            {
                "schema": "rmbench_gr00t_server_v1",
                "policy_kind": "robottt" if self.use_ttt else "vanilla",
                "policy_type": "robottt" if self.use_ttt else "vanilla",
                "noise_mode": self.noise_mode,
                "memory_enabled": self.use_ttt,
                "memory_impl_version": "robottt_v1" if self.use_ttt else None,
                "policy_seed": self.policy_seed,
                "checkpoint_load_report": self.checkpoint_load_report,
            }
        )
        required = (
            "policy_type",
            "adapter_version",
            "repo_commit",
            "checkpoint_sha256",
            "processor_sha256",
            "stats_sha256",
            "statistics_sha256",
            "modality_config_sha256",
            "state_arm_semantics",
            "state_gripper_semantics",
            "arm_action_semantics",
            "gripper_action_semantics",
            "ppu_id",
        )
        blockers = [f"missing:{key}" for key in required if metadata.get(key) is None]
        for source, path_key, digest_key in (
            ("checkpoint", "checkpoint_path", "checkpoint_sha256"),
            ("processor", "processor_path", "processor_sha256"),
            ("stats", "statistics_path", "stats_sha256"),
        ):
            current_digest = _sha256_file(metadata.get(path_key))
            metadata[f"{source}_current_sha256"] = current_digest
            if current_digest != metadata.get(digest_key):
                blockers.append(f"{source}_source_changed_after_load")
        if metadata.get("stats_source_content_sha256") != metadata.get(
            "stats_runtime_content_sha256"
        ):
            blockers.append("stats_runtime_content_mismatch")
        if metadata.get("dirty") is not False:
            blockers.append("repo_not_clean")
        if metadata.get("adapter_version") == "gr00t_policy_adapter_v4":
            expected_adapter_semantics = {
                "state_arm_semantics": "actual_qpos",
                "state_gripper_semantics": "physical",
                "arm_action_semantics": "absolute",
                "gripper_action_semantics": "absolute",
                "processor_decode_output_semantics": "absolute",
                "boundary_conversion": "none",
                "decoded_absolute_action_boundary": True,
            }
            blockers.extend(
                f"adapter_v4_semantic_mismatch:{key}"
                for key, expected in expected_adapter_semantics.items()
                if metadata.get(key) != expected
            )
        if metadata.get("policy_seed_explicit") is not True:
            blockers.append("policy_seed_not_explicit")
        if not self.checkpoint_load_report:
            blockers.append("checkpoint_load_report_empty")
        for report in self.checkpoint_load_report:
            scope = report.get("scope", "unknown")
            if not report.get("loaded"):
                blockers.append(f"checkpoint_loaded_keys_empty:{scope}")
            if report.get("unexpected"):
                blockers.append(f"checkpoint_unexpected_keys:{scope}")
            if scope != "compact_delta" and report.get("missing"):
                blockers.append(f"checkpoint_missing_keys:{scope}")
        if self.use_ttt and metadata.get("parent_checkpoint_sha256") is None:
            blockers.append("missing:parent_checkpoint_sha256")
        metadata["formal_blockers"] = sorted(blockers)
        metadata["formal_eligible"] = not blockers
        return metadata

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        options = dict(options or {})
        seed = options.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise TypeError(f"reset seed must be an integer, got {type(seed).__name__}")
            seed = int(seed)
            if seed < 0:
                raise ValueError(f"reset seed must be non-negative, got {seed}")
        rng_reseeded = bool(options.get("reseed_rng", seed is not None))
        if rng_reseeded:
            if seed is not None:
                self.policy_seed = seed
            self._reseed_policy_generators(self.policy_seed)

        clear_memory = bool(options.get("clear_memory", True))
        reset_ttt_state = getattr(self.model, "reset_ttt_state", None)
        memory_cleared = bool(clear_memory and self.use_ttt and reset_ttt_state is not None)
        if memory_cleared:
            reset_ttt_state()

        # A seed-bearing reset marks an episode boundary. Memory-only ablations
        # call reset() without options and must preserve episode-common noise.
        if bool(options.get("reset_noise", seed is not None)):
            self._begin_noise_episode()
        return {
            "memory_cleared": memory_cleared,
            "rng_reseeded": rng_reseeded,
            "policy_seed": self.policy_seed,
            "noise_mode": self.noise_mode,
            "noise_id": self._current_noise_id,
        }


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language = _sim_language_batch_to_sequence(observation[language_key])

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple, list, or numpy array. "
                f"Got {type(observation[language_key])}"
            )
            assert batched_language, f"Language key '{language_key}' must not be empty"

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    arr = _sim_language_batch_to_sequence(arr)
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
