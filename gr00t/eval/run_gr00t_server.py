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

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.rmbench_adapter import RMBenchDecodedActionPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import torch
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555
RMBENCH_SERVER_VERSION = "rmbench_gr00t_server_v1"
RMBENCH_ADAPTER_VERSION = "gr00t_policy_adapter_v4"


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_file(root: Path, candidates: tuple[str, ...]) -> Path | None:
    return next((root / name for name in candidates if (root / name).is_file()), None)


def _repo_identity(repo: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repo,
                text=True,
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _build_server_metadata(config: "ServerConfig") -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]
    commit, dirty = _repo_identity(repo)
    try:
        process_start_time = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        process_start_time = datetime.now(timezone.utc).isoformat()
    model_dir = Path(config.model_path).resolve() if config.model_path else None
    processor_dir = (
        Path(config.processor_path).resolve()
        if config.processor_path
        else model_dir
    )
    checkpoint_file = None if model_dir is None else _first_file(
        model_dir,
        ("trainable_model.safetensors", "model.safetensors"),
    )
    processor_file = None if processor_dir is None else _first_file(
        processor_dir,
        ("processor_config.json", "processor/processor_config.json"),
    )
    statistics_file = None if processor_dir is None else _first_file(
        processor_dir,
        ("statistics.json", "processor/statistics.json"),
    )
    modality_file = (
        Path(config.modality_config_path).resolve()
        if config.modality_config_path
        else None
    )
    return {
        "server_version": RMBENCH_SERVER_VERSION,
        "adapter_version": RMBENCH_ADAPTER_VERSION,
        "process_start_time": process_start_time,
        "server_host": config.host,
        "server_port": config.port,
        "policy_type": "robottt" if config.use_ttt else "vanilla",
        "repo_commit": commit,
        "dirty": dirty,
        "checkpoint_path": None if checkpoint_file is None else str(checkpoint_file),
        "checkpoint_sha256": _sha256_file(checkpoint_file),
        "processor_path": None if processor_file is None else str(processor_file),
        "processor_sha256": _sha256_file(processor_file),
        "statistics_path": None if statistics_file is None else str(statistics_file),
        "statistics_sha256": _sha256_file(statistics_file),
        # Contract alias: stats_sha256 is the digest of the exact statistics
        # file loaded by AutoProcessor.from_pretrained below.
        "stats_sha256": _sha256_file(statistics_file),
        "modality_config_path": None if modality_file is None else str(modality_file),
        "modality_config_sha256": _sha256_file(modality_file),
        "parent_checkpoint_sha256": config.parent_checkpoint_sha256,
        "state_arm_semantics": config.state_arm_semantics,
        "state_gripper_semantics": config.state_gripper_semantics,
        "arm_action_semantics": config.arm_action_semantics,
        "gripper_action_semantics": config.gripper_action_semantics,
        "processor_decode_output_semantics": "absolute",
        "boundary_conversion": "none",
        "decoded_absolute_action_boundary": config.decoded_absolute_action_boundary,
        "memory_impl_version": "robottt_v1" if config.use_ttt else None,
        "ppu_id": config.ppu_id,
        "policy_seed_explicit": config.policy_seed is not None,
    }


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    base_model_path: str | None = None
    """Optional base model override for compact trainable-only checkpoints."""

    processor_path: str | None = None
    """Optional processor/stats checkpoint, useful for a schema-matched raw-base control."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    use_ttt: bool = False
    """Enable RoboTTT fast-weight memory in the action DiT."""

    ttt_num_layers: int = 2
    """Number of final DiT layers augmented with TTT."""

    ttt_dim: int = 256
    """Fast-model input/output width."""

    ttt_hidden_dim: int = 1024
    """Hidden width of the two-layer GeLU fast model."""

    ttt_num_register_tokens: int = 0
    """Learned register tokens per timestep (16 matches the paper)."""

    ttt_gate_init: float = 0.001
    """Initial tanh gate value; use 0 for exact Vanilla equivalence tests."""

    policy_seed: int | None = None
    """Private policy RNG seed. Formal launches should set this explicitly."""

    noise_mode: str = "independent"
    """Initial action noise mode: independent or episode_common."""

    ppu_id: int | None = None
    """Physical PPU id recorded in metadata; do not infer this from remapped cuda:0."""

    parent_checkpoint_sha256: str | None = None
    """Immutable parent checkpoint identity for a RoboTTT candidate."""

    state_arm_semantics: str = "actual_qpos"
    state_gripper_semantics: str = "physical"
    arm_action_semantics: str = "relative"
    gripper_action_semantics: str = "absolute"
    decoded_absolute_action_boundary: bool = False
    """Validate and expose processor-decoded absolute actions without conversion."""

    launch_manifest_path: str | None = None
    """Optional JSON path for the auditable process and model launch manifest."""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        metadata = _build_server_metadata(config)
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
            base_model_path=config.base_model_path,
            processor_path=config.processor_path,
            policy_seed=config.policy_seed,
            noise_mode=config.noise_mode,
            metadata=metadata,
            model_config_overrides=(
                {
                    "use_ttt": True,
                    "ttt_num_layers": config.ttt_num_layers,
                    "ttt_dim": config.ttt_dim,
                    "ttt_hidden_dim": config.ttt_hidden_dim,
                    "ttt_num_register_tokens": config.ttt_num_register_tokens,
                    "ttt_gate_init": config.ttt_gate_init,
                }
                if config.use_ttt
                else None
            ),
        )
        if config.decoded_absolute_action_boundary:
            expected_semantics = {
                "state_arm_semantics": (config.state_arm_semantics, "actual_qpos"),
                "arm_action_semantics": (config.arm_action_semantics, "absolute"),
                "gripper_action_semantics": (
                    config.gripper_action_semantics,
                    "absolute",
                ),
            }
            mismatches = [
                f"{name}={actual!r} (expected {expected!r})"
                for name, (actual, expected) in expected_semantics.items()
                if actual != expected
            ]
            if mismatches:
                raise ValueError(
                    "RMBench decoded-absolute action boundary requires "
                    + ", ".join(mismatches)
                )
            policy = RMBenchDecodedActionPolicyWrapper(policy, strict=config.strict)
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    if config.launch_manifest_path is not None:
        manifest_path = Path(config.launch_manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        process_start = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
        ).strip()
        manifest = {
            "schema": "rmbench_gr00t_launch_manifest_v1",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "process_start_time": process_start,
            "argv": [str(Path(sys.executable).resolve()), *sys.argv],
            "cwd": str(Path.cwd().resolve()),
            "host": config.host,
            "port": config.port,
            "device": config.device,
            "ppu_id": config.ppu_id,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            "metadata": getattr(policy, "get_metadata", lambda: {})(),
        }
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Wrote launch manifest: {manifest_path}", flush=True)

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
