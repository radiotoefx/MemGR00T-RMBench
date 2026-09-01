"""Utilities for resolving chained trainable-only GR00T checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


TRAINABLE_WEIGHTS_NAME = "trainable_model.safetensors"
TRAINABLE_MANIFEST_NAME = "trainable_model.json"


def validate_finite_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, source: str
) -> None:
    """Reject model tensors containing NaN or Inf before they can propagate."""

    invalid: list[str] = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            continue
        finite = torch.isfinite(tensor)
        if not bool(finite.all()):
            invalid.append(f"{name} ({int((~finite).sum())}/{tensor.numel()} non-finite)")
    if invalid:
        preview = ", ".join(invalid[:8])
        suffix = "" if len(invalid) <= 8 else f", ... and {len(invalid) - 8} more"
        raise FloatingPointError(
            f"Refusing non-finite checkpoint/state from {source}: {preview}{suffix}"
        )


def resolve_compact_checkpoint_chain(
    checkpoint: str | Path,
) -> tuple[str, list[Path]]:
    """Return the full base model and parent-first compact delta chain."""

    current = Path(checkpoint).resolve()
    child_first: list[Path] = []
    visited: set[Path] = set()
    base_model_path: str | None = None

    while (current / TRAINABLE_WEIGHTS_NAME).is_file():
        if current in visited:
            raise ValueError(f"Compact checkpoint parent cycle at {current}")
        visited.add(current)
        manifest_path = current / TRAINABLE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Compact checkpoint is missing {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        manifest_base = manifest.get("base_model_path")
        if manifest_base:
            resolved_base = str(Path(manifest_base).resolve())
            if base_model_path is None:
                base_model_path = resolved_base
            elif base_model_path != resolved_base:
                raise ValueError(
                    "Compact checkpoint chain has inconsistent base models: "
                    f"{base_model_path} != {resolved_base}"
                )
        child_first.append(current)
        parent = manifest.get("parent_checkpoint")
        if not parent:
            break
        current = Path(parent).resolve()

    if base_model_path is None:
        raise ValueError(f"Compact checkpoint {checkpoint} does not identify a base model")
    return base_model_path, list(reversed(child_first))
