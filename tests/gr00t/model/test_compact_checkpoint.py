import json

import pytest
from safetensors.torch import save_file
import torch

from gr00t.utils.compact_checkpoint import (
    resolve_compact_checkpoint_chain,
    validate_finite_state_dict,
)


def _compact(path, base, parent=None):
    path.mkdir()
    save_file({"weight": torch.ones(1)}, path / "trainable_model.safetensors")
    manifest = {"base_model_path": str(base)}
    if parent is not None:
        manifest["parent_checkpoint"] = str(parent)
    (path / "trainable_model.json").write_text(json.dumps(manifest))


def test_resolve_compact_checkpoint_chain_parent_first(tmp_path) -> None:
    base = tmp_path / "base"
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    _compact(parent, base)
    _compact(child, base, parent)

    resolved_base, chain = resolve_compact_checkpoint_chain(child)

    assert resolved_base == str(base.resolve())
    assert chain == [parent.resolve(), child.resolve()]


def test_resolve_compact_checkpoint_chain_rejects_cycle(tmp_path) -> None:
    base = tmp_path / "base"
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    _compact(parent, base, child)
    _compact(child, base, parent)

    with pytest.raises(ValueError, match="parent cycle"):
        resolve_compact_checkpoint_chain(child)


def test_validate_finite_state_dict_accepts_finite_tensors() -> None:
    validate_finite_state_dict({"weight": torch.tensor([0.0, 1.0])}, source="test")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validate_finite_state_dict_rejects_nonfinite_tensors(bad: float) -> None:
    with pytest.raises(FloatingPointError, match="weight.*non-finite"):
        validate_finite_state_dict({"weight": torch.tensor([0.0, bad])}, source="test")
