import pytest
import torch

from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


@pytest.mark.parametrize(
    "module",
    [
        CategorySpecificMLP(4, 3, 5, 2),
        MultiEmbodimentActionEncoder(action_dim=3, hidden_size=6, num_embodiments=4),
    ],
)
def test_copy_category_warm_starts_only_target(module):
    before = {name: value.detach().clone() for name, value in module.named_parameters()}

    module.copy_category_(source_id=2, target_id=1)

    for name, value in module.named_parameters():
        assert torch.equal(value[1], value[2]), name
        assert torch.equal(value[0], before[name][0]), name
        assert torch.equal(value[2], before[name][2]), name
        assert torch.equal(value[3], before[name][3]), name


def test_copy_category_rejects_invalid_id():
    module = CategorySpecificMLP(2, 3, 5, 2)
    with pytest.raises(ValueError, match="outside"):
        module.copy_category_(source_id=2, target_id=0)
