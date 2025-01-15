import torch
import pytest
from mlops_project.dataset import FruitsDataset

# Define expected dataset sizes
EXPECTED_TRAIN_SIZE = 70491
EXPECTED_TEST_SIZE = 23619
NUM_CLASSES = 141


@pytest.mark.parametrize("train", [True, False])
def test_fruits_dataset(train):
    # Initialize dataset
    dataset = FruitsDataset(data_folder="data/processed", train=train)

    # Check dataset size
    expected_size = EXPECTED_TRAIN_SIZE if train else EXPECTED_TEST_SIZE
    assert len(dataset) == expected_size, f"Dataset size mismatch for {'train' if train else 'test'}"

    # Test individual samples (limit to first 10 for efficiency)
    for i in range(min(10, len(dataset))):
        x, y = dataset[i]
        assert isinstance(x, torch.Tensor), "Image is not a tensor"
        assert x.shape == (3, 100, 100), "Image shape mismatch"
        assert isinstance(y, torch.Tensor), "Label is not a tensor"
        assert y.item() in range(NUM_CLASSES), "Label is out of range"

    # Check unique labels (using slicing for efficiency)
    unique_labels = torch.unique(torch.tensor([dataset[i][1].item() for i in range(min(100, len(dataset)))]))
    assert (unique_labels <= NUM_CLASSES - 1).all(), f"Unexpected labels in {'train' if train else 'test'} dataset"
