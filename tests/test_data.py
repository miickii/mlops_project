import torch
from mlops_project.dataset import get_dataset

def test_fruits_dataset():
    train_dataset, test_dataset = get_dataset()

    # Check dataset sizes
    assert len(train_dataset) == 70491, "Train dataset size mismatch"
    assert len(test_dataset) == 23619, "Test dataset size mismatch"

    # Test individual samples
    for dataset in [train_dataset, test_dataset]:
        for x, y in dataset:  # Test the first 5 samples
            assert isinstance(x, torch.Tensor), "Image is not a tensor"
            assert x.shape == (3, 100, 100), "Image shape mismatch"
            assert isinstance(y, torch.Tensor), "Label is not a tensor"
            assert y.item() in range(141), "Label is out of range"

    # Check unique labels
    train_labels = torch.tensor([train_dataset[i][1].item() for i in range(len(train_dataset))])
    test_labels = torch.tensor([test_dataset[i][1].item() for i in range(len(test_dataset))])

    train_targets = torch.unique(train_labels)
    test_targets = torch.unique(test_labels)

    assert (train_targets == torch.arange(0, 141)).all(), "Train dataset does not contain all classes"
    assert (test_targets == torch.arange(0, 141)).all(), "Test dataset does not contain all classes"
