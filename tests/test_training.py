import pytest
import torch
from pytorch_lightning import Trainer
from mlops_project.dataset import get_dataloaders
from mlops_project.model import ProjectModel
from mlops_project.train_lightning import FruitClassifierModule

@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("lr", [1e-3, 1e-4])
@pytest.mark.parametrize("epochs", [1])
def test_training_pipeline(batch_size, lr, epochs):
    """
    Test the training pipeline for initialization and integration.

    Args:
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
    """
    # Get the dataloaders
    train_loader, _ = get_dataloaders(batch_size=batch_size, transform=None)
    assert train_loader is not None, "Train DataLoader is None"
    assert len(train_loader) > 0, "Train DataLoader has no batches"

    # Use a smaller dataset subset for faster testing
    small_subset = torch.utils.data.Subset(train_loader.dataset, range(100))  # First 100 samples
    sampled_loader = torch.utils.data.DataLoader(small_subset, batch_size=batch_size, shuffle=False)

    # Calculate dynamic limit for train batches
    total_batches = len(sampled_loader)
    dynamic_limit = max(1 / total_batches, 1.0 / total_batches) if total_batches > 0 else 1.0

    # Initialize model and module
    num_classes = 141
    base_model = ProjectModel(num_classes=num_classes)
    model = FruitClassifierModule(base_model, num_classes, lr)

    # Initialize trainer with dynamic batch limit
    trainer = Trainer(max_epochs=epochs, limit_train_batches=dynamic_limit)
    assert isinstance(trainer, Trainer), "Trainer is not an instance of pytorch_lightning.Trainer"

    # Run a sanity check to ensure the training loop initializes correctly
    trainer.fit(model, sampled_loader)
