import torch
from pytorch_lightning import Trainer
from mlops_project.dataset import FruitsDataset
from mlops_project.train_lightning import FruitClassifierModel

def test_training_pipeline(batch_size=32, lr=1e-3, epochs=1):
    """
    Test the training pipeline for initialization and integration.

    Args:
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
    """
    # Initialize dataset and DataLoader
    train_dataset = FruitsDataset(data_folder="data/processed", train=True)
    assert len(train_dataset) > 0, "Train dataset is empty"
    train_loader = train_dataset.get_dataloader(batch_size=32)
    assert train_loader is not None, "Train DataLoader is None"
    assert len(train_loader) > 0, "Train DataLoader has no batches"

    # Use a smaller dataset subset for faster testing
    small_subset = torch.utils.data.Subset(train_dataset, range(100))  # First 100 samples
    sampled_loader = torch.utils.data.DataLoader(small_subset, batch_size=32, shuffle=False)
    assert len(sampled_loader) > 0, "Sampled DataLoader has no batches"

    # Calculate dynamic limit for train batches
    total_batches = len(sampled_loader)
    dynamic_limit = min(1.0, max(1 / total_batches, 1.0 / total_batches)) if total_batches > 0 else 1.0

    # Initialize model and module
    model = FruitClassifierModel(1e-3)

    # Initialize trainer with dynamic batch limit
    trainer = Trainer(max_epochs=1, limit_train_batches=dynamic_limit, devices="cpu")
    assert isinstance(trainer, Trainer), "Trainer is not an instance of pytorch_lightning.Trainer"

    # Run a sanity check to ensure the training loop initializes correctly
    trainer.fit(model, sampled_loader)
