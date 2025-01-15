from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

if TYPE_CHECKING:
    import torchvision.transforms as transforms


class FruitsDataset(Dataset):
    """
    Custom Dataset for loading preprocessed tensors with optional transformations.

    Args:
        data_folder: Path to the data folder containing tensors.
        train: Whether to load training or test data.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """
    def __init__(
        self,
        data_folder: str = "data/processed",
        train: bool = True,
        img_transform: Optional[transforms.Compose] = None,
        target_transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""
        if self.train:
            images_file = os.path.join(self.data_folder, "train_images.pt")
            targets_file = os.path.join(self.data_folder, "train_targets.pt")
        else:
            images_file = os.path.join(self.data_folder, "test_images.pt")
            targets_file = os.path.join(self.data_folder, "test_targets.pt")

        self.images = torch.load(images_file)
        self.targets = torch.load(targets_file)

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Return DataLoader for the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        image, target = self.images[idx], self.targets[idx]
        if self.img_transform:
            image = self.img_transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.targets)

if __name__ == "__main__":
    # Initialize datasets
    train_dataset = FruitsDataset(data_folder="data/processed", train=True)
    test_dataset = FruitsDataset(data_folder="data/processed", train=False)

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Inspect the first sample
    x, y = train_dataset[0]
    print(f"Image shape: {x.shape}")
    print(f"Label: {y}")
