import torch
from torch.utils.data import Dataset, DataLoader

class FruitsDataset(Dataset):
    """
    Custom Dataset for loading preprocessed tensors with optional transformations.
    """
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (Tensor): Tensor containing image data.
            labels (Tensor): Tensor containing labels.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataset(transform=None):
    train_image_file = "data/processed/train_images.pt"
    train_target_file = "data/processed/train_targets.pt"
    test_image_file = "data/processed/test_images.pt"
    test_target_file = "data/processed/test_targets.pt"

     # Load tensors
    train_images = torch.load(train_image_file)
    train_targets = torch.load(train_target_file)
    test_images = torch.load(test_image_file)
    test_targets = torch.load(test_target_file)

    # Create datasets
    train_dataset = FruitsDataset(train_images, train_targets, transform=transform)
    test_dataset = FruitsDataset(test_images, test_targets, transform=transform)

    return train_dataset, test_dataset



def get_dataloaders(batch_size=32, transform=None):
    """
    Load preprocessed data and return DataLoaders for training and testing.

    Args:
        train_image_file (str): Path to training images tensor (.pt file).
        train_target_file (str): Path to training labels tensor (.pt file).
        test_image_file (str): Path to testing images tensor (.pt file).
        test_target_file (str): Path to testing labels tensor (.pt file).
        batch_size (int): Batch size for DataLoader.
        transform (callable, optional): Transform to apply to the images.

    Returns:
        tuple: Train and test DataLoaders.
    """
    
    train_dataset, test_dataset = get_dataset(transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
