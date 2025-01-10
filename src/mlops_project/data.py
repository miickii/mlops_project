import os
from torchvision import datasets, transforms
import torch

def flatten_tensor(tensor):
    return tensor.view(-1)

def load_datasets(data_dir):
    # Define paths for training and testing datasets
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Test")
    
    # Define transformations: normalize and flatten the images
    transform = transforms.Compose([
        transforms.ToTensor(),       # Convert PIL images to tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
        transforms.Lambda(flatten_tensor)  # Flatten the tensor
    ])
    
    # Load datasets from subfolders
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    test_set = datasets.ImageFolder(test_dir, transform=transform)
    
    return train_set, test_set

def save_datasets(train_set, test_set, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)  # Ensure the directory exists
    train_path = os.path.join(processed_dir, "train_set.pt")
    test_path = os.path.join(processed_dir, "test_set.pt")
    
    train_set.transform = None
    test_set.transform = None


    # Save the datasets
    torch.save(train_set, train_path)
    torch.save(test_set, test_path)

    train_set.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(flatten_tensor)
    ])
    test_set.transform = train_set.transform

    print(f"Train set saved to {train_path}")
    print(f"Test set saved to {test_path}")

def inspect_pt_file(file_path):
    # Load the .pt file
    data = torch.load(file_path)
    
    # Check the type of the object
    print(f"Type of loaded object: {type(data)}")
    
    if isinstance(data, torch.utils.data.Dataset):
        # For ImageFolder datasets
        print(f"Number of samples: {len(data)}")
        print(f"Classes: {data.classes}")
        print(f"Class-to-Index Mapping: {data.class_to_idx}")
        
        # Inspect a few samples
        for i in range(500, 505):  # Show up to 3 samples
            image, label = data[i]
            print(f"Sample {i}:")
            print(f"  Flattened Image Tensor: {image}")
            print(f"  Label: {label}")
    else:
        print("The file does not contain a Dataset object.")

if __name__ == "__main__":
    train_file_path = "data/processed/train_set.pt"
    test_file_path = "data/processed/test_set.pt"
    
    print("Inspecting Train Set:")
    inspect_pt_file(train_file_path)
    
    print("\nInspecting Test Set:")
    inspect_pt_file(test_file_path)

# if __name__ == "__main__":
#     data_dir = "data/raw/fruits-360_dataset_100x100/fruits-360"
#     processed_dir = "data/processed"
#     train_set, test_set = load_datasets(data_dir)
    
#     save_datasets(train_set, test_set, processed_dir)