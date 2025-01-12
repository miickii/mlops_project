import os
import torch
from torchvision import datasets, transforms
from PIL import Image

# Function to process and save data
def save_dataset_as_pt(data_dir, output_image_file, output_target_file):
    image_tensors = []
    targets = []
    class_to_idx = {}

    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Map class name to index
        if not class_to_idx:
            class_to_idx = {name: idx for idx, name in enumerate(sorted(os.listdir(data_dir)))}
        
        label = class_to_idx[class_name]
        
        # Process images
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path).convert("RGB")  # Ensure RGB format
            image_tensor = transform(image)
            image_tensors.append(image_tensor)
            targets.append(label)

    # Save as .pt files
    image_tensors = torch.stack(image_tensors)  # Stack all tensors into a single tensor
    targets = torch.tensor(targets)  # Convert targets to a single tensor

    torch.save(image_tensors, os.path.join(output_dir, output_image_file))
    torch.save(targets, os.path.join(output_dir, output_target_file))
    print(f"Saved {len(targets)} samples to {output_image_file} and {output_target_file}.")

if __name__ == "__main__":
    # Define paths
    train_dir = "data/raw/fruits-360_dataset_100x100/fruits-360/Training"
    test_dir = "data/raw/fruits-360_dataset_100x100/fruits-360/Test"
    
    # Process train and test datasets
    save_dataset_as_pt(train_dir, "train_images.pt", "train_targets.pt")
    save_dataset_as_pt(test_dir, "test_images.pt", "test_targets.pt")