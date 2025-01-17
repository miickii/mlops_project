import os
import torch
from torchvision import transforms
from PIL import Image
import typer

def preprocess_and_save(data_dir: str, output_dir: str, output_image_file: str, output_target_file: str) -> None:
    """Preprocess dataset (ToTensor and Normalize) and save as .pt files."""
    image_tensors = []
    targets = []
    class_to_idx = {}

    # Define transformation (ToTensor + Normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    os.makedirs(output_dir, exist_ok=True)

    # Process the dataset directory
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Map class names to indices
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

    # Convert to tensors and save
    image_tensors = torch.stack(image_tensors)  # Combine all images into a single tensor
    targets = torch.tensor(targets, dtype=torch.long)  # Convert targets to a tensor

    torch.save(image_tensors, os.path.join(output_dir, output_image_file))
    torch.save(targets, os.path.join(output_dir, output_target_file))
    print(f"Saved {len(targets)} samples to {output_image_file} and {output_target_file}.")

def preprocess_data() -> None:
    """Preprocess and save train and test datasets."""
    preprocess_and_save(
        data_dir=os.path.join("data/raw/fruits-dataset-100x100", "Training"),
        output_dir="data/processed",
        output_image_file="train_images.pt",
        output_target_file="train_targets.pt"
    )
    preprocess_and_save(
        data_dir=os.path.join("data/raw/fruits-dataset-100x100", "Test"),
        output_dir="data/processed",
        output_image_file="test_images.pt",
        output_target_file="test_targets.pt"
    )

def main():
    typer.run(preprocess_data)
    # Eksempel på at process Training data: python src/mlops_project/data.py
