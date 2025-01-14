import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import typer
from model import ProjectModel  # Import your model definition
from dataset import CustomDataset  # Import your dataset class

# Set device for computation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_checkpoint: str, test_data_path: str, test_labels_path: str) -> None:
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        model_checkpoint (str): Path to the trained model checkpoint.
        test_data_path (str): Path to the test images tensor (.pt file).
        test_labels_path (str): Path to the test labels tensor (.pt file).
    """
    print("Starting evaluation...")

    # Load the test dataset
    print("Loading test data...")
    test_images = torch.load(test_data_path)
    test_targets = torch.load(test_labels_path)

    # Define transformations
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    test_dataset = CustomDataset(test_images, test_targets, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model
    print("Loading model...")
    num_classes = 141  # Update with your actual number of classes
    model = ProjectModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(DEVICE)

    # Evaluate the model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    typer.run(evaluate)
