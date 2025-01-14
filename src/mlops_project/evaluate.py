from mlops_project.dataset import FruitsDataset
from mlops_project.model import ProjectModel
from mlops_project.train_lightning import FruitClassifierModule
import torch
import typer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
app = typer.Typer()

@app.command()
def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    print("Starting evaluation...")

    # Initialize test dataset and DataLoader
    test_dataset = FruitsDataset(data_folder="data/processed", train=False)
    test_loader = test_dataset.get_dataloader(batch_size=batch_size)

    # Load the model based on the checkpoint extension
    if model_checkpoint.endswith(".pth"):
        print("Loading model from .pth checkpoint...")
        num_classes = 141
        model = ProjectModel(num_classes=num_classes)

        # Load the saved model state_dict from the checkpoint
        checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

    elif model_checkpoint.endswith(".ckpt"):
        print("Loading model from .ckpt Lightning checkpoint...")
        num_classes = 141
        model = FruitClassifierModule.load_from_checkpoint(
            model_checkpoint,
            model=ProjectModel(num_classes=num_classes),
            num_classes=num_classes,
            lr=1e-4
        )
        model.to(DEVICE)

    else:
        raise ValueError("Unsupported checkpoint file extension. Please use .pth or .pth.ckpt files.")

    model.eval()
    test_loss, correct, total = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    app()
