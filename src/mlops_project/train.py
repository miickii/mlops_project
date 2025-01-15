from mlops_project.dataset import FruitsDataset
from mlops_project.model import ProjectModel
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import argparse

def train_model(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epochs:int = 4, model_name:str ="model.pth"):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    "loss": f"{train_loss / total:.4f}",
                    "accuracy": f"{100. * correct / total:.2f}%"
                })

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

    # Save the model
    checkpoint_path = os.path.join("models/", model_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # Ensure the models directory exists
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss / len(train_loader),
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

# Example script: python train.py --epochs 20 --batch_size 128
def main():
    parser = argparse.ArgumentParser(description="Train a fruit classification model.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer (default: 1e-4)")
    parser.add_argument("--model_name", type=str, default="model.pth", help="Filename for saving the trained model (default: model.pth)")

    args = parser.parse_args()

    # Initialize dataset and DataLoader
    train_dataset = FruitsDataset(data_folder="data/processed", train=True)
    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size)

    # Initialize model
    num_classes = 141
    model = ProjectModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, device, epochs=args.epochs, model_name=args.model_name)


<<<<<<< HEAD
def train():
    print("Training model...")
    # Training loop
    epochs = 4
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Use tqdm to track progress
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{train_loss / (total // labels.size(0)):.4f}",
                    "accuracy": f"{100. * correct / total:.2f}%"
                })

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

    # Validate the model after training
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with tqdm(test_loader, desc="Validation", unit="batch") as pbar:
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{test_loss / (total // labels.size(0)):.4f}",
                    "accuracy": f"{100. * correct / total:.2f}%"
                })

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100.*correct/total:.2f}%")

def main():
    typer.run(train)
=======
if __name__ == "__main__":
    main()
>>>>>>> 816d2397c79530132338195073e7ddd68d3a127d
