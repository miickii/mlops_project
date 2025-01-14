from mlops_project.dataset import get_dataloaders
from mlops_project.model import ProjectModel
import torch
from torch import nn, optim
from tqdm import tqdm
import os
import argparse

def train_model(train_loader, model, criterion, optimizer, device, epochs=4, model_name="model.pth"):
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
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss / len(train_loader),
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

# eksempel script: train --epochs 20 --batch_size 128
def main():
    parser = argparse.ArgumentParser(description="Train a fruit classification model.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs (default: 4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer (default: 1e-4)")
    parser.add_argument("--model_name", type=str, default="model.pth", help="Filename for saving the trained model (default: model.pth)")

    args = parser.parse_args()

    train_loader, _ = get_dataloaders(batch_size=args.batch_size, transform=None)

    num_classes = 141
    model = ProjectModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(train_loader, model, criterion, optimizer, device, epochs=args.epochs, model_name=args.model_name)


if __name__ == "__main__":
    train_image_file = "data/processed/train_images.pt"
    train_target_file = "data/processed/train_targets.pt"
    test_image_file = "data/processed/test_images.pt"
    test_target_file = "data/processed/test_targets.pt"

    # No normalization transform since preprocessing already handled it
    train_loader, test_loader = get_dataloaders(
        train_image_file, train_target_file, test_image_file, test_target_file, batch_size=32, transform=None
    )

    num_classes = 141
    model = ProjectModel(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(train_loader, model, criterion, optimizer, device, epochs=4)
