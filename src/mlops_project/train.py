from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from torch import nn, optim
from tqdm import tqdm

# Define Custom Dataset for .pt Files
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
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

# Load the .pt files
train_images = torch.load("data/processed/train_images.pt")  # Training images tensor
train_targets = torch.load("data/processed/train_targets.pt")  # Training labels tensor

test_images = torch.load("data/processed/test_images.pt")  # Test images tensor
test_targets = torch.load("data/processed/test_targets.pt")  # Test labels tensor

# Define transformations
train_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize only
])

test_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize only
])

# Create datasets
train_dataset = CustomDataset(train_images, train_targets, transform=train_transform)
test_dataset = CustomDataset(test_images, test_targets, transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet
model = timm.create_model("resnet18", pretrained=True)

# Modify the final layer to match the number of classes in your dataset
num_classes = 141  # Replace with your actual number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
