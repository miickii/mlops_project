import timm
import torch
from torch import nn

def ProjectModel(num_classes, pretrained=True):
    """
    Create and configure the model.
    
    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained model.

    Returns:
        torch.nn.Module: Configured model.
    """
    # Load pretrained ResNet
    model = timm.create_model("resnet18", pretrained=pretrained)

    # Modify the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

if __name__ == "__main__":
    # Pass the required num_classes argument
    num_classes = 141  # Replace with the actual number of classes
    model = ProjectModel(num_classes)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Dummy input with 3 channels (e.g., RGB image)
    dummy_input = torch.randn(1, 3, 100, 100)  # ResNet expects input size of (3, H, W)
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():  # No gradients needed for inference
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
