import timm
import torch
from torch import nn
from torchinfo import summary

def ProjectModel(num_classes, pretrained=True):
    """
    Create and configure the model.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use a pretrained model.

    Returns:
        torch.nn.Module: Configured model.
    """
    if num_classes <= 0:
        raise ValueError("Number of classes must be greater than zero.")

    # Load pretrained ResNet
    model = timm.create_model("resnet18", pretrained=pretrained)

    # Modify the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

if __name__ == "__main__":
    num_classes = 141
    model = ProjectModel(num_classes)
    summary(model, input_size=(1, 3, 100, 100))

    dummy_input = torch.randn(1, 3, 100, 100)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
