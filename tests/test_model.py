import torch
from mlops_project.model import ProjectModel

def test_model():
    model = ProjectModel(141)
    x = torch.randn(1, 3, 100, 100)
    y = model(x)
    assert y.shape == (1, 141)