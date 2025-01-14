import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlops_project.dataset import get_dataloaders
from mlops_project.model import ProjectModel
from mlops_project.train_lightning import FruitClassifierModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Add Typer command
app = typer.Typer()

@app.command()
def visualize(model_checkpoint: str, figure_name: str = "embeddings.png", batch_size: int = 32) -> None:
    """
    Visualize model embeddings using t-SNE.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint.
        figure_name (str): Name of the output visualization file.
        batch_size (int): Batch size for DataLoader.
    """
    print("Starting visualization...")

    _, test_loader = get_dataloaders(batch_size=batch_size, transform=None)

    # Check the file extension and load the model accordingly
    if model_checkpoint.endswith(".pth"):
        print("Loading model from .pth checkpoint...")
        num_classes = 141
        model = ProjectModel(num_classes=num_classes)

        # Load the saved model state_dict from the checkpoint
        checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

        # Replace the final fully connected layer with an identity layer for embeddings
        model.fc = torch.nn.Identity()

    elif model_checkpoint.endswith(".pth.ckpt"):
        print("Loading model from .pth.ckpt Lightning checkpoint...")
        num_classes = 141
        model = FruitClassifierModule.load_from_checkpoint(
            model_checkpoint,
            model=ProjectModel(num_classes=num_classes),
            num_classes=num_classes,
            lr=1e-4
        )
        model.model.fc = torch.nn.Identity()  # Replace the final layer with an identity layer for embeddings
        model.to(DEVICE)

    else:
        raise ValueError("Unsupported checkpoint file extension. Please use .pth or .pth.ckpt files.")

    model.eval()

    print("Extracting embeddings...")
    embeddings, targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            embeddings.append(outputs.cpu())
            targets.append(labels)

    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    # Reduce dimensionality if needed
    if embeddings.shape[1] > 500:
        print("Reducing dimensionality using PCA...")
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    print("Applying t-SNE for 2D visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Generating visualization...")
    plt.figure(figsize=(10, 10))
    unique_labels = set(targets)
    for label in unique_labels:
        mask = targets == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=str(label), alpha=0.7)

    plt.legend(title="Classes")
    plt.title("t-SNE Visualization of Test Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(f"reports/figures/{figure_name}")
    print(f"Visualization saved as {figure_name}")

# visualize --figure-name embeddings_pth.png --batch-size 32 models/fruits_model.pth
if __name__ == "__main__":
    app()
