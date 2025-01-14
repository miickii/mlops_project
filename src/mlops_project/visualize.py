import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dataset import CustomDataset  # Import your custom dataset
from model import ProjectModel  # Import your model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize(model_checkpoint: str, test_data_path: str, test_labels_path: str, figure_name: str = "embeddings.png") -> None:
    """
    Visualize model embeddings using t-SNE.

    Args:
        model_checkpoint (str): Path to the trained model checkpoint.
        test_data_path (str): Path to the test images tensor (.pt file).
        test_labels_path (str): Path to the test labels tensor (.pt file).
        figure_name (str): Name of the output visualization file.
    """
    print("Loading model...")
    num_classes = 141  # Update with your actual number of classes
    model = ProjectModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(DEVICE)
    model.eval()

    # Replace the fully connected layer with an identity layer to extract embeddings
    model.fc = torch.nn.Identity()

    print("Loading test data...")
    test_images = torch.load(test_data_path)
    test_targets = torch.load(test_labels_path)
    test_dataset = CustomDataset(test_images, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

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

if __name__ == "__main__":
    typer.run(visualize)
    # Eksempel på kommando: python visualize.py "path/to/model_checkpoint.pth" "data/processed/test_images.pt" "data/processed/test_targets.pt" --figure-name="embedding_visualization.png"
