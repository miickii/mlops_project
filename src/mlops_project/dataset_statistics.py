import matplotlib.pyplot as plt
import torch
import typer
from mlops_project.dataset import FruitsDataset
from mlops_project.utils import show_images_with_labels

app = typer.Typer()

@app.command()
def dataset_statistics(data_folder: str = "data/processed") -> None:
    """Compute dataset statistics and generate visualizations."""
    # Load datasets
    train_dataset = FruitsDataset(data_folder=data_folder, train=True)
    test_dataset = FruitsDataset(data_folder=data_folder, train=False)

    # Print dataset information
    print("Train dataset:")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\nTest dataset:")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # Visualize images from different classes
    print("Saving class-wise sample images...")
    num_classes = 141
    images_per_class = []

    for class_id in range(num_classes):
        for i in range(len(train_dataset)):
            image, label = train_dataset[i]
            if label.item() == class_id:
                images_per_class.append((image, label))
                break  # Take only one sample per class
        if len(images_per_class) == num_classes:
            break

    # Check if all classes are covered
    if len(images_per_class) < num_classes:
        print(f"Warning: Only found samples for {len(images_per_class)} out of {num_classes} classes.")

    # Extract images and labels for visualization
    class_images, class_labels = zip(*images_per_class)
    class_images = torch.stack(class_images)
    class_labels = torch.tensor([label.item() for label in class_labels])

    # Save visualization
    show_images_with_labels(
        images=class_images,
        labels=class_labels,
        save_path="reports/figures/class_samples.png",
        show=False
    )

    # Compute label distributions
    print("Computing label distributions...")
    train_labels = torch.tensor([train_dataset[i][1].item() for i in range(len(train_dataset))])
    test_labels = torch.tensor([test_dataset[i][1].item() for i in range(len(test_dataset))])

    train_label_distribution = torch.bincount(train_labels, minlength=num_classes)
    test_label_distribution = torch.bincount(test_labels, minlength=num_classes)

    # Plot train label distribution
    print("Saving train label distribution...")
    plt.bar(torch.arange(num_classes).numpy(), train_label_distribution.numpy())
    plt.title("Train Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/train_label_distribution.png")
    plt.close()

    # Plot test label distribution
    print("Saving test label distribution...")
    plt.bar(torch.arange(num_classes).numpy(), test_label_distribution.numpy())
    plt.title("Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("reports/figures/test_label_distribution.png")
    plt.close()

    print("Dataset statistics and visualizations saved in 'reports/figures/'.")

if __name__ == "__main__":
    app()
