import torch
def show_images_with_labels(images: torch.Tensor, labels:torch.Tensor, save_path=None, show:bool = True):
    """
    Display a grid of images with their corresponding labels.

    Args:
        images (torch.Tensor): Batch of images to display (shape: [N, C, H, W]).
        labels (torch.Tensor): Corresponding labels for the images (shape: [N]).
        save_path (str): Path to save the visualization. If None, visualization is not saved.
        show (bool): If True, display the plot. Otherwise, only save the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Normalize images to [0, 1] for display
    images = images.clone()  # Avoid modifying the original tensor
    images = (images - images.min()) / (images.max() - images.min())

    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i].permute(1, 2, 0).numpy()  # Convert to HWC format
            ax.imshow(img)
            ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved image grid to {save_path}")
    if show:
        plt.show()
    plt.close()
