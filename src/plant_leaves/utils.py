from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def show_image_and_target(images, targets, show=True, grid_size=(5, 5)):
    """
    Display a grid of images with their corresponding targets.

    Args:
        images (list or np.ndarray): A list or array of images to display.
        targets (list or np.ndarray): A list or array of targets corresponding to the images.
        show (bool): Whether to display the plot interactively. Defaults to True.
        grid_size (tuple): The dimensions of the grid (rows, cols). Defaults to (5, 5).
    """
    num_images = len(images)
    rows, cols = grid_size

    if num_images < rows * cols:
        print(f"Warning: Not enough images to fill the grid ({rows}x{cols}).")
        rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            ax.imshow(images[i].to(torch.uint8), cmap="gray")
            ax.set_title(f"Target: {targets[i]}")
        ax.axis("off")

    if show:
        plt.tight_layout()
        plt.show()
