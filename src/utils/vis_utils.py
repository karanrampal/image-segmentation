"""Visualization utils"""

import math
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def vis_data(img: np.ndarray,
             target: Dict[str, Any],
             categories: Dict[int, str],
             num_cols: int = 3,
             figsize: Tuple[int, int] = (20, 30)) -> None:
    """Visualze the data
    Args:
        img: Input image
        target: Dictionary of ground truth labels or predictions
        num_cols: Number of columns in the visuazilation grid
        figsize: Figure size
    """
    boxes = target["boxes"].detach().numpy()
    labels = target["labels"].detach().numpy()
    masks = target["masks"].detach().numpy()
    if "scores" in target:
        scores = target["scores"].detach().numpy()
    num = len(boxes) + 1
    num_rows = math.ceil(num / num_cols)
    _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    ax = ax.flatten()

    ax[0].imshow(img)
    ax[0].set_title(f"Input image")
    ax[0].axis("off")
    for i in range(1, num):
        j = i - 1

        ax[i].imshow(img)
        ax[i].imshow(masks[j].squeeze(), alpha=0.7)

        label = labels[j] - 1
        title = f"Label: {label} ({categories[label]})"
        if "scores" in target:
            title += f" Score: {scores[j]:.3f}"
        ax[i].set_title(title)

        x1, y1, x2, y2 = boxes[j]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax[i].add_patch(rect)
        ax[i].axis("off")

    for axi in ax[i:]:
        axi.axis("off")