"""Visualization utils"""

import math
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def vis_data(img: np.ndarray, target: Dict[str, Any], cats: Dict[int, str], score: bool = False, num_cols: int = 3, figsize: Tuple[int, int] = (20, 30)) -> None:
    """Visualze the data"""
    boxes = target["boxes"].detach().numpy()
    labels = target["labels"].detach().numpy()
    masks = target["masks"].detach().numpy()
    if score:
        scores = target["scores"].detach().numpy()
    num = len(boxes)
    num_rows = math.ceil(num / num_cols)
    _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    ax = ax.flatten()

    for i in range(num):
        ax[i].imshow(img)
        ax[i].imshow(masks[i].squeeze(), alpha=0.7)

        label = labels[i]
        title = f"Label: {label} ({cats[label - 1]})"
        if score:
            title += f" Score: {scores[i]}"
        ax[i].set_title(title)

        x1, y1, x2, y2 = boxes[i]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax[i].add_patch(rect)

    for axi in ax[i:]:
        axi.axis("off")