"""Define the neural network model by extending torch.nn.Module class"""

from typing import Any, Dict

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes: int) -> torch.nn.Module:
    """Get a model for instance segmentation
    Args:
        num_classes: Number of classes
    """
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def loss_fn(
    outputs: torch.Tensor, ground_truth: torch.Tensor, params: Dict[str, Any]
) -> torch.Tensor:
    """Compute the loss given outputs and ground_truth.
    Args:
        outputs: Output of network forward pass
        ground_truth: Batch of ground truth
        params: Hyperparameters
    Returns:
        loss for all the inputs in the batch
    """
    criterion = torch.nn.NLLLoss()
    loss = criterion(outputs, ground_truth)
    return loss


def accuracy(outputs: np.ndarray, labels: np.ndarray) -> int:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Output of the network B X num_classes
        labels: Ground truth labels Bx1
    Returns:
        Accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=-1)
    return np.sum(outputs == labels) / labels.size


# Maintain all metrics required during training and evaluation.
def get_metrics():
    """Returns a dictionary of all the metrics to be used"""
    metrics = {}
    return metrics
