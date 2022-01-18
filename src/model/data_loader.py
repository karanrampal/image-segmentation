"""Define custom dataset class extending the Pytorch Dataset class"""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.mask import decode, frPyObjects
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from utils import transform_utils, utils


class FashionpediaDataset(Dataset):
    """Custom class for Fashionpedia dataset
    Args:
        root: Directory containing the dataset
        file_path: Path of the train/val/test file relative to the root
        img_path: Path of the images for train/val/test relative to the root
        transforms: Data augmentation to be done
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        img_path: str,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.img_path = img_path
        self.transforms = transforms

        with open(os.path.join(root, file_path), "r") as f:
            self.data = json.load(f)

        self.im_id_path_dict = {
            blob["id"]: blob["file_name"] for blob in self.data["images"]
        }
        self.categories = {blob["id"]: blob["name"] for blob in self.data["categories"]}

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """Get an item from the dataset given the index idx"""
        annotation = self.data["annotations"][idx]

        im_name = self.im_id_path_dict[annotation["image_id"]]
        im_path = os.path.join(self.root, self.img_path, im_name)
        img = Image.open(im_path).convert("RGB")

        masks = np.zeros((len(annotation["segmentation"]), img.height, img.width))
        for i, tmp in enumerate(annotation["segmentation"]):
            rle = (
                frPyObjects(tmp, img.height, img.width)
                if isinstance(tmp, list)
                else tmp
            )
            mask = decode(rle)
            if mask.ndim == 3:
                mask = mask.sum(2, dtype=np.uint8, keepdims=False)
            masks[i, :, :] = mask
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        boxes = np.array(annotation["bbox"])
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        bboxes = torch.as_tensor(boxes, dtype=torch.float32)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.tensor(
            np.array(annotation["category_id"]) + 1, dtype=torch.int64
        )
        target["masks"] = masks
        target["image_id"] = torch.tensor([annotation["image_id"]], dtype=torch.int64)
        target["area"] = torch.tensor(annotation["area"], dtype=torch.float32)
        target["iscrowd"] = torch.tensor(annotation["iscrowd"], dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.data["annotations"])


def get_transform(is_train: bool) -> transform_utils.Compose:
    """Data augmentation
    Args:
        is_train: If the dataset is training
    Returns:
        Composition of all the data transforms
    """
    trans = []
    trans.append(transform_utils.ToTensor())
    if is_train:
        trans.append(transform_utils.RandomHorizontalFlip(0.5))
    return transform_utils.Compose(trans)


def get_dataloader(
    modes: List[str],
    params: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """Get DataLoader objects.
    Args:
        modes: Mode of operation i.e. 'train', 'val', 'test'
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataloaders = {}

    for mode in modes:
        if mode == "train":
            ds_dict = {
                "file_path": "annotations/train.json",
                "img_path": "images/train",
                "transforms": get_transform(True),
            }
            dl_dict = {
                "shuffle": False if params.distributed else True,
            }
        else:
            ds_dict = {
                "file_path": "annotations/val.json",
                "img_path": "images/test",
                "transforms": get_transform(False),
            }
            dl_dict = {
                "shuffle": False,
            }

        dataset = FashionpediaDataset(root=params.data_path, **ds_dict)
        if params.distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=params.world_size, rank=params.rank, shuffle=True, seed=42
            )
        else:
            sampler = None

        d_l = DataLoader(
            dataset,
            batch_size=params.batch_size,
            sampler=sampler,
            num_workers=params.num_workers,
            collate_fn=utils.collate_fn,
            pin_memory=params.pin_memory,
            **dl_dict
        )

        dataloaders[mode] = (d_l, sampler)

    return dataloaders
