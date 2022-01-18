#!/usr/bin/env python3
"""Make Fashionpedia dataset"""

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd


def args_parser() -> argparse.Namespace:
    """Parse cli arguments"""
    parser = argparse.ArgumentParser(description="CLI argument parser")
    parser.add_argument(
        "-d",
        "--data-path",
        default="/dbfs/fashionpedia",
        type=str,
        help="Root data directory",
    )
    parser.add_argument(
        "-j",
        "--json-path",
        default="annotations/instances_attributes_train2020.json",
        type=str,
        help="Relative path to train/val/test json file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="annotations/train.json",
        type=str,
        help="Relative path to output file",
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = args_parser()

    with open(os.path.join(args.data_path, args.json_path), "r") as f:
        data = json.load(f)

    data_df = pd.read_json(json.dumps(data["annotations"]))

    data_df = data_df.drop(["id", "attribute_ids"], axis=1)
    valid_df = data_df[~(data_df["area"] == 0.0)]
    out_df = valid_df.groupby("image_id").agg(list)

    out_dict: Dict[str, Any] = {}
    out_dict["annotations"] = out_df.reset_index().to_dict(orient="records")
    out_dict["categories"] = data["categories"]
    out_dict["images"] = data["images"]

    with open(os.path.join(args.data_path, args.output_file), "w") as f:
        json.dump(out_dict, f)


if __name__ == "__main__":
    main()
