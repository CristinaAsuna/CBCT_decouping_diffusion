from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from .metrics import compute_channelwise_fid, mae, mse, psnr, ssim


def build_dataset(dataset_cfg: dict):
    dataset_type = dataset_cfg.get("type", "paired_dirs")
    if dataset_type == "paired_dirs":
        return NpyConditionTargetDataset(
            condition_dir=dataset_cfg["condition_dir"],
            target_dirs=dataset_cfg["target_dirs"],
            image_size=dataset_cfg.get("image_size"),
            normalize=dataset_cfg.get("normalize", "range_m11"),
            names_file=dataset_cfg.get("names_file"),
        )
    if dataset_type == "case_folders":
        return CaseFolderNpyDataset(
            case_root=dataset_cfg["case_root"],
            condition_file=dataset_cfg.get("condition_file"),
            target_files=dataset_cfg.get("target_files"),
            image_size=dataset_cfg.get("image_size"),
            normalize=dataset_cfg.get("normalize", "range_m11"),
            case_names_file=dataset_cfg.get("case_names_file"),
            include_patterns=dataset_cfg.get("include_patterns"),
            variants=dataset_cfg.get("variants"),
            condition_template=dataset_cfg.get("condition_template"),
            target_templates=dataset_cfg.get("target_templates"),
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated .npy predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--with-fid", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg["dataset"][args.split])
    pred_dir = Path(args.pred_dir)
    preds = []
    targets = []
    for sample in dataset:
        pred_path = pred_dir / f"{sample['name']}.npy"
        if not pred_path.exists():
            continue
        preds.append(torch.from_numpy(np.load(pred_path)).float())
        targets.append(sample["target"].float())

    if not preds:
        raise ValueError("No prediction files were found for evaluation.")

    pred_tensor = torch.stack(preds, dim=0)
    target_tensor = torch.stack(targets, dim=0)
    results = {
        "mae": mae(pred_tensor, target_tensor),
        "mse": mse(pred_tensor, target_tensor),
        "psnr": psnr(pred_tensor, target_tensor),
        "ssim": ssim(pred_tensor, target_tensor),
    }
    if args.with_fid:
        results["fid_per_channel"] = compute_channelwise_fid(pred_tensor, target_tensor)
        results["fid_mean"] = float(sum(results["fid_per_channel"]) / len(results["fid_per_channel"]))
    print(results)


if __name__ == "__main__":
    main()
