from __future__ import annotations

import argparse

import torch

from .config import ensure_dir, load_config
from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from .diffusion import GaussianConditionalDiffusion
from .utils import pick_device, save_tensor_npy, save_tensor_png


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
    parser = argparse.ArgumentParser(description="Inference for palette_decoupling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg["dataset"][args.split])
    device = pick_device(cfg.get("device"))
    image_size = cfg["model"].get("image_size") or dataset[0]["condition"].shape[-1]
    model = GaussianConditionalDiffusion(
        condition_channels=dataset.condition_channels,
        target_channels=dataset.target_channels,
        image_size=image_size,
        inner_channel=cfg["model"]["inner_channel"],
        channel_mults=cfg["model"]["channel_mults"],
        attn_res=cfg["model"]["attn_res"],
        res_blocks=cfg["model"]["res_blocks"],
        dropout=cfg["model"].get("dropout", 0.0),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    output_dir = ensure_dir(args.output_dir)
    with torch.no_grad():
        for sample in dataset:
            condition = sample["condition"].unsqueeze(0).to(device)
            pred = model.sample(condition, sample_steps=cfg["diffusion"].get("sample_steps"))[0]
            name = str(sample["name"])
            save_tensor_npy(pred, output_dir / f"{name}.npy")
            for channel_idx in range(pred.shape[0]):
                save_tensor_png(pred[channel_idx : channel_idx + 1], output_dir / f"{name}_ch{channel_idx}.png")


if __name__ == "__main__":
    main()
