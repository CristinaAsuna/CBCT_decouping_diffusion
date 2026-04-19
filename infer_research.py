from __future__ import annotations

import argparse

import torch

from .config import ensure_dir, load_config
from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from .diffusion import GaussianConditionalDiffusion
from .train_research import run_sampler
from .utils import pick_device, save_tensor_npy, save_tensor_png


def build_dataset(dataset_cfg: dict):
    dataset_type = dataset_cfg.get("type", "paired_dirs")
    common_kwargs = {
        "image_size": dataset_cfg.get("image_size"),
        "normalize": dataset_cfg.get("normalize", "range_m11"),
        "value_range": dataset_cfg.get("value_range"),
        "clip_range": dataset_cfg.get("clip_range"),
        "target_mode": dataset_cfg.get("target_mode", "multi_channel"),
        "side_labels": dataset_cfg.get("side_labels"),
    }
    if dataset_type == "paired_dirs":
        return NpyConditionTargetDataset(
            condition_dir=dataset_cfg["condition_dir"],
            target_dirs=dataset_cfg["target_dirs"],
            names_file=dataset_cfg.get("names_file"),
            **common_kwargs,
        )
    if dataset_type == "case_folders":
        return CaseFolderNpyDataset(
            case_root=dataset_cfg["case_root"],
            condition_file=dataset_cfg.get("condition_file"),
            target_files=dataset_cfg.get("target_files"),
            case_names_file=dataset_cfg.get("case_names_file"),
            include_patterns=dataset_cfg.get("include_patterns"),
            variants=dataset_cfg.get("variants"),
            condition_template=dataset_cfg.get("condition_template"),
            target_templates=dataset_cfg.get("target_templates"),
            split=dataset_cfg.get("split"),
            split_seed=dataset_cfg.get("split_seed", 1234),
            train_ratio=dataset_cfg.get("train_ratio", 0.9),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def build_model(cfg: dict, dataset, device: torch.device) -> GaussianConditionalDiffusion:
    image_size = cfg["model"].get("image_size") or dataset[0]["condition"].shape[-1]
    return GaussianConditionalDiffusion(
        condition_channels=dataset.condition_channels,
        target_channels=dataset.target_channels,
        image_size=image_size,
        inner_channel=cfg["model"]["inner_channel"],
        channel_mults=cfg["model"]["channel_mults"],
        attn_res=cfg["model"]["attn_res"],
        res_blocks=cfg["model"]["res_blocks"],
        dropout=cfg["model"].get("dropout", 0.0),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        num_side_classes=getattr(dataset, "num_side_classes", 0),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-style inference for palette_decoupling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    dataset = build_dataset(cfg["dataset"][args.split])
    model = build_model(cfg, dataset, device)
    state = torch.load(args.checkpoint, map_location=device)

    if args.weights == "ema" and "ema_model" in state:
        model.load_state_dict(state["ema_model"])
    else:
        model.load_state_dict(state["model"])
    model.eval()

    output_dir = ensure_dir(args.output_dir)
    sampler_cfg = cfg.get("validation", {}).get("sampler", {"sampler": "ddim", "steps": 50, "eta": 0.0})
    with torch.no_grad():
        for sample in dataset:
            condition = sample["condition"].unsqueeze(0).to(device)
            side_ids = sample.get("side_id")
            side_ids = side_ids.unsqueeze(0).to(device) if isinstance(side_ids, torch.Tensor) else None
            pred = run_sampler(model, condition, sampler_cfg, side_ids=side_ids)[0]
            name = str(sample["name"])
            save_tensor_npy(pred, output_dir / f"{name}.npy")
            for channel_idx in range(pred.shape[0]):
                save_tensor_png(pred[channel_idx : channel_idx + 1], output_dir / f"{name}_ch{channel_idx}.png")


if __name__ == "__main__":
    main()
