from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

try:
    from .config import ensure_dir, load_config
    from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
    from .diffusion import GaussianConditionalDiffusion
    from .utils import pick_device, save_tensor_gif, save_tensor_grid_png, save_tensor_npy, save_tensor_png
except ImportError:
    from config import ensure_dir, load_config
    from dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
    from diffusion import GaussianConditionalDiffusion
    from utils import pick_device, save_tensor_gif, save_tensor_grid_png, save_tensor_npy, save_tensor_png


def build_dataset(dataset_cfg: dict):
    dataset_type = dataset_cfg.get("type", "paired_dirs")
    common_kwargs = {
        "image_size": dataset_cfg.get("image_size"),
        "normalize": dataset_cfg.get("normalize", "range_m11"),
        "value_range": dataset_cfg.get("value_range"),
        "clip_range": dataset_cfg.get("clip_range"),
        "target_mode": dataset_cfg.get("target_mode"),
        "target_side": dataset_cfg.get("target_side"),
        "target_sides": dataset_cfg.get("target_sides"),
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
            target_template=dataset_cfg.get("target_template"),
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
        num_side_classes=getattr(dataset, "num_side_classes", None),
    ).to(device)


def merge_infer_config(cli_cfg: dict, checkpoint_cfg: dict | None) -> dict:
    merged = copy.deepcopy(cli_cfg)
    if not checkpoint_cfg:
        return merged
    for key in ("dataset", "model", "diffusion", "validation"):
        if key in checkpoint_cfg:
            merged[key] = copy.deepcopy(checkpoint_cfg[key])
    return merged


def run_sampler(
    model: GaussianConditionalDiffusion,
    condition: torch.Tensor,
    sample_cfg: dict,
    side_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    sampler = sample_cfg.get("sampler", "ddim")
    steps = sample_cfg.get("steps", 50)
    if sampler == "ddim":
        return model.sample_ddim(condition, sample_steps=steps, eta=sample_cfg.get("eta", 0.0), side_labels=side_labels)
    if sampler == "ddpm":
        return model.sample(condition, sample_steps=steps, side_labels=side_labels)
    raise ValueError(f"Unsupported sampler: {sampler}")


def get_sample_side_label(sample: dict, device: torch.device) -> torch.Tensor | None:
    side = sample.get("side")
    if side is None:
        return None
    if not isinstance(side, torch.Tensor):
        raise TypeError("sample['side'] must be a torch.Tensor")
    return side.unsqueeze(0).to(device=device, dtype=torch.long) if side.ndim == 0 else side.to(device=device, dtype=torch.long)


def select_sample(
    dataset,
    case_name: str | None = None,
    variant: str | None = None,
    sample_name: str | None = None,
    side: str | None = None,
):
    if sample_name is not None:
        for sample in dataset:
            if str(sample["name"]) == sample_name:
                return sample
        raise ValueError(f"sample_name={sample_name} was not found in dataset")

    if case_name is None:
        return dataset[0]

    prefix = f"{case_name}__"
    matches = [sample for sample in dataset if str(sample["name"]).startswith(prefix)]
    if variant is not None:
        variant_prefix = f"{case_name}__{variant}"
        matches = [sample for sample in matches if str(sample["name"]).startswith(variant_prefix)]
    if side is not None:
        matches = [sample for sample in matches if str(sample.get("side_name", "")) == side or str(sample["name"]).endswith(f"__{side}")]
    if not matches:
        raise ValueError(f"No sample found for case_name={case_name}, variant={variant}, side={side}")
    return matches[0]


def save_trace_visuals(
    output_dir: Path,
    name: str,
    trace: dict[str, torch.Tensor | list[torch.Tensor] | list[int]],
    channel_idx: int = 0,
    gif_duration_ms: int = 120,
) -> None:
    sample_history = trace["samples"]
    pred_history = trace["predictions"]
    timesteps = trace["timesteps"]
    assert isinstance(sample_history, list)
    assert isinstance(pred_history, list)
    assert isinstance(timesteps, list)

    sample_frames = [tensor[0, channel_idx : channel_idx + 1] for tensor in sample_history]
    pred_frames = [tensor[0, channel_idx : channel_idx + 1] for tensor in pred_history]

    trace_dir = ensure_dir(output_dir / f"{name}_trace")
    for idx, frame in enumerate(sample_frames):
        timestep = timesteps[idx] if idx < len(timesteps) else -1
        save_tensor_png(frame, trace_dir / f"sample_step_{idx:03d}_t{timestep:04d}_ch{channel_idx}.png")
    for idx, frame in enumerate(pred_frames):
        timestep = timesteps[idx] if idx < len(timesteps) else -1
        save_tensor_png(frame, trace_dir / f"pred_x0_step_{idx:03d}_t{timestep:04d}_ch{channel_idx}.png")

    save_tensor_gif(sample_frames, output_dir / f"{name}_ch{channel_idx}_sampling.gif", duration_ms=gif_duration_ms)
    save_tensor_grid_png(sample_frames, output_dir / f"{name}_ch{channel_idx}_sampling_grid.png")
    save_tensor_gif(pred_frames, output_dir / f"{name}_ch{channel_idx}_predx0.gif", duration_ms=gif_duration_ms)
    save_tensor_grid_png(pred_frames, output_dir / f"{name}_ch{channel_idx}_predx0_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-style inference for palette_decoupling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    parser.add_argument("--case-root", default=None, help="Optional override for dataset.case_root")
    parser.add_argument("--case-name", default=None, help="Case folder name such as Bone_0001")
    parser.add_argument("--variant", default=None, help="Variant suffix such as std or aug")
    parser.add_argument("--side", default=None, help="Optional side label such as left or right")
    parser.add_argument("--sample-name", default=None, help="Exact sample name, e.g. Bone_0001__std")
    parser.add_argument("--save-trace", action="store_true", help="Save DDIM intermediate sampling results")
    parser.add_argument("--trace-channel", type=int, default=0, help="Target channel index to visualize")
    parser.add_argument("--gif-duration-ms", type=int, default=120, help="Frame duration for exported GIF")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to infer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    state = torch.load(args.checkpoint, map_location=device)
    checkpoint_cfg = state.get("config") if isinstance(state, dict) else None
    cfg = merge_infer_config(cfg, checkpoint_cfg)
    dataset_cfg = dict(cfg["dataset"][args.split])
    if args.case_root is not None:
        dataset_cfg["case_root"] = args.case_root
    if args.case_name is not None or args.sample_name is not None:
        dataset_cfg.pop("split", None)
        dataset_cfg.pop("train_ratio", None)
        dataset_cfg.pop("split_seed", None)
    dataset = build_dataset(dataset_cfg)
    model = build_model(cfg, dataset, device)

    if args.weights == "ema" and isinstance(state, dict) and "ema_model" in state:
        model.load_state_dict(state["ema_model"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    output_dir = ensure_dir(args.output_dir)
    sampler_cfg = cfg.get("validation", {}).get("sampler", {"sampler": "ddim", "steps": 50, "eta": 0.0})
    selected_sample = None
    if args.case_name is not None or args.sample_name is not None:
        selected_sample = select_sample(dataset, case_name=args.case_name, variant=args.variant, sample_name=args.sample_name, side=args.side)
        dataset_iterable = [selected_sample]
    else:
        dataset_iterable = dataset

    with torch.no_grad():
        for sample_idx, sample in enumerate(dataset_iterable):
            if args.limit is not None and sample_idx >= args.limit:
                break
            condition = sample["condition"].unsqueeze(0).to(device)
            side_labels = get_sample_side_label(sample, device)
            name = str(sample["name"])
            if args.save_trace and sampler_cfg.get("sampler", "ddim") == "ddim":
                trace = model.sample_ddim_trace(
                    condition,
                    sample_steps=int(sampler_cfg.get("steps", 50)),
                    eta=float(sampler_cfg.get("eta", 0.0)),
                    side_labels=side_labels,
                )
                pred = trace["final"][0]
                save_trace_visuals(
                    output_dir=output_dir,
                    name=name,
                    trace=trace,
                    channel_idx=args.trace_channel,
                    gif_duration_ms=args.gif_duration_ms,
                )
            else:
                pred = run_sampler(model, condition, sampler_cfg, side_labels=side_labels)[0]

            save_tensor_npy(sample["condition"], output_dir / f"{name}_condition.npy")
            save_tensor_png(sample["condition"][0:1], output_dir / f"{name}_condition_ch0.png")
            if "target" in sample:
                save_tensor_npy(sample["target"], output_dir / f"{name}_target.npy")
                if args.trace_channel < sample["target"].shape[0]:
                    save_tensor_png(
                        sample["target"][args.trace_channel : args.trace_channel + 1],
                        output_dir / f"{name}_target_ch{args.trace_channel}.png",
                    )
            save_tensor_npy(pred, output_dir / f"{name}.npy")
            for channel_idx in range(pred.shape[0]):
                save_tensor_png(pred[channel_idx : channel_idx + 1], output_dir / f"{name}_ch{channel_idx}.png")


if __name__ == "__main__":
    main()
