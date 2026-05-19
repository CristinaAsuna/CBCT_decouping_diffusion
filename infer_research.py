
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

try:
    from .checkpointing import load_checkpoint_file, select_model_state
    from .config import ensure_dir, load_config
    from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
    from .diffusion import GaussianConditionalDiffusion
    from .physics import (
        apply_branch_physics_correction,
        compute_branch_physics_metrics,
        safe_filename,
        save_branch_comparison_visualization,
        save_physics_visualization,
    )
    from .utils import pick_device, save_tensor_gif, save_tensor_grid_png, save_tensor_npy, save_tensor_png
except ImportError:
    from checkpointing import load_checkpoint_file, select_model_state
    from config import ensure_dir, load_config
    from dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
    from diffusion import GaussianConditionalDiffusion
    from physics import (
        apply_branch_physics_correction,
        compute_branch_physics_metrics,
        safe_filename,
        save_branch_comparison_visualization,
        save_physics_visualization,
    )
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
        branch_decoder_cfg=cfg["model"].get("branch_decoder"),
        branch_loss_weights=cfg["diffusion"].get("branch_loss_weights"),
        consistency_cfg=cfg["diffusion"].get("consistency_loss"),
        data_norm_cfg=cfg.get("dataset", {}).get("train") or cfg.get("dataset", {}).get("val") or {},
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
    parser.add_argument("--no-physics-visuals", action="store_true", help="Disable branch physics visualization output")
    parser.add_argument("--no-branch-quality-visuals", action="store_true", help="Disable GT-vs-pred branch quality visualization output")
    parser.add_argument("--branch-correction", default="none", choices=["none", "equal", "proportional"], help="Apply inference-time full = left + right correction for branch models")
    parser.add_argument("--branch-correction-strength", type=float, default=1.0, help="Correction strength; 1.0 makes the corrected sum match full exactly before optional visualization")
    parser.add_argument("--branch-correction-min-residual-mae", type=float, default=None, help="Only correct samples whose physics_residual_mae is above this threshold")
    parser.add_argument("--branch-correction-min-rel-mae", type=float, default=None, help="Only correct samples whose physics_residual_rel_mae_masked is above this threshold")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.get("device"))
    state = load_checkpoint_file(args.checkpoint, device)
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

    model_state, model_state_name = select_model_state(state, weights=args.weights)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    is_branch_model = bool(getattr(getattr(model, "denoise_fn", None), "branch_decoder", False))
    print({"loaded_weights": model_state_name, "checkpoint": str(args.checkpoint)})

    output_dir = ensure_dir(args.output_dir)
    physics_dir = ensure_dir(output_dir / "physics")
    branch_quality_dir = ensure_dir(output_dir / "branch_quality")
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

            branch_metrics = None
            raw_pred = pred.detach().clone()
            correction_applied = False
            if is_branch_model and pred.shape[0] >= 2:
                target = sample.get("target")
                target_tensor = target if isinstance(target, torch.Tensor) else None
                branch_metrics = compute_branch_physics_metrics(
                    sample["condition"],
                    pred,
                    target_tensor,
                    data_norm_cfg=dataset_cfg,
                )
                should_correct = args.branch_correction != "none"
                if should_correct and args.branch_correction_min_residual_mae is not None:
                    should_correct = branch_metrics["physics_residual_mae"] > args.branch_correction_min_residual_mae
                if should_correct and args.branch_correction_min_rel_mae is not None:
                    should_correct = branch_metrics["physics_residual_rel_mae_masked"] > args.branch_correction_min_rel_mae
                if args.branch_correction != "none" and not should_correct:
                    print(
                        {
                            "sample": name,
                            "branch_correction": args.branch_correction,
                            "correction_applied": False,
                            "physics_residual_mae": branch_metrics["physics_residual_mae"],
                            "physics_residual_rel_mae_masked": branch_metrics["physics_residual_rel_mae_masked"],
                        }
                    )
                if should_correct:
                    corrected_pred = apply_branch_physics_correction(
                        sample["condition"],
                        pred,
                        data_norm_cfg=dataset_cfg,
                        mode=args.branch_correction,
                        strength=args.branch_correction_strength,
                    )
                    corrected_metrics = compute_branch_physics_metrics(
                        sample["condition"],
                        corrected_pred,
                        target_tensor,
                        data_norm_cfg=dataset_cfg,
                    )
                    print(
                        {
                            "sample": name,
                            "branch_correction": args.branch_correction,
                            "branch_correction_strength": args.branch_correction_strength,
                            "correction_applied": True,
                            "before_physics_residual_mae": branch_metrics["physics_residual_mae"],
                            "after_physics_residual_mae": corrected_metrics["physics_residual_mae"],
                            "before_left_loss": branch_metrics.get("left_loss"),
                            "after_left_loss": corrected_metrics.get("left_loss"),
                            "before_right_loss": branch_metrics.get("right_loss"),
                            "after_right_loss": corrected_metrics.get("right_loss"),
                        }
                    )
                    pred = corrected_pred
                    branch_metrics = corrected_metrics
                    correction_applied = True

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
            if branch_metrics is not None:
                print(
                    {
                        "sample": name,
                        "branch_correction": args.branch_correction,
                        "branch_left_loss": branch_metrics.get("left_loss"),
                        "branch_right_loss": branch_metrics.get("right_loss"),
                        "branch_left_mae": branch_metrics.get("left_mae"),
                        "branch_right_mae": branch_metrics.get("right_mae"),
                        "physics_residual_mae": branch_metrics["physics_residual_mae"],
                        "physics_residual_rmse": branch_metrics["physics_residual_rmse"],
                        "physics_residual_rel_mae_masked": branch_metrics["physics_residual_rel_mae_masked"],
                        "physics_corr_masked": branch_metrics["physics_corr_masked"],
                    }
                )
                if not args.no_physics_visuals:
                    title = (
                        f"{name} | "
                        f"L loss={branch_metrics.get('left_loss', 0.0):.6g}, "
                        f"R loss={branch_metrics.get('right_loss', 0.0):.6g}, "
                        f"res MAE={branch_metrics['physics_residual_mae']:.6g}"
                    )
                    save_physics_visualization(
                        sample["condition"],
                        pred,
                        physics_dir / f"{safe_filename(name)}_physics.png",
                        title=title,
                        data_norm_cfg=dataset_cfg,
                    )
                target = sample.get("target")
                if not args.no_branch_quality_visuals and isinstance(target, torch.Tensor) and target.shape[0] >= 2:
                    branch_title = (
                        f"{name} | "
                        f"L MAE={branch_metrics.get('left_mae', 0.0):.6g}, "
                        f"R MAE={branch_metrics.get('right_mae', 0.0):.6g}, "
                        f"res MAE={branch_metrics['physics_residual_mae']:.6g}"
                    )
                    save_branch_comparison_visualization(
                        sample["condition"],
                        pred,
                        target,
                        branch_quality_dir / f"{safe_filename(name)}_branch_quality.png",
                        title=branch_title,
                        data_norm_cfg=dataset_cfg,
                    )
                    if correction_applied:
                        raw_metrics = compute_branch_physics_metrics(
                            sample["condition"],
                            raw_pred,
                            target,
                            data_norm_cfg=dataset_cfg,
                        )
                        raw_title = (
                            f"{name} | raw | "
                            f"L MAE={raw_metrics.get('left_mae', 0.0):.6g}, "
                            f"R MAE={raw_metrics.get('right_mae', 0.0):.6g}, "
                            f"res MAE={raw_metrics['physics_residual_mae']:.6g}"
                        )
                        save_branch_comparison_visualization(
                            sample["condition"],
                            raw_pred,
                            target,
                            branch_quality_dir / f"{safe_filename(name)}_raw_branch_quality.png",
                            title=raw_title,
                            data_norm_cfg=dataset_cfg,
                        )


if __name__ == "__main__":
    main()
