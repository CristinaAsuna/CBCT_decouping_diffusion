from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from config import ensure_dir
    from metrics import mae, mse, psnr, ssim
    from tiny_ablation.modules.branch_experiments import TinyBranchDiffusion
    from train_research import build_loader, is_better_metric, resolve_step_lr, set_optimizer_lr
    from utils import pick_device, save_tensor_npy, save_tensor_png, set_seed
except ImportError:  # pragma: no cover - used when imported as a package.
    from CBCT_decouping_diffusion.config import ensure_dir
    from CBCT_decouping_diffusion.metrics import mae, mse, psnr, ssim
    from CBCT_decouping_diffusion.tiny_ablation.modules.branch_experiments import TinyBranchDiffusion
    from CBCT_decouping_diffusion.train_research import build_loader, is_better_metric, resolve_step_lr, set_optimizer_lr
    from CBCT_decouping_diffusion.utils import pick_device, save_tensor_npy, save_tensor_png, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one tiny branch ablation.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def maybe_init_tensorboard(logging_cfg: dict[str, Any], output_dir: Path):
    if not logging_cfg.get("use_tensorboard", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard is not installed; tiny training will continue without it.")
        return None
    logdir = logging_cfg.get("tensorboard_logdir") or str(output_dir / "tensorboard")
    ensure_dir(logdir)
    print({"tensorboard_logdir": str(logdir)})
    return SummaryWriter(log_dir=str(logdir))


def add_scalars(writer, namespace: str, values: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in values.items():
        if key == "step":
            continue
        if isinstance(value, bool):
            writer.add_scalar(f"{namespace}/{key}", int(value), step)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(f"{namespace}/{key}", float(value), step)


def add_preview_images(writer, step: int, condition: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> None:
    if writer is None:
        return

    def prep(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().float().cpu().clamp(-1.0, 1.0)
        return (tensor + 1.0) / 2.0

    writer.add_image("preview/condition_full", prep(condition[0]), step)
    for channel_idx in range(target.shape[1]):
        writer.add_image(f"preview/target_ch{channel_idx}", prep(target[0, channel_idx : channel_idx + 1]), step)
        writer.add_image(f"preview/pred_ch{channel_idx}", prep(pred[0, channel_idx : channel_idx + 1]), step)


def average_dict(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(sum(item[key] for item in items) / len(items)) for key in keys}


def build_model(cfg: dict[str, Any], device: torch.device) -> TinyBranchDiffusion:
    train_dataset_cfg = cfg["dataset"]["train"]
    condition_channels = int(train_dataset_cfg.get("condition_channels", 1))
    target_channels = len(train_dataset_cfg.get("target_templates", [])) or int(train_dataset_cfg.get("target_channels", 2))
    if target_channels != 2:
        raise ValueError("Branch tiny ablation expects dual left/right target templates.")
    diffusion_cfg = cfg["diffusion"]
    model = TinyBranchDiffusion(
        condition_channels=condition_channels,
        target_channels=target_channels,
        model_cfg=cfg["model"],
        beta_schedule=diffusion_cfg["beta_schedule"],
        branch_loss_weights=diffusion_cfg.get("branch_loss_weights"),
        consistency_cfg=diffusion_cfg.get("consistency_loss"),
        data_norm_cfg=train_dataset_cfg,
        experiment_cfg=cfg.get("tiny_branch", {}),
    )
    return model.to(device)


def _adapt_first_input_conv_weight(model: torch.nn.Module, key: str, value: torch.Tensor) -> torch.Tensor | None:
    current = model.state_dict()
    if key not in current:
        return None
    target = current[key]
    if tuple(target.shape[:1] + target.shape[2:]) != tuple(value.shape[:1] + value.shape[2:]):
        return None
    if value.ndim != 4 or target.ndim != 4:
        return None
    denoise_fn = getattr(model, "denoise_fn", None)
    if denoise_fn is None or key != "denoise_fn.unet.input_blocks.0.0.weight":
        return None

    target_channels = int(getattr(denoise_fn, "target_channels", 0))
    condition_channels = int(getattr(denoise_fn, "condition_channels", 0))
    condition_input_channels = int(getattr(denoise_fn, "condition_input_channels", max(target.shape[1] - target_channels, 0)))
    encoded_channels = int(getattr(denoise_fn, "encoded_channels", 0))
    retain_raw = bool(getattr(denoise_fn, "retain_raw_condition", True))
    old_in = int(value.shape[1])
    new_in = int(target.shape[1])
    if target_channels <= 0 or old_in < target_channels or new_in < target_channels:
        return None

    old_condition_channels = old_in - target_channels
    if old_condition_channels <= 0:
        return None

    adapted = torch.zeros_like(target)
    if retain_raw:
        raw_to_copy = min(condition_channels, old_condition_channels, new_in)
        adapted[:, :raw_to_copy] = value[:, :raw_to_copy]
    elif encoded_channels > 0:
        # Encoder-only warm-start cannot be function-preserving. Sharing the old
        # raw-condition filter across encoded channels keeps the first layer on a
        # comparable scale while the encoder learns useful features.
        raw = value[:, :old_condition_channels].mean(dim=1, keepdim=True)
        adapted[:, :encoded_channels] = raw / max(encoded_channels, 1)

    old_target_start = old_in - target_channels
    new_target_start = condition_input_channels
    target_to_copy = min(target_channels, new_in - new_target_start)
    if target_to_copy > 0:
        adapted[:, new_target_start : new_target_start + target_to_copy] = value[:, old_target_start : old_target_start + target_to_copy]
    return adapted


def _adapt_unit_conv_projection(target: torch.Tensor, value: torch.Tensor) -> torch.Tensor | None:
    if value.ndim == 3 and target.ndim == 4 and value.shape[0] == target.shape[0] and value.shape[1] == target.shape[1]:
        if value.shape[-1] == 1 and target.shape[-2:] == (1, 1):
            return value[:, :, None, :]
    if value.ndim == 4 and target.ndim == 3 and value.shape[0] == target.shape[0] and value.shape[1] == target.shape[1]:
        if value.shape[-2:] == (1, 1) and target.shape[-1] == 1:
            return value[:, :, 0, :]
    return None


def filter_compatible_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    adapt_input_conv: bool = True,
    adapt_unit_projections: bool = True,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    adapted_keys: list[str] = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif adapt_input_conv and (adapted := _adapt_first_input_conv_weight(model, key, value)) is not None:
            compatible[key] = adapted
            adapted_keys.append(key)
        elif adapt_unit_projections and key in current and (adapted := _adapt_unit_conv_projection(current[key], value)) is not None:
            compatible[key] = adapted
            adapted_keys.append(key)
        else:
            skipped.append(key)
    return compatible, skipped, adapted_keys


def load_checkpoint_if_needed(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]) -> tuple[int, float]:
    resume_path = train_cfg.get("resume_checkpoint")
    if not resume_path:
        return 0, float("inf")
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"resume_checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    strict = bool(train_cfg.get("strict_resume", True))
    if strict:
        model.load_state_dict(state, strict=True)
    else:
        compatible, skipped, adapted = filter_compatible_state(
            model,
            state,
            adapt_input_conv=bool(train_cfg.get("adapt_input_conv", True)),
            adapt_unit_projections=bool(train_cfg.get("adapt_unit_projections", True)),
        )
        model_state = model.state_dict()
        model_state.update(compatible)
        model.load_state_dict(model_state, strict=False)
        print(
            {
                "warm_start_loaded": len(compatible),
                "warm_start_adapted": len(adapted),
                "warm_start_skipped": len(skipped),
                "warm_start_adapted_keys": adapted,
            }
        )
    if train_cfg.get("resume_optimizer", False) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_metric = float(checkpoint.get("best_metric", float("inf"))) if train_cfg.get("inherit_best_metric", False) else float("inf")
    return int(checkpoint.get("step", 0)), best_metric


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: dict[str, Any], step: int, best_metric: float) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": int(step),
            "best_metric": float(best_metric),
        },
        path,
    )


def write_parameter_summary(output_dir: Path, model: TinyBranchDiffusion) -> dict[str, int]:
    summary = model.parameter_summary()
    (output_dir / "parameter_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@torch.no_grad()
def validate(
    model: TinyBranchDiffusion,
    val_loader,
    device: torch.device,
    validation_cfg: dict[str, Any],
    output_dir: Path,
    step: int,
    writer,
) -> dict[str, float]:
    model.eval()
    sample_cfg = validation_cfg.get("sampler", {})
    max_batches = int(validation_cfg.get("max_batches", 4))
    batch_metrics: list[dict[str, float]] = []
    preview_written = False
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        loss = model(target, condition, global_step=step)
        if sample_cfg.get("sampler", "ddim") == "ddpm":
            pred = model.sample(condition, sample_steps=sample_cfg.get("steps"))
        else:
            pred = model.sample_ddim(
                condition,
                sample_steps=int(sample_cfg.get("steps", 20)),
                eta=float(sample_cfg.get("eta", 0.0)),
            )
        pred = pred.clamp(-1.0, 1.0)
        breakdown = copy.deepcopy(model.last_loss_breakdown)
        metrics = {
            "val_loss": float(loss.detach().item()),
            "val_mae": mae(pred, target),
            "val_mse": mse(pred, target),
            "val_psnr": psnr(pred, target),
            "val_ssim": ssim(pred, target),
            "val_left_mae": mae(pred[:, 0:1], target[:, 0:1]),
            "val_right_mae": mae(pred[:, 1:2], target[:, 1:2]),
            "val_left_psnr": psnr(pred[:, 0:1], target[:, 0:1]),
            "val_right_psnr": psnr(pred[:, 1:2], target[:, 1:2]),
            "val_diff_loss": float(breakdown.get("diff_loss", 0.0)),
            "val_left_diff_loss": float(breakdown.get("left_diff_loss", 0.0)),
            "val_right_diff_loss": float(breakdown.get("right_diff_loss", 0.0)),
            "val_consistency_loss": float(breakdown.get("consistency_loss", 0.0)),
            "val_consistency_term": float(breakdown.get("consistency_term", 0.0)),
            "val_consistency_ratio": float(breakdown.get("consistency_ratio", 0.0)),
            "val_consistency_active": float(breakdown.get("consistency_active", 0.0)),
        }
        metrics.update(model.consistency_metrics(condition, pred, "val_pred"))
        metrics.update(model.consistency_metrics(condition, target, "val_target"))
        batch_metrics.append(metrics)
        if not preview_written:
            preview_dir = output_dir / "previews"
            save_tensor_png(condition[0], preview_dir / f"step_{step:06d}_condition.png")
            save_tensor_npy(pred[0], preview_dir / f"step_{step:06d}_pred.npy")
            for channel_idx in range(pred.shape[1]):
                save_tensor_png(target[0, channel_idx : channel_idx + 1], preview_dir / f"step_{step:06d}_target_ch{channel_idx}.png")
                save_tensor_png(pred[0, channel_idx : channel_idx + 1], preview_dir / f"step_{step:06d}_pred_ch{channel_idx}.png")
            add_preview_images(writer, step, condition, target, pred)
            preview_written = True
    model.train()
    result = average_dict(batch_metrics)
    result["step"] = float(step)
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    device = pick_device(cfg.get("device"))
    output_dir = ensure_dir(cfg["output"]["root"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    writer = maybe_init_tensorboard(cfg.get("logging", {}), output_dir)

    _, train_loader, _ = build_loader(
        cfg["dataset"]["train"],
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    _, val_loader, _ = build_loader(
        cfg["dataset"]["val"],
        batch_size=int(cfg["train"].get("val_batch_size", cfg["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    start_step, best_metric = load_checkpoint_if_needed(model, optimizer, cfg["train"])
    parameter_summary = write_parameter_summary(output_dir, model)
    add_scalars(writer, "model", parameter_summary, 0)
    best_metric_name = cfg["train"].get("best_metric", "val_mae")
    print(
        {
            "device": str(device),
            "start_step": start_step,
            "best_metric": best_metric,
            **parameter_summary,
            "attention_type": model.denoise_fn.attention_type,
            "window_attention_blocks": model.denoise_fn.window_attention_blocks,
        }
    )

    train_iter = iter(train_loader)
    final_step = int(cfg["train"]["max_steps"])
    if start_step > 0 and cfg["train"].get("finetune_steps") is not None:
        final_step = start_step + int(cfg["train"]["finetune_steps"])
    progress = tqdm(range(start_step + 1, final_step + 1), desc=Path(args.config).stem)
    for step in progress:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        current_lr = resolve_step_lr(cfg["train"], step)
        set_optimizer_lr(optimizer, current_lr)
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(target, condition, global_step=step)
        loss.backward()
        if cfg["train"].get("grad_clip_norm") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
        optimizer.step()

        train_log = copy.deepcopy(model.last_loss_breakdown)
        train_log.update({"loss": float(loss.detach().item()), "lr": current_lr, "step": float(step)})
        progress.set_postfix(
            loss=f"{train_log['loss']:.4g}",
            left=f"{train_log['left_diff_loss']:.4g}",
            right=f"{train_log['right_diff_loss']:.4g}",
            cons_r=f"{train_log['consistency_ratio']:.3f}",
        )
        if step % int(cfg["train"].get("log_every_steps", 20)) == 0:
            add_scalars(writer, "train", train_log, step)
            if writer is not None:
                writer.flush()

        if step % int(cfg["train"].get("validate_every_steps", 500)) == 0:
            val_metrics = validate(model, val_loader, device, cfg["validation"], output_dir, step, writer)
            val_metrics["lr"] = current_lr
            print(val_metrics)
            add_scalars(writer, "val", val_metrics, step)
            metric_value = float(val_metrics[best_metric_name])
            if is_better_metric(best_metric_name, metric_value, best_metric):
                best_metric = metric_value
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, cfg, step, best_metric)
            if writer is not None:
                writer.flush()

        if step % int(cfg["train"].get("save_every_steps", 5000)) == 0:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, cfg, step, best_metric)
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step, best_metric)

    save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, cfg, final_step, best_metric)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
