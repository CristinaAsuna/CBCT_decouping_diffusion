from __future__ import annotations

import argparse
import copy
from itertools import cycle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config
from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from .diffusion import GaussianConditionalDiffusion
from .metrics import compute_channelwise_fid, mae, mse, psnr, ssim
from .utils import pick_device, save_tensor_npy, save_tensor_png, set_seed


def build_dataset(dataset_cfg: dict):
    dataset_type = dataset_cfg.get("type", "paired_dirs")
    common_kwargs = {
        "image_size": dataset_cfg.get("image_size"),
        "normalize": dataset_cfg.get("normalize", "range_m11"),
        "value_range": dataset_cfg.get("value_range"),
        "clip_range": dataset_cfg.get("clip_range"),
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


def build_loader(dataset_cfg: dict, batch_size: int, shuffle: bool, num_workers: int):
    dataset = build_dataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
    return dataset, loader


def maybe_init_swanlab(cfg: dict):
    if not cfg.get("use_swanlab", False):
        return None
    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed, training will continue without it.")
        return None
    return swanlab.init(
        project=cfg.get("project", "palette_decoupling"),
        experiment_name=cfg.get("experiment_name"),
        config=cfg,
    )


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.ema_model.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            if not torch.is_floating_point(value):
                value.copy_(model_state[key])
            else:
                value.mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)


def run_sampler(model: GaussianConditionalDiffusion, condition: torch.Tensor, sample_cfg: dict) -> torch.Tensor:
    sampler = sample_cfg.get("sampler", "ddim")
    steps = sample_cfg.get("steps", 50)
    if sampler == "ddim":
        return model.sample_ddim(condition, sample_steps=steps, eta=sample_cfg.get("eta", 0.0))
    if sampler == "ddpm":
        return model.sample(condition, sample_steps=steps)
    raise ValueError(f"Unsupported sampler: {sampler}")


@torch.no_grad()
def validate(model, loader, device, sample_cfg: dict, max_batches: int | None = None, compute_fid: bool = False) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    mses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    fid_preds = []
    fid_targets = []
    progress_total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch_idx, batch in enumerate(tqdm(loader, desc="validate", leave=False, total=progress_total)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        loss = model(target, condition)
        pred = run_sampler(model, condition, sample_cfg)
        losses.append(float(loss.item()))
        maes.append(mae(pred, target))
        mses.append(mse(pred, target))
        psnrs.append(psnr(pred, target))
        ssims.append(ssim(pred, target))
        if compute_fid:
            fid_preds.append(pred.detach().cpu())
            fid_targets.append(target.detach().cpu())

    metrics = {
        "val_loss": sum(losses) / max(len(losses), 1),
        "val_mae": sum(maes) / max(len(maes), 1),
        "val_mse": sum(mses) / max(len(mses), 1),
        "val_psnr": sum(psnrs) / max(len(psnrs), 1),
        "val_ssim": sum(ssims) / max(len(ssims), 1),
    }
    if compute_fid and fid_preds:
        pred_tensor = torch.cat(fid_preds, dim=0)
        target_tensor = torch.cat(fid_targets, dim=0)
        fid_per_channel = compute_channelwise_fid(pred_tensor, target_tensor)
        metrics["fid_mean"] = float(sum(fid_per_channel) / len(fid_per_channel))
    return metrics


def save_checkpoint(path, model, optimizer, ema: EMA, step: int, cfg: dict, best_metric: float | None = None) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": step,
            "best_metric": best_metric,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-style step-based trainer for palette_decoupling.")
    parser.add_argument("--config", required=True, help="Path to yaml/json config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    device = pick_device(cfg.get("device"))

    output_dir = ensure_dir(cfg["output"]["root"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    sample_dir = ensure_dir(output_dir / "samples")

    train_dataset, train_loader = build_loader(
        cfg["dataset"]["train"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    val_dataset, val_loader = build_loader(
        cfg["dataset"]["val"],
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    print(
        {
            "device": str(device),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "condition_channels": train_dataset.condition_channels,
            "target_channels": train_dataset.target_channels,
        }
    )

    image_size = cfg["model"].get("image_size") or train_dataset[0]["condition"].shape[-1]
    model = GaussianConditionalDiffusion(
        condition_channels=train_dataset.condition_channels,
        target_channels=train_dataset.target_channels,
        image_size=image_size,
        inner_channel=cfg["model"]["inner_channel"],
        channel_mults=cfg["model"]["channel_mults"],
        attn_res=cfg["model"]["attn_res"],
        res_blocks=cfg["model"]["res_blocks"],
        dropout=cfg["model"].get("dropout", 0.0),
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    ema = EMA(model, decay=cfg["train"].get("ema_decay", 0.9999))
    run = maybe_init_swanlab(cfg.get("logging", {}))

    max_steps = int(cfg["train"]["max_steps"])
    validate_every = int(cfg["train"]["validate_every_steps"])
    save_every = int(cfg["train"].get("save_every_steps", validate_every))
    log_every = int(cfg["train"].get("log_every_steps", 50))
    ema_start = int(cfg["train"].get("ema_start_step", 0))
    best_metric = float("inf")
    best_metric_name = cfg["train"].get("best_metric", "val_mae")
    train_iter = cycle(train_loader)
    progress = tqdm(range(1, max_steps + 1), desc="train steps", leave=True)
    recent_losses: list[float] = []

    for step in progress:
        batch = next(train_iter)
        model.train()
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(target, condition)
        loss.backward()
        optimizer.step()
        if step >= ema_start:
            ema.update(model)

        loss_value = float(loss.item())
        recent_losses.append(loss_value)
        if len(recent_losses) > log_every:
            recent_losses.pop(0)
        progress.set_postfix(loss=f"{loss_value:.6f}", avg=f"{sum(recent_losses)/len(recent_losses):.6f}")

        if run is not None and step % log_every == 0:
            run.log({"step": step, "train_loss": sum(recent_losses) / len(recent_losses)})

        if step % validate_every == 0 or step == max_steps:
            eval_model = ema.ema_model
            val_metrics = validate(
                eval_model,
                val_loader,
                device,
                sample_cfg=cfg["validation"]["sampler"],
                max_batches=cfg["validation"].get("max_batches"),
                compute_fid=(step % int(cfg["validation"].get("fid_every_steps", validate_every)) == 0),
            )
            val_metrics["step"] = step
            print(val_metrics)
            if run is not None:
                run.log(val_metrics)

            metric_value = float(val_metrics.get(best_metric_name, val_metrics["val_mae"]))
            if metric_value < best_metric:
                best_metric = metric_value
                save_checkpoint(checkpoint_dir / "best_ema.pt", model, optimizer, ema, step, cfg, best_metric=best_metric)

            preview_batch = next(iter(val_loader))
            preview_condition = preview_batch["condition"].to(device)
            preview_pred = run_sampler(eval_model, preview_condition, cfg["validation"]["sampler"])
            preview_name = str(preview_batch["name"][0])
            save_tensor_npy(preview_pred[0], sample_dir / f"step_{step:06d}_{preview_name}_pred.npy")
            for channel_idx in range(preview_pred.shape[1]):
                save_tensor_png(preview_pred[0, channel_idx : channel_idx + 1], sample_dir / f"step_{step:06d}_{preview_name}_ch{channel_idx}.png")

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.pt", model, optimizer, ema, step, cfg, best_metric=best_metric)
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, ema, step, cfg, best_metric=best_metric)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
