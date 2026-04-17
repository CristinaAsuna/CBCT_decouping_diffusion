from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ensure_dir, load_config
from .dataset import CaseFolderNpyDataset, NpyConditionTargetDataset
from .diffusion import GaussianConditionalDiffusion
from .metrics import mae, mse, psnr, ssim
from .utils import pick_device, save_tensor_npy, save_tensor_png, set_seed


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


def build_loader(dataset_cfg: dict, batch_size: int, shuffle: bool, num_workers: int):
    dataset = build_dataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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


def validate(model, loader, device, sample_steps: int | None) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    maes: list[float] = []
    mses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = model(target, condition)
            pred = model.sample(condition, sample_steps=sample_steps)
            losses.append(float(loss.item()))
            maes.append(mae(pred, target))
            mses.append(mse(pred, target))
            psnrs.append(psnr(pred, target))
            ssims.append(ssim(pred, target))
    return {
        "val_loss": sum(losses) / max(len(losses), 1),
        "val_mae": sum(maes) / max(len(maes), 1),
        "val_mse": sum(mses) / max(len(mses), 1),
        "val_psnr": sum(psnrs) / max(len(psnrs), 1),
        "val_ssim": sum(ssims) / max(len(ssims), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train palette_decoupling baseline.")
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
    val_loader = None
    if "val" in cfg["dataset"]:
        _, val_loader = build_loader(
            cfg["dataset"]["val"],
            batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=cfg["train"].get("num_workers", 0),
        )

    print(
        {
            "device": str(device),
            "train_samples": len(train_dataset),
            "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
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
    run = maybe_init_swanlab(cfg.get("logging", {}))
    best_val = float("inf")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_losses: list[float] = []
        train_bar = tqdm(train_loader, desc=f"train epoch {epoch}", leave=True)
        for batch in train_bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(target, condition)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            train_bar.set_postfix(loss=f"{loss_value:.6f}")

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        log_data = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None and epoch % cfg["train"].get("val_every", 1) == 0:
            val_sample_steps = cfg["train"].get("val_sample_steps", cfg["diffusion"].get("sample_steps"))
            metrics = validate(model, val_loader, device, val_sample_steps)
            log_data.update(metrics)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "epoch": epoch}, checkpoint_dir / "best.pt")

            preview_batch = next(iter(val_loader))
            preview_condition = preview_batch["condition"].to(device)
            preview_pred = model.sample(preview_condition, sample_steps=val_sample_steps)
            preview_name = str(preview_batch["name"][0])
            save_tensor_npy(preview_pred[0], sample_dir / f"epoch_{epoch:04d}_{preview_name}_pred.npy")
            for channel_idx in range(preview_pred.shape[1]):
                save_tensor_png(preview_pred[0, channel_idx : channel_idx + 1], sample_dir / f"epoch_{epoch:04d}_{preview_name}_ch{channel_idx}.png")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "epoch": epoch}, checkpoint_dir / "last.pt")
        print(log_data)
        if run is not None:
            run.log(log_data)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
