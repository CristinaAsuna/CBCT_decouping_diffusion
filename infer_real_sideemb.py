from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, load_config
from .infer_research import build_dataset, build_model
from .preprocess_real_ceph import (
    align_real_ceph,
    collect_inputs,
    load_grayscale,
    parse_output_range,
    parse_output_size,
)
from .train_research import run_sampler
from .utils import pick_device, save_tensor_npy, save_tensor_png


def save_scaled_npy_png(array_scaled: np.ndarray, output_dir: Path, stem: str, suffix: str) -> None:
    np.save(output_dir / f"{stem}_{suffix}.npy", array_scaled.astype(np.float32))
    array_01 = array_scaled
    min_val = float(array_01.min())
    max_val = float(array_01.max())
    if max_val - min_val > 1e-8:
        array_01 = (array_01 - min_val) / (max_val - min_val)
    else:
        array_01 = np.zeros_like(array_01, dtype=np.float32)
    save_tensor_png(torch.from_numpy(array_01[None, ...] * 2.0 - 1.0), output_dir / f"{stem}_{suffix}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess real ceph images and infer left/right with side_emb.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Single image/.npy file or a directory containing real ceph inputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    parser.add_argument("--split-for-shape", default="train", choices=["train", "val", "test"])
    parser.add_argument("--face-direction", choices=["left", "right"], default="right")
    parser.add_argument("--posterior-trim-ratio", type=float, default=0.18)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--threshold-floor", type=float, default=0.10)
    parser.add_argument("--input-percentiles", nargs=2, type=float, default=[1.0, 99.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--output-percentiles", nargs=2, type=float, default=[1.0, 99.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--invert", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"][args.split_for_shape]
    if dataset_cfg.get("target_mode") != "side_emb":
        raise ValueError("infer_real_sideemb requires a config whose dataset target_mode is 'side_emb'.")

    device = pick_device(cfg.get("device"))
    dataset = build_dataset(dataset_cfg)
    model = build_model(cfg, dataset, device)
    state = torch.load(args.checkpoint, map_location=device)
    if args.weights == "ema" and "ema_model" in state:
        model.load_state_dict(state["ema_model"])
    else:
        model.load_state_dict(state["model"])
    model.eval()

    output_dir = ensure_dir(args.output_dir)
    preprocess_dir = ensure_dir(output_dir / "preprocessed")
    pred_dir = ensure_dir(output_dir / "predictions")
    sampler_cfg = cfg.get("validation", {}).get("sampler", {"sampler": "ddim", "steps": 50, "eta": 0.0})
    output_size = parse_output_size(cfg, None)
    output_range = parse_output_range(cfg, None)

    side_labels = list(dataset_cfg.get("side_labels") or ["left", "right"])
    side_to_idx = {label: idx for idx, label in enumerate(side_labels)}
    required_labels = {"left", "right"}
    if not required_labels.issubset(side_to_idx):
        raise ValueError(f"side_labels must contain both left and right, got {side_labels}")

    inputs = collect_inputs(Path(args.input))
    print(
        {
            "num_inputs": len(inputs),
            "output_size": output_size,
            "output_range": output_range,
            "side_labels": side_labels,
            "sampler": sampler_cfg,
        }
    )

    with torch.no_grad():
        for input_path in inputs:
            raw = load_grayscale(input_path)
            aligned_scaled, aligned_01, display_01, bbox = align_real_ceph(
                raw=raw,
                output_size=output_size,
                face_direction=args.face_direction,
                posterior_trim_ratio=args.posterior_trim_ratio,
                bbox_pad_ratio=args.bbox_pad_ratio,
                threshold_floor=args.threshold_floor,
                input_low_pct=float(args.input_percentiles[0]),
                input_high_pct=float(args.input_percentiles[1]),
                output_low_pct=float(args.output_percentiles[0]),
                output_high_pct=float(args.output_percentiles[1]),
                output_range=output_range,
                invert_mode=args.invert,
            )

            stem = input_path.stem
            save_scaled_npy_png(aligned_scaled, preprocess_dir, stem, "aligned")
            if args.save_debug:
                from .preprocess_real_ceph import make_debug_preview

                make_debug_preview(display_01, bbox, preprocess_dir / f"{stem}_debug_bbox.png")

            condition_m11 = (aligned_scaled - output_range[0]) / max(output_range[1] - output_range[0], 1e-8)
            condition_m11 = np.clip(condition_m11, 0.0, 1.0) * 2.0 - 1.0
            condition = torch.from_numpy(condition_m11[None, None, ...]).float().to(device)

            for side_label in ("left", "right"):
                side_id = torch.tensor([side_to_idx[side_label]], dtype=torch.long, device=device)
                pred = run_sampler(model, condition, sampler_cfg, side_ids=side_id)[0]
                save_tensor_npy(pred, pred_dir / f"{stem}_{side_label}.npy")
                for channel_idx in range(pred.shape[0]):
                    save_tensor_png(
                        pred[channel_idx : channel_idx + 1],
                        pred_dir / f"{stem}_{side_label}_ch{channel_idx}.png",
                    )


if __name__ == "__main__":
    main()
