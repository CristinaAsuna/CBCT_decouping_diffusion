from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import ensure_dir, load_config


def load_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[-1] == 1:
                array = array[..., 0]
            else:
                raise ValueError(f"Expected single-channel .npy, got shape {array.shape}")
        if array.ndim != 2:
            raise ValueError(f"Expected 2D grayscale input, got shape {array.shape}")
        return array
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def robust_minmax(array: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    low = float(np.percentile(array, low_pct))
    high = float(np.percentile(array, high_pct))
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def maybe_invert(array_01: np.ndarray, mode: str) -> np.ndarray:
    if mode == "never":
        return array_01
    if mode == "always":
        return 1.0 - array_01
    h, w = array_01.shape
    border = np.concatenate(
        [
            array_01[: max(1, h // 12), :].ravel(),
            array_01[-max(1, h // 12) :, :].ravel(),
            array_01[:, : max(1, w // 12)].ravel(),
            array_01[:, -max(1, w // 12) :].ravel(),
        ]
    )
    center = array_01[h // 4 : h - h // 4, w // 4 : w - w // 4]
    if center.size == 0:
        return array_01
    return 1.0 - array_01 if float(border.mean()) > float(center.mean()) else array_01


def otsu_threshold(array_01: np.ndarray) -> float:
    hist, _ = np.histogram(array_01, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.25
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return float(np.argmax(sigma_b) / 255.0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = image_shape
    bw = x1 - x0
    bh = y1 - y0
    pad_x = max(1, int(round(bw * pad_ratio)))
    pad_y = max(1, int(round(bh * pad_ratio)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(w, x1 + pad_x),
        min(h, y1 + pad_y),
    )


def trim_posterior(
    bbox: tuple[int, int, int, int],
    face_direction: str,
    posterior_trim_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    trim = int(round(width * posterior_trim_ratio))
    if trim <= 0:
        return bbox
    if face_direction == "right":
        x0 = min(x1 - 1, x0 + trim)
    elif face_direction == "left":
        x1 = max(x0 + 1, x1 - trim)
    else:
        raise ValueError(f"Unsupported face_direction: {face_direction}")
    return x0, y0, x1, y1


def crop_and_resize(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    cropped = image[y0:y1, x0:x1]
    pil = Image.fromarray((np.clip(cropped, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    resized = pil.resize(output_size[::-1], resample=Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


def scale_to_training_range(
    image_01: np.ndarray,
    out_min: float,
    out_max: float,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    stretched = robust_minmax(image_01, low_pct=low_pct, high_pct=high_pct)
    return (stretched * (out_max - out_min) + out_min).astype(np.float32)


def make_debug_preview(
    original_01: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    preview = Image.fromarray((np.clip(original_01, 0.0, 1.0) * 255.0).round().astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(preview)
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 64, 64), width=2)
    preview.save(output_path)


def align_real_ceph(
    raw: np.ndarray,
    output_size: tuple[int, int],
    face_direction: str,
    posterior_trim_ratio: float,
    bbox_pad_ratio: float,
    threshold_floor: float,
    input_low_pct: float,
    input_high_pct: float,
    output_low_pct: float,
    output_high_pct: float,
    output_range: tuple[float, float],
    invert_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    display_01 = robust_minmax(raw, low_pct=input_low_pct, high_pct=input_high_pct)
    display_01 = maybe_invert(display_01, invert_mode)

    threshold = max(otsu_threshold(display_01), threshold_floor)
    mask = display_01 > threshold
    bbox = bbox_from_mask(mask)
    if bbox is None:
        h, w = display_01.shape
        bbox = (0, 0, w, h)

    bbox = pad_bbox(bbox, display_01.shape, bbox_pad_ratio)
    bbox = trim_posterior(bbox, face_direction=face_direction, posterior_trim_ratio=posterior_trim_ratio)

    aligned_01 = crop_and_resize(display_01, bbox, output_size=output_size)
    aligned_scaled = scale_to_training_range(
        aligned_01,
        out_min=output_range[0],
        out_max=output_range[1],
        low_pct=output_low_pct,
        high_pct=output_high_pct,
    )
    return aligned_scaled, aligned_01, display_01, bbox


def preprocess_single(
    input_path: Path,
    output_dir: Path,
    output_size: tuple[int, int],
    face_direction: str,
    posterior_trim_ratio: float,
    bbox_pad_ratio: float,
    threshold_floor: float,
    input_low_pct: float,
    input_high_pct: float,
    output_low_pct: float,
    output_high_pct: float,
    output_range: tuple[float, float],
    invert_mode: str,
    save_debug: bool,
) -> None:
    raw = load_grayscale(input_path)
    aligned_scaled, aligned_01, display_01, bbox = align_real_ceph(
        raw=raw,
        output_size=output_size,
        face_direction=face_direction,
        posterior_trim_ratio=posterior_trim_ratio,
        bbox_pad_ratio=bbox_pad_ratio,
        threshold_floor=threshold_floor,
        input_low_pct=input_low_pct,
        input_high_pct=input_high_pct,
        output_low_pct=output_low_pct,
        output_high_pct=output_high_pct,
        output_range=output_range,
        invert_mode=invert_mode,
    )

    stem = input_path.stem
    np.save(output_dir / f"{stem}_aligned.npy", aligned_scaled.astype(np.float32))
    Image.fromarray((np.clip(aligned_01, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(output_dir / f"{stem}_aligned.png")
    if save_debug:
        make_debug_preview(display_01, bbox, output_dir / f"{stem}_debug_bbox.png")


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
    return [path for path in sorted(input_path.iterdir()) if path.is_file() and path.suffix.lower() in exts]


def parse_output_size(config: dict | None, cli_size: list[int] | None) -> tuple[int, int]:
    if cli_size is not None:
        return int(cli_size[0]), int(cli_size[1])
    if config is not None:
        size = config.get("dataset", {}).get("train", {}).get("image_size") or config.get("model", {}).get("image_size")
        if isinstance(size, list) and len(size) == 2:
            return int(size[0]), int(size[1])
        if isinstance(size, int):
            return int(size), int(size)
    return 256, 256


def parse_output_range(config: dict | None, cli_range: list[float] | None) -> tuple[float, float]:
    if cli_range is not None:
        return float(cli_range[0]), float(cli_range[1])
    if config is not None:
        value_range = config.get("dataset", {}).get("train", {}).get("value_range")
        if isinstance(value_range, list) and len(value_range) == 2:
            return float(value_range[0]), float(value_range[1])
    return 0.0, 6.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess real lateral ceph images to better match training full.npy inputs.")
    parser.add_argument("--input", required=True, help="Single image/.npy file or a directory containing inputs.")
    parser.add_argument("--output-dir", required=True, help="Directory to save aligned .npy/.png outputs.")
    parser.add_argument("--config", default=None, help="Optional training config used to infer output size/value range.")
    parser.add_argument("--output-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--output-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--face-direction", choices=["left", "right"], default="right")
    parser.add_argument("--posterior-trim-ratio", type=float, default=0.18)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--threshold-floor", type=float, default=0.10)
    parser.add_argument("--input-percentiles", nargs=2, type=float, default=[1.0, 99.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--output-percentiles", nargs=2, type=float, default=[1.0, 99.0], metavar=("LOW", "HIGH"))
    parser.add_argument("--invert", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    output_dir = ensure_dir(args.output_dir)
    output_size = parse_output_size(cfg, args.output_size)
    output_range = parse_output_range(cfg, args.output_range)

    inputs = collect_inputs(Path(args.input))
    if not inputs:
        raise ValueError("No valid input files were found.")

    print(
        {
            "num_inputs": len(inputs),
            "output_size": output_size,
            "output_range": output_range,
            "face_direction": args.face_direction,
            "posterior_trim_ratio": args.posterior_trim_ratio,
        }
    )

    for input_path in inputs:
        preprocess_single(
            input_path=input_path,
            output_dir=output_dir,
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
            save_debug=args.save_debug,
        )


if __name__ == "__main__":
    main()
