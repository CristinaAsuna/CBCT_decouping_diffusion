from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize case-folder .npy projection files as model-facing images.")
    parser.add_argument("--case-root", required=True, help="Root directory containing Bone_* case folders.")
    parser.add_argument("--out-dir", default="tiny_ablation/npy_visualization", help="Directory for PNG sheets and index.html.")
    parser.add_argument("--case-glob", default="Bone_*", help="Case folder glob pattern.")
    parser.add_argument("--file-glob", default="*.npy", help="NPY file glob pattern inside each case.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit for quick inspection.")
    parser.add_argument(
        "--mode",
        choices=["model_m11", "fixed_range", "percentile", "minmax"],
        default="model_m11",
        help="Visualization scaling. model_m11 matches fixed_range_m11 -> TensorBoard image mapping.",
    )
    parser.add_argument("--value-range", type=float, nargs=2, default=(0.0, 6.0), metavar=("MIN", "MAX"))
    parser.add_argument("--percentile-range", type=float, nargs=2, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--image-size", type=int, default=256, help="Thumbnail size in pixels.")
    parser.add_argument("--columns", type=int, default=3, help="Columns in each case contact sheet.")
    return parser.parse_args()


def as_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and 1 in array.shape:
        return np.squeeze(array).astype(np.float32)
    raise ValueError(f"Expected a 2D array or singleton-channel array, got shape={array.shape}")


def array_stats(array: np.ndarray) -> dict[str, Any]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"shape": list(array.shape), "finite": 0}
    return {
        "shape": list(array.shape),
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
    }


def scale_array(array: np.ndarray, mode: str, value_range: tuple[float, float], percentile_range: tuple[float, float]) -> np.ndarray:
    if mode in {"model_m11", "fixed_range"}:
        lo, hi = value_range
    elif mode == "percentile":
        lo, hi = np.percentile(array[np.isfinite(array)], percentile_range)
    elif mode == "minmax":
        lo, hi = float(np.nanmin(array)), float(np.nanmax(array))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    clipped = np.clip(array, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return np.round(normalized * 255.0).astype(np.uint8)


def save_png(array: np.ndarray, path: Path, mode: str, value_range: tuple[float, float], percentile_range: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(scale_array(array, mode, value_range, percentile_range), mode="L")
    image.save(path)


def load_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheet(entries: list[dict[str, Any]], path: Path, image_size: int, columns: int) -> None:
    if not entries:
        return
    columns = max(1, columns)
    label_h = 72
    pad = 12
    rows = (len(entries) + columns - 1) // columns
    width = columns * image_size + (columns + 1) * pad
    height = rows * (image_size + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(13)
    for idx, entry in enumerate(entries):
        row = idx // columns
        col = idx % columns
        x = pad + col * (image_size + pad)
        y = pad + row * (image_size + label_h + pad)
        image = Image.open(entry["png_path"]).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        sheet.paste(Image.merge("RGB", (image, image, image)), (x, y))
        stats = entry["stats"]
        lines = [
            entry["name"],
            f"shape={tuple(stats.get('shape', []))}",
            f"min={stats.get('min', 0):.4g} max={stats.get('max', 0):.4g}",
            f"mean={stats.get('mean', 0):.4g} std={stats.get('std', 0):.4g}",
        ]
        for line_idx, line in enumerate(lines):
            draw.text((x, y + image_size + 4 + line_idx * 16), line, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def physics_summary(case_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ("std", "aug"):
        full_path = case_dir / f"{variant}_full.npy"
        left_path = case_dir / f"{variant}_left.npy"
        right_path = case_dir / f"{variant}_right.npy"
        if not (full_path.exists() and left_path.exists() and right_path.exists()):
            continue
        full = as_2d(np.load(full_path))
        left = as_2d(np.load(left_path))
        right = as_2d(np.load(right_path))
        residual = left + right - full
        denom = float(np.mean(np.abs(full)) + 1e-8)
        corr = float(np.corrcoef(full.reshape(-1), (left + right).reshape(-1))[0, 1])
        result[variant] = {
            "mae": float(np.mean(np.abs(residual))),
            "nmae": float(np.mean(np.abs(residual)) / denom),
            "max_abs": float(np.max(np.abs(residual))),
            "corr": corr,
        }
    return result


def case_sort_key(path: Path) -> tuple[str, int | str]:
    suffix = path.name.split("_")[-1]
    try:
        return (path.name.rsplit("_", 1)[0], int(suffix))
    except ValueError:
        return (path.name, path.name)


def render_html(out_dir: Path, cases: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows = []
    for case in cases:
        physics = html.escape(json.dumps(case["physics"], ensure_ascii=False, indent=2))
        image_rel = html.escape(Path(case["sheet_path"]).relative_to(out_dir).as_posix())
        rows.append(
            f"""
            <section class="case">
              <h2>{html.escape(case['name'])}</h2>
              <p><a href="{image_rel}" target="_blank">Open contact sheet</a></p>
              <img src="{image_rel}" alt="{html.escape(case['name'])} contact sheet">
              <details>
                <summary>Stats and physics residual</summary>
                <pre>{physics}</pre>
              </details>
            </section>
            """
        )
    body = "\n".join(rows)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NPY Case Visualization</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f6f2; color: #1d1d1b; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #555; margin-bottom: 24px; }}
    .case {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 24px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    pre {{ white-space: pre-wrap; background: #111; color: #f4f4f4; padding: 12px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>NPY Case Visualization</h1>
  <div class="meta">
    case_root={html.escape(str(args.case_root))}<br>
    mode={html.escape(args.mode)}, value_range={tuple(args.value_range)}
  </div>
  {body}
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted([path for path in case_root.glob(args.case_glob) if path.is_dir()], key=case_sort_key)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise FileNotFoundError(f"No case folders matched {case_root / args.case_glob}")

    all_cases: list[dict[str, Any]] = []
    value_range = (float(args.value_range[0]), float(args.value_range[1]))
    percentile_range = (float(args.percentile_range[0]), float(args.percentile_range[1]))
    for case_dir in cases:
        case_out = out_dir / case_dir.name
        entries = []
        for npy_path in sorted(case_dir.glob(args.file_glob)):
            array = as_2d(np.load(npy_path))
            png_path = case_out / f"{npy_path.stem}.png"
            save_png(array, png_path, args.mode, value_range, percentile_range)
            entries.append(
                {
                    "name": npy_path.name,
                    "png_path": png_path,
                    "stats": array_stats(array),
                }
            )
        sheet_path = case_out / f"{case_dir.name}_sheet.png"
        make_contact_sheet(entries, sheet_path, image_size=int(args.image_size), columns=int(args.columns))
        summary = {
            "name": case_dir.name,
            "sheet_path": str(sheet_path),
            "entries": [
                {
                    "name": item["name"],
                    "png": item["png_path"].relative_to(out_dir).as_posix(),
                    "stats": item["stats"],
                }
                for item in entries
            ],
            "physics": physics_summary(case_dir),
        }
        all_cases.append(summary)

    (out_dir / "summary.json").write_text(json.dumps(all_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    render_html(out_dir, all_cases, args)
    print({"num_cases": len(all_cases), "out_dir": str(out_dir), "index": str(out_dir / "index.html")})


if __name__ == "__main__":
    main()
