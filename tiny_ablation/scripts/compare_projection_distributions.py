from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare .npy projection value distributions across two case-folder datasets.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Dataset roots. Example: D:/nnunet/2D D:/nnunet/2d_projection_physics_consistent",
    )
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels matching --roots.")
    parser.add_argument("--out-dir", default="tiny_ablation/distribution_compare", help="Output directory for CSV/JSON summaries.")
    parser.add_argument("--case-glob", default="Bone_*")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
    parser.add_argument("--sample-pixels", type=int, default=200000, help="Reservoir size per dataset/file kind for global percentiles.")
    return parser.parse_args()


def npy_kind(path: Path) -> tuple[str, str]:
    # std_full.npy -> ("std", "full")
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return "unknown", stem
    return parts[0], parts[1]


def stats(values: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if values.size == 0:
        return {"count": 0}
    result: dict[str, float] = {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "sum": float(np.sum(values)),
    }
    for percentile in percentiles:
        key = f"p{str(percentile).replace('.', '_')}"
        result[key] = float(np.percentile(values, percentile))
    return result


def update_running(bucket: dict[str, Any], array: np.ndarray, sample_pixels: int, rng: np.random.Generator) -> None:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False).reshape(-1)
    if finite.size == 0:
        return
    bucket["count"] += int(finite.size)
    bucket["sum"] += float(np.sum(finite))
    bucket["sum_sq"] += float(np.sum(finite * finite))
    bucket["min"] = min(bucket["min"], float(np.min(finite)))
    bucket["max"] = max(bucket["max"], float(np.max(finite)))

    if finite.size > sample_pixels:
        finite = rng.choice(finite, size=sample_pixels, replace=False)
    sample = bucket["sample"]
    if sample.size == 0:
        bucket["sample"] = finite[:sample_pixels]
        return
    merged = np.concatenate([sample, finite])
    if merged.size > sample_pixels:
        merged = rng.choice(merged, size=sample_pixels, replace=False)
    bucket["sample"] = merged


def finalize_running(bucket: dict[str, Any], percentiles: list[float]) -> dict[str, float]:
    count = int(bucket["count"])
    if count == 0:
        return {"count": 0}
    mean = bucket["sum"] / count
    var = max(bucket["sum_sq"] / count - mean * mean, 0.0)
    result: dict[str, float] = {
        "count": count,
        "min": float(bucket["min"]),
        "max": float(bucket["max"]),
        "mean": float(mean),
        "std": float(np.sqrt(var)),
    }
    sample = bucket["sample"]
    for percentile in percentiles:
        key = f"p{str(percentile).replace('.', '_')}"
        result[key] = float(np.percentile(sample, percentile)) if sample.size else float("nan")
    return result


def empty_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
        "sample": np.array([], dtype=np.float64),
    }


def consistency_stats(case_dir: Path, prefix: str, percentiles: list[float]) -> dict[str, float] | None:
    full_path = case_dir / f"{prefix}_full.npy"
    left_path = case_dir / f"{prefix}_left.npy"
    right_path = case_dir / f"{prefix}_right.npy"
    if not (full_path.exists() and left_path.exists() and right_path.exists()):
        return None
    full = np.load(full_path).astype(np.float32)
    left = np.load(left_path).astype(np.float32)
    right = np.load(right_path).astype(np.float32)
    residual = left + right - full
    abs_residual = np.abs(residual)
    denom = float(np.mean(np.abs(full)) + 1e-8)
    result = stats(abs_residual.reshape(-1), percentiles)
    result.update(
        {
            "mae": float(np.mean(abs_residual)),
            "nmae": float(np.mean(abs_residual) / denom),
            "signed_mean": float(np.mean(residual)),
            "signed_std": float(np.std(residual)),
            "corr_full_sum": float(np.corrcoef(full.reshape(-1), (left + right).reshape(-1))[0, 1]),
            "full_mean": float(np.mean(full)),
            "left_mean": float(np.mean(left)),
            "right_mean": float(np.mean(right)),
            "sum_mean": float(np.mean(left + right)),
            "sum_over_full_mean_ratio": float(np.mean(left + right) / (np.mean(full) + 1e-8)),
        }
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(root: Path, label: str, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, Any]:
    case_dirs = sorted([path for path in root.glob(args.case_glob) if path.is_dir()])
    if args.max_cases is not None:
        case_dirs = case_dirs[: args.max_cases]
    running: dict[tuple[str, str], dict[str, Any]] = defaultdict(empty_bucket)
    per_file_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        for npy_path in sorted(case_dir.glob("*.npy")):
            prefix, target = npy_kind(npy_path)
            array = np.load(npy_path)
            file_stats = stats(array.reshape(-1), args.percentiles)
            row = {
                "dataset": label,
                "root": str(root),
                "case": case_dir.name,
                "file": npy_path.name,
                "prefix": prefix,
                "target": target,
                **file_stats,
            }
            per_file_rows.append(row)
            update_running(running[(prefix, target)], array, int(args.sample_pixels), rng)

        for prefix in ("std", "aug"):
            item = consistency_stats(case_dir, prefix, args.percentiles)
            if item is not None:
                consistency_rows.append(
                    {
                        "dataset": label,
                        "root": str(root),
                        "case": case_dir.name,
                        "prefix": prefix,
                        **item,
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for (prefix, target), bucket in sorted(running.items()):
        aggregate_rows.append(
            {
                "dataset": label,
                "root": str(root),
                "prefix": prefix,
                "target": target,
                **finalize_running(bucket, args.percentiles),
            }
        )

    return {
        "label": label,
        "root": str(root),
        "num_cases": len(case_dirs),
        "aggregate_rows": aggregate_rows,
        "per_file_rows": per_file_rows,
        "consistency_rows": consistency_rows,
    }


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in args.roots]
    if args.labels is None:
        labels = [root.name for root in roots]
    else:
        labels = args.labels
    if len(labels) != len(roots):
        raise ValueError("--labels must have the same length as --roots")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    summaries = [summarize_dataset(root, label, args, rng) for root, label in zip(roots, labels)]
    aggregate_rows = [row for summary in summaries for row in summary["aggregate_rows"]]
    per_file_rows = [row for summary in summaries for row in summary["per_file_rows"]]
    consistency_rows = [row for summary in summaries for row in summary["consistency_rows"]]

    write_csv(out_dir / "aggregate_distribution.csv", aggregate_rows)
    write_csv(out_dir / "per_file_distribution.csv", per_file_rows)
    write_csv(out_dir / "consistency_distribution.csv", consistency_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "label": summary["label"],
                        "root": summary["root"],
                        "num_cases": summary["num_cases"],
                    }
                    for summary in summaries
                ],
                "aggregate_distribution": aggregate_rows,
                "consistency_distribution": consistency_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== Aggregate Value Distribution =====")
    for row in aggregate_rows:
        print(
            f"{row['dataset']:>12s} {row['prefix']:>3s}_{row['target']:<5s} "
            f"mean={row['mean']:.6g} std={row['std']:.6g} "
            f"min={row['min']:.6g} p1={row.get('p1', float('nan')):.6g} "
            f"p50={row.get('p50', float('nan')):.6g} p99={row.get('p99', float('nan')):.6g} max={row['max']:.6g}"
        )

    print("\n===== Consistency full ~= left + right =====")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in consistency_rows:
        grouped[(row["dataset"], row["prefix"])].append(row)
    for (dataset, prefix), rows in sorted(grouped.items()):
        mae = np.array([row["mae"] for row in rows], dtype=np.float64)
        nmae = np.array([row["nmae"] for row in rows], dtype=np.float64)
        ratio = np.array([row["sum_over_full_mean_ratio"] for row in rows], dtype=np.float64)
        corr = np.array([row["corr_full_sum"] for row in rows], dtype=np.float64)
        print(
            f"{dataset:>12s} {prefix:>3s}: "
            f"MAE={mae.mean():.8g} NMAE={nmae.mean():.8g} "
            f"sum/full mean ratio={ratio.mean():.6g} corr={corr.mean():.8g} n={len(rows)}"
        )

    print({"out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
