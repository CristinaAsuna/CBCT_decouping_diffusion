from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny ablation configs sequentially on one GPU.")
    parser.add_argument("--case-root", default=None, help="Override dataset.train/val.case_root.")
    parser.add_argument("--case-names-file", default="tiny_ablation/case_splits/tiny_cases.txt")
    parser.add_argument("--output-root", default="tiny_ablation/outputs")
    parser.add_argument("--generated-config-dir", default="tiny_ablation/generated_configs")
    parser.add_argument("--configs", nargs="*", default=None, help="Specific config files to run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps.")
    parser.add_argument("--sampler-steps", type=int, default=None, help="Override validation.sampler.steps.")
    parser.add_argument("--enable-swanlab", action="store_true", help="Keep SwanLab enabled in generated configs.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard in generated configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def update_config(cfg: dict, cfg_path: Path, args: argparse.Namespace, root: Path) -> dict:
    cfg = dict(cfg)
    case_names = Path(args.case_names_file)
    if not case_names.is_absolute():
        case_names = root / case_names

    for split in ("train", "val"):
        dataset_cfg = cfg["dataset"][split]
        if args.case_root:
            dataset_cfg["case_root"] = args.case_root
        dataset_cfg["case_names_file"] = str(case_names)
        dataset_cfg["split"] = None

    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.sampler_steps is not None:
        cfg["validation"]["sampler"]["steps"] = int(args.sampler_steps)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    run_output = output_root / cfg_path.stem
    cfg["output"]["root"] = str(run_output)

    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg["use_swanlab"] = bool(args.enable_swanlab)
    logging_cfg["use_tensorboard"] = not args.no_tensorboard
    logging_cfg["tensorboard_logdir"] = str(run_output / "tensorboard")
    return cfg


def main() -> None:
    args = parse_args()
    root = repo_root()
    if args.configs:
        config_paths = [Path(item) for item in args.configs]
    else:
        config_paths = sorted((root / "tiny_ablation" / "configs").glob("*.yaml"))

    generated_dir = Path(args.generated_config_dir)
    if not generated_dir.is_absolute():
        generated_dir = root / generated_dir

    for cfg_path in config_paths:
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        cfg = update_config(load_yaml(cfg_path), cfg_path, args, root)
        generated_path = generated_dir / cfg_path.name
        write_yaml(generated_path, cfg)
        command = [sys.executable, str(root / "train_research.py"), "--config", str(generated_path)]
        print({"config": str(generated_path), "command": " ".join(command)})
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
