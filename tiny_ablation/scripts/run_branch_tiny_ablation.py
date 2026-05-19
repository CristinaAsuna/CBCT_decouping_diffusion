from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run branch-only tiny ablations sequentially on one GPU.")
    parser.add_argument("--case-root", default=None, help="Override dataset.train/val.case_root.")
    parser.add_argument("--case-names-file", default="tiny_ablation/case_splits/tiny_cases.txt")
    parser.add_argument("--output-root", default="tiny_ablation/outputs")
    parser.add_argument("--run-tag", default=None, help="Optional subdirectory under output-root for this whole ablation run.")
    parser.add_argument("--generated-config-dir", default="tiny_ablation/generated_branch_configs")
    parser.add_argument("--configs", nargs="*", default=None, help="Specific branch config files to run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override train.max_steps for all runs.")
    parser.add_argument("--sampler-steps", type=int, default=None, help="Override validation.sampler.steps for all runs.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard in generated configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def format_path_template(value: Any, output_root: Path) -> Any:
    if not isinstance(value, str):
        return value
    return value.format(output_root=str(output_root).replace("\\", "/"))


def update_config(cfg: dict[str, Any], cfg_path: Path, args: argparse.Namespace, root: Path) -> dict[str, Any]:
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
        if cfg["train"].get("resume_checkpoint"):
            cfg["train"]["finetune_steps"] = int(args.max_steps)
    if args.sampler_steps is not None:
        cfg["validation"]["sampler"]["steps"] = int(args.sampler_steps)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    if args.run_tag:
        output_root = output_root / args.run_tag
    run_output = output_root / cfg_path.stem
    cfg["output"]["root"] = str(run_output)

    resume_checkpoint = cfg["train"].get("resume_checkpoint")
    cfg["train"]["resume_checkpoint"] = format_path_template(resume_checkpoint, output_root)

    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg["use_swanlab"] = False
    logging_cfg["use_tensorboard"] = not args.no_tensorboard
    logging_cfg["tensorboard_logdir"] = str(run_output / "tensorboard")
    return cfg


def main() -> None:
    args = parse_args()
    root = repo_root()
    if args.configs:
        config_paths = [Path(item) for item in args.configs]
    else:
        config_paths = sorted((root / "tiny_ablation" / "branch_configs").glob("*.yaml"))

    generated_dir = Path(args.generated_config_dir)
    if not generated_dir.is_absolute():
        generated_dir = root / generated_dir
    if args.run_tag:
        generated_dir = generated_dir / args.run_tag

    for cfg_path in config_paths:
        if not cfg_path.is_absolute():
            cfg_path = root / cfg_path
        cfg = update_config(load_yaml(cfg_path), cfg_path, args, root)
        generated_path = generated_dir / cfg_path.name
        write_yaml(generated_path, cfg)
        command = [sys.executable, "-m", "tiny_ablation.train_branch_tiny", "--config", str(generated_path)]
        print({"config": str(generated_path), "command": " ".join(command)})
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
