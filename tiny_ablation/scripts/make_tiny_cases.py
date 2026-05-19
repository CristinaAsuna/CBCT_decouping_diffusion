from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny case-name list for ablation smoke tests.")
    parser.add_argument("--case-root", required=True, help="Directory containing case folders.")
    parser.add_argument("--out", default="tiny_ablation/case_splits/tiny_cases.txt", help="Output text file.")
    parser.add_argument("--num-cases", type=int, default=16, help="Number of case folders to sample.")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
    parser.add_argument("--variants", nargs="+", default=["std", "aug"], help="Variants to check.")
    parser.add_argument("--condition-template", default="{variant}_full.npy")
    parser.add_argument("--target-templates", nargs="+", default=["{variant}_left.npy", "{variant}_right.npy"])
    return parser.parse_args()


def case_has_required_files(
    case_dir: Path,
    variants: list[str],
    condition_template: str,
    target_templates: list[str],
) -> bool:
    for variant in variants:
        condition_path = case_dir / condition_template.format(variant=variant)
        target_paths = [case_dir / template.format(variant=variant) for template in target_templates]
        if condition_path.exists() and all(path.exists() for path in target_paths):
            return True
    return False


def main() -> None:
    args = parse_args()
    case_root = Path(args.case_root)
    if not case_root.is_dir():
        raise FileNotFoundError(f"case root not found: {case_root}")

    valid_cases = [
        path.name
        for path in sorted(case_root.iterdir())
        if path.is_dir()
        and case_has_required_files(
            path,
            args.variants,
            args.condition_template,
            args.target_templates,
        )
    ]
    if not valid_cases:
        raise RuntimeError(f"no valid case folders found under {case_root}")

    rng = random.Random(args.seed)
    rng.shuffle(valid_cases)
    selected_cases = sorted(valid_cases[: args.num_cases])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected_cases) + "\n", encoding="utf-8")
    print(
        {
            "case_root": str(case_root),
            "num_valid_cases": len(valid_cases),
            "num_selected_cases": len(selected_cases),
            "out": str(out_path),
        }
    )


if __name__ == "__main__":
    main()
