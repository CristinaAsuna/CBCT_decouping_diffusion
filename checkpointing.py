from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_file(path: str | Path, device: torch.device) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        return torch.load(path, map_location=device)


def _is_raw_state_dict(state: Any) -> bool:
    if not isinstance(state, dict) or not state:
        return False
    return all(isinstance(key, str) for key in state.keys())


def require_training_checkpoint(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(
            "Expected a full training checkpoint with at least a 'model' entry. "
            "Use a checkpoint produced by train_research.py such as last.pt, step_xxxxxx.pt, or best_ema.pt."
        )
    return state


def select_model_state(state: Any, weights: str = "ema") -> tuple[dict[str, Any], str]:
    if isinstance(state, dict) and "model" in state:
        if weights == "ema" and "ema_model" in state:
            return state["ema_model"], "ema_model"
        return state["model"], "model"
    if _is_raw_state_dict(state):
        return state, "state_dict"
    raise ValueError("Checkpoint does not contain loadable model weights.")
