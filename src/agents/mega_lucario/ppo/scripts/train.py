"""One epoch of clipped PPO from the exact behavior checkpoint and complete rollouts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse

from agents.mega_lucario.ppo.scripts.collect import (
    CONTRACT_ID,
    DECK_COUNTS,
    DEFAULTS,
    NAMES,
    ROOT,
    load_checkpoint,
)
from agents.mega_lucario.ppo.train import train

if TYPE_CHECKING:
    from typing import Any


def add_training_args(p: argparse.ArgumentParser) -> None:
    """Register PPO optimization and device options on the command parser."""
    for name in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip",
        "value_clip",
        "value_coefficient",
        "entropy_coefficient",
        "weight_decay",
        "max_grad_norm",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name])
    for name in ("batch_size", "updates", "holdout_decisions", "micro_batch_size"):
        p.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    p.add_argument("--audit-threads", type=int, default=1)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")


def training_config(a: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Merge parsed optimization options with the fixed rollout training contract."""
    result = {
        k: getattr(a, k)
        for k in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip",
            "value_clip",
            "value_coefficient",
            "entropy_coefficient",
            "weight_decay",
            "max_grad_norm",
            "batch_size",
            "updates",
            "holdout_decisions",
            "micro_batch_size",
            "device",
            "audit_threads",
        )
    }
    result.update(
        seed=seed,
        epochs=1,
        advantage_normalization="training decisions only, population std",
        value_target="GAE return",
        value_loss="51-bin interpolated cross entropy + clipped expectation MSE",
        entropy="sum along sampled prefix path",
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "rollout-dir", "output-dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--seed", type=int, default=204)
    add_training_args(p)
    a = p.parse_args()
    model, payload = load_checkpoint(a.checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    train(model, payload, a.checkpoint, a.rollout_dir, a.output_dir, training_config(a, a.seed))
