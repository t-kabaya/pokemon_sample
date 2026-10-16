"""Freeze Policy/Value -> fresh C++ rollout -> GPU PPO -> export -> native validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json

from agents.mega_lucario.ppo.scripts.collect import (
    CONTRACT_ID,
    DECK_COUNTS,
    DEFAULTS,
    NAMES,
    ROOT,
    collect,
    load_checkpoint,
)
from agents.mega_lucario.ppo.scripts.initialize import initialize
from agents.mega_lucario.ppo.scripts.train import add_training_args, train, training_config
from agents.mega_lucario.transformer.scripts.export import export
from agents.mega_lucario.transformer.scripts.validate import validate


def main() -> None:
    """Parse command-line arguments and run this tool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--gbdt-model",
        type=Path,
        required=True,
        help="Fixed opponent and post-update validation baseline",
    )
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--start-cycle", type=int, default=0)
    p.add_argument("--games-per-opponent", type=int)
    p.add_argument("--games", type=int, default=DEFAULTS["games_per_cycle"])
    p.add_argument("--opponent-config", type=Path)
    p.add_argument("--opponents", nargs="+", default=DEFAULTS["opponents"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--extract-workers", type=int, default=1)
    p.add_argument("--rollout-chunk-games", type=int, default=64)
    p.add_argument("--validation-workers", type=int, default=1)
    p.add_argument("--validation-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=202640000)
    p.add_argument("--sample-seed", type=int, default=202650000)
    p.add_argument("--update-seed", type=int, default=204)
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    add_training_args(p)
    a = p.parse_args()
    if a.cycles < 1:
        p.error("cycles must be positive")
    if a.start_cycle < 0:
        p.error("start-cycle must be nonnegative")
    if a.start_cycle == 0 and a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("output-dir must be empty")
    for cycle in range(a.start_cycle, a.start_cycle + a.cycles):
        output = a.output_dir / f"cycle-{cycle:03d}"
        if output.exists() and any(output.iterdir()):
            p.error(f"{output} must be empty")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    initial = a.output_dir / (
        "initial.pt" if a.start_cycle == 0 else f"initial-{a.start_cycle:03d}.pt"
    )
    current = initialize(a.checkpoint, initial)
    cycles_path = a.output_dir / "cycles.json"
    reports = (
        json.loads(cycles_path.read_text()) if a.start_cycle > 0 and cycles_path.exists() else []
    )
    games = len(a.opponents) * a.games_per_opponent if a.games_per_opponent is not None else a.games
    roster = json.loads(a.opponent_config.read_text()) if a.opponent_config else {}
    for cycle in range(a.start_cycle, a.start_cycle + a.cycles):
        output = a.output_dir / f"cycle-{cycle:03d}"
        cycle_number = cycle - a.start_cycle + 1
        print(f"PPO cycle {cycle_number}/{a.cycles}: rollout collection start", flush=True)
        collection = collect(
            current,
            output / "rollout",
            a.binary,
            a.gbdt_model,
            a.opponents,
            a.games_per_opponent,
            a.seed + cycle * games,
            a.sample_seed + cycle * games * 2,
            a.temperature,
            a.workers,
            games=a.games,
            extract_workers=a.extract_workers,
            rollout_chunk_games=a.rollout_chunk_games,
            audit_threads=a.audit_threads,
            opponent_weights=roster.get("weights"),
            opponent_models=roster.get("models"),
        )
        print(f"PPO cycle {cycle_number}/{a.cycles}: training start", flush=True)
        model, payload = load_checkpoint(current, CONTRACT_ID, DECK_COUNTS, NAMES)
        training = train(
            model,
            payload,
            current,
            output / "rollout",
            output / "models",
            training_config(a, a.update_seed + cycle),
        )
        current = output / "models/model.pt"
        print(f"PPO cycle {cycle_number}/{a.cycles}: export start", flush=True)
        export(current, output / "models/model.bin")
        print(f"PPO cycle {cycle_number}/{a.cycles}: validation start", flush=True)
        validation = validate(
            current,
            output / "models/model.bin",
            a.gbdt_model,
            output / "validation",
            a.binary,
            workers=a.validation_workers,
            threads=a.validation_threads,
        )
        report = dict(
            cycle=cycle,
            games=collection["collection"]["total_games"],
            episodes=len(collection["episodes"]),
            decisions=collection["decisions"],
            training=training,
            validation=validation,
        )
        reports.append(report)
        print(f"PPO cycle {cycle} complete: {current}", flush=True)
    (a.output_dir / "cycles.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
