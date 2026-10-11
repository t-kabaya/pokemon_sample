"""Seeded C++ on-policy games, aligned transitions and behavior-statistics audit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from agents.mega_lucario.ppo.extract import extract
from agents.mega_lucario.ppo.rollout import audit_behavior
from agents.mega_lucario.ppo.scripts.initialize import (
    CONTRACT_ID,
    DECK_COUNTS,
    NAMES,
    ROOT,
    load_checkpoint,
)
from agents.mega_lucario.replay import sha256
from agents.mega_lucario.transformer.scripts.export import export

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


DEFAULTS = json.loads((ROOT / "src/agents/mega_lucario/ppo/config.json").read_text())


def allocate_games(
    opponents: Sequence[str], weights: dict[str, float], games: int
) -> dict[str, int]:
    """Allocate exact game count in seat-balanced pairs, including zero weights."""
    import math

    if games < 2 or games % 2:
        raise ValueError("Total games must be a positive even number")
    if (
        set(weights) != set(opponents)
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Finite nonnegative weights required for each opponent")
    exact = [games // 2 * weights[n] / sum(weights.values()) for n in opponents]
    pairs = [math.floor(v) for v in exact]
    for i in sorted(range(len(pairs)), key=lambda i: (-(exact[i] - pairs[i]), i))[
        : games // 2 - sum(pairs)
    ]:
        pairs[i] += 1
    return {n: 2 * p for n, p in zip(opponents, pairs)}


def collect(
    checkpoint: Path,
    output: Path,
    binary: Path,
    gbdt_model: Path | None = None,
    opponents: Sequence[str] | None = None,
    games_per_opponent: int | None = None,
    seed: int = 202640000,
    sample_seed: int = 202650000,
    temperature: float = 1.0,
    workers: int = 2,
    *,
    games: int = 5000,
    extract_workers: int = 1,
    opponent_weights: dict[str, float] | None = None,
    opponent_models: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze a policy, collect fresh sampled matches, and audit its behavior statistics."""
    if DEFAULTS["reward"] != {"win": 1.0, "draw": 0.0, "loss": -1.0, "intermediate": 0.0}:
        raise ValueError("Only terminal win/draw/loss rewards are supported")
    opponents = DEFAULTS["opponents"] if opponents is None else opponents
    if (
        not opponents
        or len(set(opponents)) != len(opponents)
        or set(opponents) - set(DEFAULTS["opponents"]) - set(opponent_models or {})
    ):
        raise ValueError("Unknown/duplicate opponent")
    if any(not isinstance(n, str) or not n or not n.replace("_", "").isalnum() for n in opponents):
        raise ValueError("Opponent aliases must be identifiers")
    if workers < 1:
        raise ValueError("Use positive workers")
    if extract_workers < 1:
        raise ValueError("Use positive extract workers")
    if games_per_opponent is not None and (games_per_opponent < 2 or games_per_opponent % 2):
        raise ValueError("Use a positive even game count per opponent")
    if not 0 < temperature < 100 or not 0 <= seed <= 2**32 - 1 or not 0 <= sample_seed <= 2**32 - 1:
        raise ValueError("Invalid temperature/seeds")
    weights = (
        {n: DEFAULTS["opponent_weights"].get(n, 1.0) for n in opponents}
        if opponent_weights is None
        else opponent_weights
    )
    if games_per_opponent is not None and opponent_weights is not None:
        raise ValueError("Use total --games with weighted opponents")
    allocation = (
        allocate_games(opponents, weights, games)
        if games_per_opponent is None
        else {n: games_per_opponent for n in opponents}
    )
    total = sum(allocation.values())
    opponent_models = {} if opponent_models is None else opponent_models
    if set(opponent_models) - set(opponents) or "self" in opponent_models:
        raise ValueError("Invalid fixed opponent models")
    if seed + total > 2**32 or sample_seed + 2 * total > 2**32:
        raise ValueError("Seed range overflow")
    if (
        allocation.get("mega_lucario_gbdt", 0) > 0
        and "mega_lucario_gbdt" not in opponent_models
        and gbdt_model is None
    ):
        raise ValueError("GBDT opponent requires --gbdt-model")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Rollout output must be empty")
    model, payload = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    if model.value_head is None:
        raise ValueError("Run ppo/scripts/initialize.py first")
    output.mkdir(parents=True, exist_ok=True)
    export(checkpoint, output / "behavior.bin")
    jobs = []
    offset = 0
    for opponent in opponents:
        for seat in (0, 1):
            count = allocation[opponent] // 2
            if count:
                jobs.append((opponent, seat, count, offset))
            offset += count
    print(
        f"PPO rollout collection start: {total} games across {len(jobs)} batches "
        f"with {workers} workers",
        flush=True,
    )

    def run(job: tuple[str, int, int, int]) -> list[dict[str, Any]]:
        """Run and validate one sampled opponent/seat match batch."""
        opponent, seat, count, offset = job
        rival = (
            "mega_lucario_transformer"
            if opponent == "self" or opponent in opponent_models
            else opponent
        )
        names = (
            ["mega_lucario_transformer", rival]
            if seat == 0
            else [rival, "mega_lucario_transformer"]
        )
        command = [
            str(binary.resolve()),
            "--games",
            str(count),
            "--seed",
            str(seed + offset),
            "--temperature",
            str(temperature),
            "--replay-dir",
            str(output / "replays" / f"{opponent}-seat{seat}"),
        ]
        for side, name in enumerate(names):
            letter = "a" if side == 0 else "b"
            deck = (
                "mega_lucario"
                if name in ("mega_lucario_transformer", "mega_lucario_gbdt")
                else name
            )
            command += [
                f"--agent-{letter}",
                name,
                f"--deck-{letter}",
                str(ROOT / f"src/agents/{deck}/deck.csv"),
            ]
            if side != seat and opponent in opponent_models:
                command += [f"--model-{letter}", str(Path(opponent_models[opponent]).resolve())]
            elif name == "mega_lucario_transformer":
                command += [
                    f"--model-{letter}",
                    str((output / "behavior.bin").resolve()),
                    f"--sample-seed-{letter}",
                    str(sample_seed + offset * 2 + side * count),
                ]
            elif name == "mega_lucario_gbdt":
                command += [f"--model-{letter}", str(gbdt_model.resolve())]
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=max(300, count * 10)
        )
        records = [json.loads(line) for line in result.stdout.splitlines()]
        if len(records) != count or any(r["agent_errors"] or r["engine_errors"] for r in records):
            raise ValueError("PPO game errors")
        print(f"Collected {opponent} seat {seat}: {count} games", flush=True)
        return records

    rows_by_job: list[list[dict[str, Any]] | None] = [None] * len(jobs)
    completed_games = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            rows_by_job[index] = future.result()
            opponent, seat, count, _ = jobs[index]
            completed_games += count
            percent = 100 * completed_games / total
            print(
                f"PPO rollout progress: {completed_games}/{total} games "
                f"({percent:.1f}%) after {opponent} seat {seat}",
                flush=True,
            )
    results = [r for rows in rows_by_job if rows is not None for r in rows]
    print("PPO rollout extraction start", flush=True)
    data, manifest = extract(
        output / "replays", output, sha256(checkpoint), temperature, extract_workers
    )
    print("PPO rollout behavior audit start", flush=True)
    torch.set_num_threads(1)
    manifest["audit"] = audit_behavior(model, data, temperature)
    manifest["behavior_bin_sha256"] = sha256(output / "behavior.bin")
    manifest["collection"] = dict(
        opponents=opponents,
        games_per_opponent=games_per_opponent,
        total_games=total,
        opponent_weights=weights,
        allocated_games=allocation,
        fixed_model_sha256={n: sha256(p) for n, p in opponent_models.items()},
        game_fraction={n: allocation[n] / total for n in opponents},
        seed=seed,
        sample_seed=sample_seed,
        temperature=temperature,
        workers=workers,
        extract_workers=extract_workers,
        gbdt_model_sha256=sha256(gbdt_model) if gbdt_model else None,
        reward=DEFAULTS["reward"],
        timestep="one learner decision, sum prefix logp; terminal-only reward",
    )
    manifest["arena_results"] = results
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                games=total,
                episodes=len(manifest["episodes"]),
                decisions=manifest["decisions"],
                audit=manifest["audit"],
            )
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--gbdt-model", type=Path)
    p.add_argument("--opponents", nargs="+", default=DEFAULTS["opponents"])
    p.add_argument("--games-per-opponent", type=int)
    p.add_argument("--games", type=int, default=DEFAULTS["games_per_cycle"])
    p.add_argument(
        "--opponent-config",
        type=Path,
        help="JSON with weights and optional models (named fixed Transformer bins)",
    )
    p.add_argument("--seed", type=int, default=202640000)
    p.add_argument("--sample-seed", type=int, default=202650000)
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--extract-workers", type=int, default=1)
    a = p.parse_args()
    roster = json.loads(a.opponent_config.read_text()) if a.opponent_config else {}
    collect(
        a.checkpoint,
        a.output_dir,
        a.binary,
        a.gbdt_model,
        a.opponents,
        a.games_per_opponent,
        a.seed,
        a.sample_seed,
        a.temperature,
        a.workers,
        games=a.games,
        extract_workers=a.extract_workers,
        opponent_weights=roster.get("weights"),
        opponent_models=roster.get("models"),
    )
