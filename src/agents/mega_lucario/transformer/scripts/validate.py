"""Exercise standalone .bin inference, self-play, deterministic replays and parity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.mega_lucario.replay import sha256
from agents.mega_lucario.transformer.scripts.audit import ROOT, audit

if TYPE_CHECKING:
    from typing import Any


def validate(
    checkpoint: Path,
    model: Path,
    teacher_model: Path,
    output: Path,
    binary: Path,
    features: Path | None = None,
    probe_queries: int = 1000,
    workers: int = 1,
    threads: int = 1,
) -> dict[str, Any]:
    """Check standalone inference, self-play, replay determinism, and Python/C++ parity."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Validation output must be empty")
    if workers < 1 or threads < 1:
        raise ValueError("validation workers and threads must be positive")
    output.mkdir(parents=True, exist_ok=True)
    matchups = [
        ("mega_lucario_transformer", "mega_lucario_gbdt"),
        ("mega_lucario_gbdt", "mega_lucario_transformer"),
        ("mega_lucario_transformer", "mega_lucario_transformer"),
    ]
    results_by_matchup: list[list[dict[str, Any]] | None] = [None] * len(matchups)
    with tempfile.TemporaryDirectory() as temporary:
        standalone = Path(temporary) / "model.bin"
        shutil.copy2(model, standalone)

        def command_for(i: int, a: str, b: str, replay_dir: Path) -> list[str]:
            """Build one isolated native validation matchup command."""
            command = [
                str(binary.resolve()),
                "--agent-a",
                a,
                "--agent-b",
                b,
                "--deck-a",
                str(ROOT / "src/agents/mega_lucario/deck.csv"),
                "--deck-b",
                str(ROOT / "src/agents/mega_lucario/deck.csv"),
                "--games",
                "2",
                "--seed",
                str(202620000 + i * 10),
                "--replay-dir",
                str(replay_dir),
            ]
            for seat, name in (("a", a), ("b", b)):
                command += [
                    f"--model-{seat}",
                    str(standalone if name.endswith("_transformer") else teacher_model.resolve()),
                ]
            return command

        def run_matchup(i: int, a: str, b: str) -> list[dict[str, Any]]:
            """Run one independent matchup without sharing a replay directory."""
            run = subprocess.run(
                command_for(i, a, b, output / "replays" / f"pair-{i}"),
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
            games = [json.loads(line) for line in run.stdout.splitlines()]
            if len(games) != 2 or any(g["agent_errors"] or g["engine_errors"] for g in games):
                raise ValueError("Native battle errors")
            return games

        with ThreadPoolExecutor(max_workers=min(workers, len(matchups))) as pool:
            futures = {
                pool.submit(run_matchup, i, a, b): (i, a, b) for i, (a, b) in enumerate(matchups)
            }
            for future in as_completed(futures):
                i, a, b = futures[future]
                results_by_matchup[i] = future.result()
                print(f"Validated {a} vs {b}", flush=True)
        results = [game for games in results_by_matchup if games is not None for game in games]
        subprocess.run(
            command_for(2, *matchups[2], output / "determinism"),
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        for i in range(2):
            if sha256(output / "determinism" / f"game-{i}.jsonl") != sha256(
                output / "replays/pair-2" / f"game-{i}.jsonl"
            ):
                raise ValueError("Non-deterministic replay")
        parity = audit(
            output / "replays",
            checkpoint,
            standalone,
            output / "parity.json",
            features,
            binary.parent / "transformer_probe",
            probe_queries,
            threads,
        )
    report = dict(games=results, deterministic_replays=True, standalone_bin=True, parity=parity)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "model", "teacher-model", "output-dir"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "build/arena_cpp")
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--probe-queries", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    validate(
        a.checkpoint,
        a.model,
        a.teacher_model,
        a.output_dir,
        a.binary,
        a.feature_dir,
        a.probe_queries,
        a.workers,
        a.threads,
    )
