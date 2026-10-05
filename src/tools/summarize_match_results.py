"""Summarize win rates from a rating_arena games.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict[str, dict[str, float | int]]:
    """Read arena match rows and return per-label win/draw/loss counts and rates."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels = sorted({row["a"] for row in rows} | {row["b"] for row in rows})
    summary: dict[str, dict[str, float | int]] = {
        label: dict(games=0, wins=0, draws=0, losses=0, win_rate=0.0, non_draw_win_rate=0.0)
        for label in labels
    }
    for row in rows:
        a, b, result = row["a"], row["b"], row["result"]
        if row.get("engine_errors") or result not in (0, 1, 2):
            continue
        for label in (a, b):
            summary[label]["games"] += 1
        if result == 2:
            summary[a]["draws"] += 1
            summary[b]["draws"] += 1
        else:
            winner = (a, b)[result]
            loser = b if winner == a else a
            summary[winner]["wins"] += 1
            summary[loser]["losses"] += 1
    for item in summary.values():
        games = int(item["games"])
        decisive = int(item["wins"]) + int(item["losses"])
        item["win_rate"] = (float(item["wins"]) + 0.5 * float(item["draws"])) / games if games else 0.0
        item["non_draw_win_rate"] = float(item["wins"]) / decisive if decisive else 0.0
    return summary


def main() -> None:
    """Parse arguments and print a compact win-rate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", type=Path)
    args = parser.parse_args()
    summary = summarize(args.games)
    for label, item in summary.items():
        print(
            f"{label}: games={item['games']} wins={item['wins']} draws={item['draws']} "
            f"losses={item['losses']} win_rate={item['win_rate']:.3f} "
            f"non_draw_win_rate={item['non_draw_win_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
