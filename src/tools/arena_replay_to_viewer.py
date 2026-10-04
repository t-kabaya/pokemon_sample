"""Convert arena JSONL replays to the Kaggle web visualizer replay shape."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWER_URL = "https://ptcgvis.heroz.jp/Visualizer/Replay/0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one arena replay JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Replay record must be an object: {path}:{line_number}")
            records.append(record)
    return records


def convert(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return both local-viewer metadata and official visualizer payload."""
    if not records or records[0].get("type") != "header":
        raise ValueError("Replay must start with a ptcg-replay-v1 header")
    header = records[0]
    if header.get("format") != "ptcg-replay-v1":
        raise ValueError("Unsupported replay format")

    decisions = [record for record in records if record.get("type") == "decision"]
    terminal = next((record for record in records if record.get("type") == "terminal"), None)
    if terminal is None:
        raise ValueError("Replay is incomplete: missing terminal record")

    payload = []
    turns = []
    visualization_data = []
    for index, record in enumerate(decisions):
        view = record.get("view") or {}
        observation = record.get("observation") or {}
        current = view.get("current") or observation.get("current") or {}
        select = view.get("select") or observation.get("select") or {}
        logs = view.get("logs") or observation.get("logs") or []
        action = record.get("action") or []
        seat = int(record.get("seat", current.get("yourIndex", 0)))

        entry = {
            "select": select,
            "logs": logs,
            "current": current,
            "selected": action,
            "obs": observation,
            "action": [action, action],
        }
        payload.append(entry)
        turns.append(
            {
                "index": index,
                "step": record.get("step", index),
                "game_turn": current.get("turn"),
                "player": seat,
                "agent": record.get("agent"),
                "context": select.get("context"),
                "action": action,
            }
        )
        visualization_data.append(
            {
                "step": record.get("step", index),
                "data": json.dumps(
                    {
                        "agent": record.get("agent"),
                        "view": view,
                        "action": action,
                        "gbdt_queries": record.get("gbdt_queries"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        )

    winner = terminal.get("result")
    result: dict[str, Any] = {
        "winner": winner if winner in (0, 1) else None,
        "outcome": "draw" if winner == 2 else "win" if winner in (0, 1) else "error",
        "total_turns": terminal.get("steps", len(decisions)),
    }
    if decisions:
        final_players = (decisions[-1].get("view") or {}).get("current", {}).get("players", [])
        if len(final_players) >= 2:
            result["player1_prizes"] = len(final_players[0].get("prize") or [])
            result["player2_prizes"] = len(final_players[1].get("prize") or [])

    agents = header.get("agents") or ["Player 1", "Player 2"]
    return {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "game": header.get("game"),
            "seed": header.get("seed"),
            "player1": agents[0] if len(agents) > 0 else "Player 1",
            "player2": agents[1] if len(agents) > 1 else "Player 2",
            "models": header.get("models"),
            "schemas": header.get("schemas"),
        },
        "result": result,
        "turns": turns,
        "visualization_data": visualization_data,
        "official_visualizer_payload": payload,
    }


def write_autoload_html(payload: list[dict[str, Any]], output: Path, viewer_url: str) -> None:
    """Write a tiny HTML page that posts embedded replay JSON to the official viewer."""
    json_data = json.dumps(payload, ensure_ascii=False)
    escaped_json = html.escape(json_data, quote=True)
    escaped_url = html.escape(viewer_url, quote=True)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pokemon TCG Replay</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: #f6f7fb;
      color: #1f2937;
    }}
    main {{
      max-width: 560px;
      padding: 32px;
      border: 1px solid #d7dce8;
      border-radius: 8px;
      background: white;
      box-shadow: 0 12px 28px rgba(31, 41, 55, 0.08);
    }}
    button {{
      margin-top: 18px;
      padding: 10px 14px;
      border: 0;
      border-radius: 6px;
      background: #2563eb;
      color: white;
      font: inherit;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Pokemon TCG Replay</h1>
    <p>Submitting the embedded replay to the official visualizer.</p>
    <form id="viewer-form" method="POST" action="{escaped_url}">
      <input type="hidden" name="json" value="{escaped_json}">
      <button type="submit">Open Viewer</button>
    </form>
  </main>
  <script>
    window.addEventListener("load", () => document.getElementById("viewer-form").submit());
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Arena game-*.jsonl replay")
    parser.add_argument("--output-json", type=Path, required=True, help="Converted viewer JSON")
    parser.add_argument("--output-html", type=Path, required=True, help="Auto-submit viewer HTML")
    parser.add_argument("--viewer-url", default=DEFAULT_VIEWER_URL)
    args = parser.parse_args()

    converted = convert(read_jsonl(args.replay))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_autoload_html(
        converted["official_visualizer_payload"], args.output_html, args.viewer_url
    )
    print(args.output_html)


if __name__ == "__main__":
    main()
