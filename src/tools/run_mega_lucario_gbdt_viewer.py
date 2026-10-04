"""Run Mega Lucario rule vs GBDT in Python and create official visualizer HTML."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CG_LIB = ROOT / "external/official_assets/cg-lib"
LEGACY_CG_LIB = ROOT.parent / "pokemon/datasets/sample_submission/sample_submission"
DEFAULT_VIEWER_URL = "https://ptcgvis.heroz.jp/Visualizer/Replay/0"


def python_cg_path() -> Path:
    """Choose a complete official cg Python package for this platform."""
    if platform.system() == "Darwin" and (LEGACY_CG_LIB / "cg/libcg.dylib").exists():
        return LEGACY_CG_LIB
    return CG_LIB


sys.path.insert(0, str(python_cg_path()))
sys.path.insert(0, str(ROOT / "src"))

from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from cg.sim import lib  # noqa: E402

from agents.mega_lucario.gbdt.bundle import load_bundle  # noqa: E402
from agents.mega_lucario.gbdt.features import (  # noqa: E402
    NAMES,
    check_deck,
    feature_row,
    options_for,
)
from agents.mega_lucario.gbdt.train import predict_banks  # noqa: E402
from agents.mega_lucario.main import agent as rule_agent  # noqa: E402
from tools.arena_replay_to_viewer import write_autoload_html  # noqa: E402


def enum_lookup(enum: type[IntEnum]) -> dict[str, int]:
    """Build a normalized lookup for enum names and display spellings."""
    values = {}
    for item in enum:
        compact = item.name.replace("_", "").lower()
        values[compact] = int(item)
    return values


def load_enum_maps() -> dict[str, dict[str, int]]:
    """Load cg enum mappings after the official package is available."""
    from cg.api import AreaType, OptionType, SelectContext, SelectType, SpecialConditionType

    return {
        "area": enum_lookup(AreaType),
        "context": enum_lookup(SelectContext),
        "option_type": enum_lookup(OptionType),
        "select_type": enum_lookup(SelectType),
        "special_condition": enum_lookup(SpecialConditionType),
    }


ENUM_MAPS = load_enum_maps()


def enum_value(kind: str, value: Any) -> Any:
    """Convert cg JSON enum strings to their integer values."""
    if not isinstance(value, str):
        return value
    compact = value.replace("_", "").lower()
    return ENUM_MAPS[kind].get(compact, value)


def numeric_observation(value: Any, *, option: bool = False) -> Any:
    """Convert an official Python observation to the numeric feature contract."""
    if isinstance(value, list):
        return [numeric_observation(item, option=option) for item in value]
    if not isinstance(value, dict):
        return value
    converted: dict[str, Any] = {}
    for key, item in value.items():
        child_option = option or key == "option"
        if key == "type":
            converted[key] = enum_value("option_type" if option else "select_type", item)
        elif key == "context":
            converted[key] = enum_value("context", item)
        elif key in ("area", "inPlayArea", "fromArea", "toArea"):
            converted[key] = enum_value("area", item)
        elif key == "specialConditionType":
            converted[key] = enum_value("special_condition", item)
        else:
            converted[key] = numeric_observation(item, option=child_option)
    return converted


def card_table() -> dict[int, dict[str, Any]]:
    """Return card metadata keyed by card ID for GBDT features."""
    cards = json.loads(lib.AllCard().decode())
    table = {}
    for card in cards:
        card_id = card.get("id", card.get("cardId"))
        if card_id is None:
            raise ValueError("Card metadata is missing card ID")
        table[int(card_id)] = card
    return table


class GbdtAgent:
    """Mega Lucario GBDT policy compatible with the official Python simulator."""

    def __init__(self, model: Path) -> None:
        """Load the exported GBDT bundle and bind the official deck contract."""
        self.training, self.boosters = load_bundle(model)
        self.table = card_table()
        self.route = NAMES.index("context_0")
        self.deck = [int(card_id) for card_id in (ROOT / "src/agents/mega_lucario/deck.csv").read_text().split()]
        check_deck(self.deck)

    def __call__(self, obs: dict[str, Any]) -> list[int]:
        """Return the deck during setup or a legal action during play."""
        if obs.get("select") is None:
            return self.deck
        numeric = numeric_observation(deepcopy(obs))
        prefix: list[int] = []
        while True:
            options = options_for(numeric, prefix)
            if not options:
                return prefix
            rows = np.asarray(
                [feature_row(numeric, option, prefix, self.table) for option in options],
                dtype=np.float32,
            )
            scores = predict_banks(self.boosters, rows, self.route)
            choice = options[int(np.argmax(scores))]
            if choice == -1:
                return prefix
            prefix.append(choice)


def run(model: Path, output_json: Path, output_html: Path, viewer_url: str) -> None:
    """Run one game, save the official visualizer payload, and write auto-submit HTML."""
    gbdt_agent = GbdtAgent(model)
    deck_obs = {"select": None, "logs": [], "current": None}
    deck_a = rule_agent(deck_obs)
    deck_b = gbdt_agent(deck_obs)
    obs, start = battle_start(deck_a, deck_b)
    if obs is None:
        raise RuntimeError(f"Battle failed to start: player={start.errorPlayer} type={start.errorType}")

    obs_log: list[Any] = [""]
    action_log: list[Any] = [None]
    max_steps = 10000
    try:
        for _ in range(max_steps):
            current = obs.get("current") or {}
            if obs.get("select") is None or current.get("result", -1) != -1:
                break
            actor = current.get("yourIndex", 0)
            action = rule_agent(obs) if actor == 0 else gbdt_agent(obs)
            clean_obs = dict(obs)
            clean_obs.pop("search_begin_input", None)
            obs_log.append(clean_obs)
            action_log.append(action)
            obs = battle_select(action)
        else:
            raise RuntimeError("Python simulator step limit reached")

        payload = json.loads(visualize_data())
        for index, entry in enumerate(payload):
            if index < len(obs_log):
                entry["obs"] = obs_log[index]
            if index < len(action_log):
                entry["action"] = [action_log[index], action_log[index]]
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_autoload_html(payload, output_html, viewer_url)
        print(output_html)
    finally:
        battle_finish()


def main() -> None:
    """Parse arguments and run the visualization match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/mega_lucario/model.gbdt")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--viewer-url", default=DEFAULT_VIEWER_URL)
    args = parser.parse_args()
    run(args.model, args.output_json, args.output_html, args.viewer_url)


if __name__ == "__main__":
    main()
