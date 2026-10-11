"""Convert complete C++ stochastic replays to aligned Lucario PPO decisions."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from agents.mega_lucario.gbdt.features import SCHEMA_ID, action_queries, check_deck
from agents.mega_lucario.replay import decisions, inspect_replay, replay_schema, sha256
from agents.mega_lucario.transformer.features import CONTRACT_ID, encode, token_indices, tokenize

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


def _empty_scalars() -> dict[str, list[Any]]:
    """Return scalar rollout columns."""
    return {
        k: [] for k in ("old_logp", "old_value", "old_entropy", "reward", "done", "episode", "step")
    }


def _extract_path(
    path: Path, replay_dir: Path, behavior_id: str, temperature: float
) -> dict[str, Any]:
    """Extract learner decisions from one replay file."""
    matrices = []
    sizes = []
    targets = []
    decision_offsets = [0]
    metadata = []
    episodes = []
    scalars = _empty_scalars()
    zero_queries = 0
    header, terminal = inspect_replay(path)
    digest = sha256(path)
    if replay_schema(header) != SCHEMA_ID or not header.get("sampling"):
        raise ValueError("Not a stochastic rollout")
    game = dict(
        path=str(path.relative_to(replay_dir)),
        sha256=digest,
        seed=header["seed"],
        agents=header["agents"],
        sampling=header["sampling"],
        result=terminal["result"],
    )
    table = {c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())}
    records = list(decisions(path))
    for seat, sampling in enumerate(header["sampling"]):
        if sampling is None:
            continue
        if (
            header["agents"][seat] != "mega_lucario_transformer"
            or header["models"][seat] != behavior_id
            or sampling["temperature"] != temperature
        ):
            raise ValueError("Rollout behavior/model/temperature mismatch")
        check_deck(header["decks"][seat])
        begin = len(metadata)
        episode = len(episodes)
        for record in records:
            if record["seat"] != seat:
                continue
            queries = list(action_queries(record["observation"], record["action"], table))
            native = record.get("transformer_queries", [])
            stats = record.get("ppo")
            if stats is None or len(native) != len(queries) or stats["queries"] != len(queries):
                raise ValueError("Missing PPO statistics")
            if not queries:
                zero_queries += 1
                continue
            for q, cpp in zip(queries, native, strict=True):
                if any(q[k] != cpp[k] for k in ("prefix", "options", "target")):
                    raise ValueError("PPO selection contract mismatch")
                raw = tokenize(record["observation"], q["prefix"], table)
                if not np.allclose(raw, np.asarray(cpp["features"]), atol=2e-6, rtol=2e-6):
                    raise ValueError("PPO feature mismatch")
                matrices.append(encode(raw))
                sizes.append(len(raw))
                targets.append(token_indices(record["observation"], [q["target"]])[0])
            decision_offsets.append(len(sizes))
            for key in ("old_logp", "old_value", "old_entropy"):
                scalars[key].append(stats[key])
            scalars["reward"].append(0.0)
            scalars["done"].append(False)
            scalars["episode"].append(episode)
            scalars["step"].append(record["step"])
            metadata.append(
                dict(
                    game=digest,
                    seat=seat,
                    step=record["step"],
                    action=record["action"],
                    queries=[{k: q[k] for k in ("prefix", "options", "target")} for q in queries],
                )
            )
        if len(metadata) == begin:
            raise ValueError("No learner decisions in episode")
        reward = 0.0 if terminal["result"] == 2 else (1.0 if terminal["result"] == seat else -1.0)
        scalars["reward"][-1] = reward
        scalars["done"][-1] = True
        episodes.append(dict(game=digest, seat=seat, begin=begin, end=len(metadata), reward=reward))
    return dict(
        matrices=matrices,
        sizes=sizes,
        targets=targets,
        decision_offsets=decision_offsets,
        metadata=metadata,
        episodes=episodes,
        game=game,
        scalars=scalars,
        zero_queries=zero_queries,
        digest=digest,
        seed=header.get("seed"),
        agents=tuple(header["agents"]),
    )


def combine_extracted(extracted: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Combine per-replay extraction results in replay order."""
    matrices = []
    sizes = []
    targets = []
    decision_offsets = [0]
    metadata = []
    episodes = []
    games = []
    scalars = _empty_scalars()
    seen = set()
    seeded = set()
    zero_queries = 0
    for item in extracted:
        if item is None:
            raise ValueError("Missing extracted replay result")
        digest = item["digest"]
        key = (item["seed"], item["agents"])
        if digest in seen or key in seeded:
            raise ValueError("Duplicate PPO game")
        seen.add(digest)
        seeded.add(key)
        decision_base = len(metadata)
        episode_base = len(episodes)
        query_base = len(sizes)
        games.append(item["game"])
        matrices.extend(item["matrices"])
        sizes.extend(item["sizes"])
        targets.extend(item["targets"])
        decision_offsets.extend(query_base + offset for offset in item["decision_offsets"][1:])
        metadata.extend(item["metadata"])
        for episode in item["episodes"]:
            episodes.append(
                dict(
                    episode,
                    begin=episode["begin"] + decision_base,
                    end=episode["end"] + decision_base,
                )
            )
        for key, values in item["scalars"].items():
            if key == "episode":
                scalars[key].extend(episode_base + value for value in values)
            else:
                scalars[key].extend(values)
        zero_queries += item["zero_queries"]
    return dict(
        matrices=matrices,
        sizes=sizes,
        targets=targets,
        decision_offsets=decision_offsets,
        metadata=metadata,
        episodes=episodes,
        games=games,
        scalars=scalars,
        zero_queries=zero_queries,
    )


def extract(
    replay_dir: Path, output: Path, behavior_id: str, temperature: float, workers: int = 1
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Extract learner decisions, behavior statistics, and terminal-only rewards."""
    paths = sorted(replay_dir.rglob("game-*.jsonl"))
    total_paths = len(paths)
    if workers < 1:
        raise ValueError("extract workers must be positive")
    print(
        f"PPO rollout extraction: reading {total_paths} replay files with {workers} workers",
        flush=True,
    )
    if workers == 1:
        extracted = []
        for index, path in enumerate(paths, start=1):
            extracted.append(_extract_path(path, replay_dir, behavior_id, temperature))
            if index == total_paths or index % 50 == 0:
                percent = 100 * index / total_paths
                decisions_so_far = sum(len(item["metadata"]) for item in extracted)
                queries_so_far = sum(len(item["sizes"]) for item in extracted)
                print(
                    f"PPO rollout extraction progress: {index}/{total_paths} files "
                    f"({percent:.1f}%), decisions={decisions_so_far}, "
                    f"queries={queries_so_far}",
                    flush=True,
                )
    else:
        extracted = [None] * total_paths
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_path, path, replay_dir, behavior_id, temperature): index
                for index, path in enumerate(paths)
            }
            for future in as_completed(futures):
                index = futures[future]
                extracted[index] = future.result()
                done = sum(1 for item in extracted if item is not None)
                if done == total_paths or done % 50 == 0:
                    percent = 100 * done / total_paths
                    decisions_so_far = sum(len(item["metadata"]) for item in extracted if item)
                    queries_so_far = sum(len(item["sizes"]) for item in extracted if item)
                    print(
                        f"PPO rollout extraction progress: {done}/{total_paths} files "
                        f"({percent:.1f}%), decisions={decisions_so_far}, "
                        f"queries={queries_so_far}",
                        flush=True,
                    )
    combined = combine_extracted(extracted)
    matrices = combined["matrices"]
    sizes = combined["sizes"]
    targets = combined["targets"]
    decision_offsets = combined["decision_offsets"]
    metadata = combined["metadata"]
    episodes = combined["episodes"]
    games = combined["games"]
    scalars = combined["scalars"]
    zero_queries = combined["zero_queries"]
    if not matrices:
        raise ValueError("No PPO trajectories")
    print("PPO rollout extraction: building arrays", flush=True)
    data = dict(
        x=np.concatenate(matrices),
        sizes=np.asarray(sizes, dtype=np.int32),
        targets=np.asarray(targets, dtype=np.int32),
        decision_offsets=np.asarray(decision_offsets, dtype=np.int64),
        query_offsets=np.r_[0, np.cumsum(sizes)],
    )
    for key, values in scalars.items():
        data[key] = np.asarray(
            values,
            dtype=bool if key == "done" else np.int64 if key in ("episode", "step") else np.float64,
        )
    output.mkdir(parents=True, exist_ok=True)
    print("PPO rollout extraction: writing rollout.npz", flush=True)
    np.savez(output / "rollout.npz", **data)
    print("PPO rollout extraction: writing decisions.json", flush=True)
    (output / "decisions.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    report = dict(
        format="ptcg-ppo-rollout-v1",
        contract_id=CONTRACT_ID,
        behavior_sha256=behavior_id,
        temperature=temperature,
        games=games,
        episodes=episodes,
        decisions=len(metadata),
        queries=len(sizes),
        rows=len(data["x"]),
        zero_query_decisions=zero_queries,
        arrays_sha256=sha256(output / "rollout.npz"),
        decisions_sha256=sha256(output / "decisions.json"),
    )
    return data, report
