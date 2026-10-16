"""Compare independent public features and PyTorch predictions with native evidence."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
import argparse
import copy
import json
import subprocess
import tempfile

import numpy as np
import torch

from agents.mega_lucario.gbdt.features import DECK_COUNTS, SCHEMA_ID, action_queries, check_deck
from agents.mega_lucario.replay import decisions, inspect_replay, replay_schema, sha256
from agents.mega_lucario.transformer.checkpoint import load_checkpoint
from agents.mega_lucario.transformer.features import (
    CONTRACT_ID,
    NAMES,
    encode,
    token_indices,
    tokenize,
)
from agents.mega_lucario.transformer.model import select_logits
from agents.mega_lucario.transformer.train import load_split

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from numpy.typing import NDArray

    from agents.mega_lucario.transformer.model import ObservationTransformer


ATOL = RTOL = 2e-5


@torch.no_grad()
def predictions(
    model: ObservationTransformer,
    matrices: Sequence[NDArray[Any]],
    batch_size: int = 64,
    with_value: bool = False,
) -> Iterator[NDArray[np.floating[Any]] | tuple[NDArray[np.floating[Any]], float]]:
    """Batch independent observations, preserving each token/selection mask."""
    dtype = next(model.parameters()).dtype
    for start in range(0, len(matrices), batch_size):
        part = matrices[start : start + batch_size]
        n = max(map(len, part))
        x = np.zeros((len(part), n, len(NAMES)), np.float32)
        mask = np.zeros((len(part), n), bool)
        for j, matrix in enumerate(part):
            x[j, : len(matrix)] = matrix
            mask[j, : len(matrix)] = True
        score_tensor, values = model.forward_with_value(
            torch.from_numpy(x).to(dtype), torch.from_numpy(mask)
        )
        scores = score_tensor.numpy()
        for j, matrix in enumerate(part):
            yield (
                (scores[j, : len(matrix)], float(values[j]))
                if with_value
                else scores[j, : len(matrix)]
            )


def audit(
    replay_dir: Path,
    checkpoint: Path,
    binary_model: Path,
    output: Path,
    feature_dir: Path | None = None,
    probe: Path | None = None,
    probe_queries: int = 1000,
    threads: int = 1,
) -> dict[str, Any]:
    """Compare token features, logits, Values, and actions across Python and C++."""
    if threads < 1:
        raise ValueError("audit threads must be positive")
    torch.set_num_threads(threads)
    model, metadata = load_checkpoint(checkpoint, CONTRACT_ID, DECK_COUNTS, NAMES)
    identity = sha256(checkpoint)
    # Export stores float32 tensors; evaluate those tensors in double as C++ does.
    float_model = copy.deepcopy(model)
    model = model.double()
    report = dict(
        contract_id=CONTRACT_ID,
        model_id=identity,
        bin_sha256=sha256(binary_model),
        atol=ATOL,
        rtol=RTOL,
        games=0,
        decisions=0,
        queries=0,
        rows=0,
        max_abs_error=0.0,
        feature_max_abs_error=0.0,
        value_max_abs_error=0.0,
        float32_max_abs_error=0.0,
        action_mismatches=0,
        agent_errors=0,
        engine_errors=0,
        probe_queries=0,
        probe_rows=0,
    )
    pending = []
    for path in sorted(replay_dir.rglob("game-*.jsonl")):
        header, terminal = inspect_replay(path)
        if replay_schema(header) != SCHEMA_ID:
            raise ValueError("Replay schema mismatch")
        for seat, name in enumerate(header["agents"]):
            if name == "mega_lucario_transformer":
                check_deck(header["decks"][seat])
                if header["models"][seat] != identity:
                    raise ValueError("Wrong replay model")
        table = {
            c["cardId"]: c for c in json.loads((path.parent / "card_metadata.json").read_text())
        }
        used = False
        for record in decisions(path):
            if record["agent"] != "mega_lucario_transformer":
                continue
            used = True
            report["decisions"] += 1
            queries = list(action_queries(record["observation"], record["action"], table))
            native = record.get("transformer_queries", [])
            if len(native) != len(queries):
                raise ValueError("Missing Transformer query evidence")
            for query, cpp in zip(queries, native, strict=True):
                if any(query[k] != cpp[k] for k in ("prefix", "options", "target")):
                    raise ValueError("Selection contract mismatch")
                raw = tokenize(record["observation"], query["prefix"], table)
                indices = token_indices(record["observation"], query["options"])
                if not np.allclose(raw, np.asarray(cpp["features"]), atol=2e-6, rtol=2e-6):
                    raise ValueError("Feature mismatch")
                report["feature_max_abs_error"] = max(
                    report["feature_max_abs_error"],
                    float(np.max(np.abs(raw - np.asarray(cpp["features"])))),
                )
                pending.append((encode(raw), indices, query, cpp))
                report["queries"] += 1
                report["rows"] += len(raw)
        report["games"] += int(used)
    matrices = [p[0] for p in pending]
    for item, double_scores, float_scores in zip(
        pending,
        predictions(model, matrices, with_value=True),
        predictions(float_model, matrices, with_value=True),
        strict=True,
    ):
        _, indices, query, cpp = item
        double_scores, double_value = double_scores
        float_scores, float_value = float_scores
        if "value" not in cpp:
            raise ValueError("Missing native Value evidence; regenerate replay")
        for value in (double_value, float_value):
            np.testing.assert_allclose(value, cpp["value"], atol=ATOL, rtol=RTOL)
            report["value_max_abs_error"] = max(
                report["value_max_abs_error"], abs(value - cpp["value"])
            )
        scores = double_scores[indices]
        float_scores = float_scores[indices]
        for name, values in [("max_abs_error", scores), ("float32_max_abs_error", float_scores)]:
            np.testing.assert_allclose(values, cpp["scores"], atol=ATOL, rtol=RTOL)
            report[name] = max(report[name], float(np.max(np.abs(values - cpp["scores"]))))
            chosen = int(select_logits(torch.from_numpy(values)))
            if query["options"][chosen] != query["target"]:
                raise ValueError("Python/C++ action mismatch")
    if not report["queries"]:
        raise ValueError("No native Transformer queries")
    if feature_dir is not None:
        manifest = json.loads((feature_dir / "manifest.json").read_text())
        if (
            manifest["contract_id"] != CONTRACT_ID
            or sha256(feature_dir / "manifest.json") != metadata["dataset_sha256"]
        ):
            raise ValueError("Checkpoint/feature dataset mismatch")
        probe_split = "test" if "test" in manifest["splits"] else "validation"
        report["probe_split"] = probe_split
        data = load_split(feature_dir, manifest, probe_split)
        count = min(probe_queries, len(data["sizes"]))
        if count < 1 or probe is None:
            raise ValueError("Positive probe query count and binary required")
        indices = np.linspace(0, len(data["sizes"]) - 1, count, dtype=int)
        selected_matrices = []
        with tempfile.TemporaryFile(mode="w+") as stream:
            for index in indices:
                a, b = data["offsets"][index : index + 2]
                x = data["x"][a:b]
                stream.write(f"{len(x)} {len(NAMES)}\n")
                np.savetxt(stream, x, fmt="%.9g")
                selected_matrices.append(x)
            stream.seek(0)
            result = subprocess.run(
                [str(probe.resolve()), str(binary_model.resolve())],
                stdin=stream,
                capture_output=True,
                text=True,
                check=True,
            )
        expected = np.concatenate(list(predictions(model, selected_matrices)))
        actual = np.fromstring(result.stdout, sep=" ")
        np.testing.assert_allclose(expected, actual, atol=ATOL, rtol=RTOL)
        report["probe_max_abs_error"] = float(
            np.max(
                np.abs(np.asarray(expected)[np.isfinite(expected)] - actual[np.isfinite(expected)])
            )
        )
        report["probe_queries"] = count
        report["probe_rows"] = len(expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-dir", "checkpoint", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--feature-dir", type=Path)
    p.add_argument("--probe", type=Path, default=ROOT / "build/transformer_probe")
    p.add_argument("--probe-queries", type=int, default=1000)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    audit(
        a.replay_dir,
        a.checkpoint,
        a.model,
        a.output,
        a.feature_dir,
        a.probe,
        a.probe_queries,
        a.threads,
    )
