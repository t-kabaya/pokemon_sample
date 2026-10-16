"""PPO terminal boundaries, joint probabilities, clipping and policy-preserving critic."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/tools"))
from agents.mega_lucario.ppo.extract import combine_extracted
from agents.mega_lucario.ppo.rollout import compute_gae, joint_stats
from agents.mega_lucario.ppo.scripts.collect import RolloutJob, split_rollout_jobs
from agents.mega_lucario.ppo.scripts.initialize import CONTRACT_ID, DECK_COUNTS, NAMES, initialize
from agents.mega_lucario.ppo.train import clipped_loss, distribution_targets, load_rollout
from agents.mega_lucario.transformer.checkpoint import load_checkpoint
from agents.mega_lucario.transformer.features import CONTRACT
from agents.mega_lucario.transformer.model import ARCHITECTURE, ObservationTransformer


class PpoContracts(unittest.TestCase):
    """Validate ppo contracts."""

    @staticmethod
    def extracted_fixture(name: str, seed: int, rows: int) -> dict[str, object]:
        """Return a small per-replay extraction result."""
        return dict(
            matrices=[np.full((rows, 3), seed, dtype=np.float32)],
            sizes=[rows],
            targets=[rows - 1],
            decision_offsets=[0, 1],
            metadata=[dict(game=name, seat=0, step=seed, action=[rows], queries=[])],
            episodes=[dict(game=name, seat=0, begin=0, end=1, reward=1.0)],
            game=dict(path=f"{name}.jsonl", sha256=name, seed=seed),
            scalars=dict(
                old_logp=[-0.1 * seed],
                old_value=[0.01 * seed],
                old_entropy=[0.2],
                reward=[1.0],
                done=[True],
                episode=[0],
                step=[seed],
            ),
            zero_queries=seed,
            digest=name,
            seed=seed,
            agents=("mega_lucario_transformer", "mega_lucario"),
        )

    def test_gae_separates_seats_and_rejects_truncation(self) -> None:
        """Check gae separates seats and rejects truncation."""
        args = (
            [0.2, -0.1, 0.4, -0.3],
            [0, 0, 1, -1],
            [False, False, True, True],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        )
        advantage, returns = compute_gae(*args, gamma=0.9, lam=0.8)
        np.testing.assert_allclose(advantage, [0.592, -0.674, 0.6, -0.7])
        np.testing.assert_allclose(returns, [0.792, -0.774, 1, -1])
        with self.assertRaises(ValueError):
            compute_gae(args[0], args[1], [False] * 4, *args[3:])
        with self.assertRaises(ValueError):
            compute_gae(*args, gamma=1.1)
        with self.assertRaises(ValueError):
            compute_gae(*args[:4], [0, 0, 0, 1])

    def test_joint_logp_sums_prefixes_and_value_uses_first(self) -> None:
        """Check joint logp sums prefixes and value uses first."""

        class Fixed(torch.nn.Module):
            """Validate fixed."""

            def forward_with_value(
                self, x: torch.Tensor, mask: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Return differentiable test logits and Values."""
                return torch.tensor([[math.log(3), 0], [0, 0]], dtype=torch.float64), torch.tensor(
                    [0.2, 0.9]
                )

        b = dict(
            x=None,
            mask=torch.ones(2, 2, dtype=torch.bool),
            targets=torch.tensor([0, 1]),
            owners=torch.tensor([0, 0]),
            first=torch.tensor([0]),
            count=1,
        )
        logp, value, entropy = joint_stats(Fixed(), b, 1.0)
        self.assertAlmostEqual(float(logp), math.log(0.75 * 0.5))
        self.assertAlmostEqual(float(value), 0.2, places=6)
        self.assertAlmostEqual(
            float(entropy), -0.75 * math.log(0.75) - 0.25 * math.log(0.25) + math.log(2)
        )

    def test_clipping_for_both_advantage_signs_and_value(self) -> None:
        """Check clipping for both advantage signs and value."""
        cfg = dict(clip=0.2, value_clip=0.2, value_coefficient=0.0, entropy_coefficient=0.0)
        zero = torch.tensor([0.0])
        for ratio, advantage, expected in [(2.0, 1.0, -1.2), (0.5, -1.0, 0.8)]:
            loss, _ = clipped_loss(
                torch.tensor([math.log(ratio)]),
                zero,
                zero,
                zero,
                zero,
                torch.tensor([advantage]),
                zero,
                cfg,
            )
            self.assertAlmostEqual(float(loss), expected, places=6)
        _, stats = clipped_loss(
            zero, torch.tensor([0.5]), zero, zero, zero, zero, torch.tensor([1.0]), cfg
        )
        self.assertAlmostEqual(float(stats["value_loss"]), 0.32, places=6)

    def test_distribution_targets(self) -> None:
        """Check distribution targets."""
        returns = torch.tensor([-2.0, -1.0, -0.98, 0.0, 0.02, 1.0, 2.0])
        targets = distribution_targets(returns)
        torch.testing.assert_close(targets.sum(-1), torch.ones(7))
        torch.testing.assert_close(
            targets @ torch.linspace(-1, 1, 51), returns.clamp(-1, 1), atol=1e-6, rtol=0
        )
        self.assertAlmostEqual(float(targets[2, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(targets[2, 1]), 0.5, places=5)

    def test_wrong_behavior_checkpoint_rejected_before_loading_arrays(self) -> None:
        """Check wrong behavior checkpoint rejected before loading arrays."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "manifest.json").write_text(
                json.dumps(
                    dict(
                        format="ptcg-ppo-rollout-v1", contract_id=CONTRACT_ID, behavior_sha256="old"
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                load_rollout(p, CONTRACT_ID, "new")

    def test_parallel_extraction_merge_preserves_sequential_contract(self) -> None:
        """Check per-replay extraction results merge like sequential extraction."""
        first = self.extracted_fixture("a", 10, 2)
        second = self.extracted_fixture("b", 20, 3)
        combined = combine_extracted([first, second])
        self.assertEqual(combined["decision_offsets"], [0, 1, 2])
        self.assertEqual(combined["sizes"], [2, 3])
        self.assertEqual(combined["targets"], [1, 2])
        self.assertEqual([m["game"] for m in combined["metadata"]], ["a", "b"])
        self.assertEqual(combined["episodes"][0]["begin"], 0)
        self.assertEqual(combined["episodes"][0]["end"], 1)
        self.assertEqual(combined["episodes"][1]["begin"], 1)
        self.assertEqual(combined["episodes"][1]["end"], 2)
        self.assertEqual(combined["scalars"]["episode"], [0, 1])
        self.assertEqual(combined["zero_queries"], 30)
        np.testing.assert_array_equal(combined["matrices"][0], np.full((2, 3), 10))
        np.testing.assert_array_equal(combined["matrices"][1], np.full((3, 3), 20))

    def test_rollout_chunks_preserve_per_game_seeds(self) -> None:
        """Check C++ chunks retain the original game and learner sampling seed ranges."""
        jobs = split_rollout_jobs([RolloutJob("self", 1, 5, 10, 5, 0)], 2)
        self.assertEqual([job.count for job in jobs], [2, 2, 1])
        self.assertEqual([job.offset for job in jobs], [10, 12, 14])
        self.assertEqual([job.batch_start for job in jobs], [0, 2, 4])
        self.assertEqual([job.learner_sample_seed(100) for job in jobs], [125, 127, 129])

    def test_extraction_merge_rejects_duplicate_games(self) -> None:
        """Check duplicate replay identities are still rejected after parallel extraction."""
        first = self.extracted_fixture("a", 10, 2)
        duplicate = self.extracted_fixture("a", 10, 3)
        with self.assertRaisesRegex(ValueError, "Duplicate PPO game"):
            combine_extracted([first, duplicate])

    def test_uncompressed_npz_preserves_rollout_arrays(self) -> None:
        """Check uncompressed npz stores the same arrays as compressed npz."""
        arrays = dict(
            x=np.arange(12, dtype=np.float32).reshape(4, 3),
            sizes=np.array([2, 2], dtype=np.int32),
            targets=np.array([1, 0], dtype=np.int32),
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            np.savez(p / "plain.npz", **arrays)
            np.savez_compressed(p / "compressed.npz", **arrays)
            with np.load(p / "plain.npz") as plain, np.load(p / "compressed.npz") as compressed:
                for key, expected in arrays.items():
                    np.testing.assert_array_equal(plain[key], expected)
                    np.testing.assert_array_equal(plain[key], compressed[key])

    def test_zero_critic_upgrade_preserves_policy(self) -> None:
        """Check zero critic upgrade preserves policy."""
        torch.set_num_threads(1)
        torch.manual_seed(9)
        model = ObservationTransformer(len(NAMES)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            torch.save(
                dict(
                    format="ptcg-transformer-checkpoint-v1",
                    architecture=ARCHITECTURE,
                    contract_id=CONTRACT_ID,
                    contract=CONTRACT,
                    deck_counts={str(k): v for k, v in DECK_COUNTS.items()},
                    feature_names=NAMES,
                    state_dict=model.state_dict(),
                ),
                p / "old.pt",
            )
            initialize(p / "old.pt", p / "new.pt")
            expanded, _ = load_checkpoint(p / "new.pt", CONTRACT_ID, DECK_COUNTS, NAMES)
            from test_transformer_contracts import TransformerContracts

            x = TransformerContracts.fixture(25).repeat(2, 1, 1)
            mask = torch.ones(2, 25, dtype=torch.bool)
            with torch.no_grad():
                logits, value = expanded.forward_with_value(x, mask)
            torch.testing.assert_close(logits, model(x, mask), atol=0, rtol=0)
            torch.testing.assert_close(value, torch.zeros(2), atol=1e-6, rtol=0)
            initialize(p / "new.pt", p / "copy.pt")
            self.assertEqual((p / "new.pt").read_bytes(), (p / "copy.pt").read_bytes())

    def test_native_sampling_distribution_and_forced_action(self) -> None:
        """Check native sampling distribution and forced action."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "sampling.cpp"
            source.write_text("""#include "agents/mega_lucario/ppo/cpp/sampling.h"
#include <iostream>
int main(){std::mt19937 a(42),b(42);int first=0;for(int i=0;i<10000;++i){
ppo::Statistics x,y;int chosen=ppo::sample({std::log(3.),0.},1.,a,x);
if(chosen!=ppo::sample({std::log(3.),0.},1.,b,y))return 1;
if(std::abs(x.logp-std::log(chosen==0?.75:.25))>1e-12)return 2;
first+=chosen==0;}
ppo::Statistics x;if(ppo::sample({9.},1.,a,x)!=0 || x.logp!=0 || x.entropy!=0)return 3;
std::cout<<first;
}""")
            subprocess.run(
                shlex.split(os.environ.get("CXX", "g++"))
                + [
                    "-std=c++20",
                    "-O2",
                    "-I",
                    str(ROOT / "src"),
                    str(source),
                    "-o",
                    str(p / "test"),
                ],
                check=True,
            )
            result = subprocess.run([str(p / "test")], check=True, capture_output=True, text=True)
            self.assertLess(abs(int(result.stdout) / 10000 - 0.75), 0.02)
