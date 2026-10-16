"""Decision-level autoregressive policy statistics and terminal-only GAE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, NDArray

    from agents.mega_lucario.transformer.model import ObservationTransformer


def compute_gae(
    values: ArrayLike,
    rewards: ArrayLike,
    done: ArrayLike,
    episodes: ArrayLike,
    steps: ArrayLike,
    gamma: float = 1.0,
    lam: float = 0.99,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute GAE and returns independently for complete seat-specific episodes."""
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid GAE parameters")
    values = np.asarray(values, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    done = np.asarray(done, dtype=bool)
    episodes = np.asarray(episodes)
    steps = np.asarray(steps)
    if (
        not (values.shape == rewards.shape == done.shape == episodes.shape == steps.shape)
        or values.ndim != 1
    ):
        raise ValueError("Invalid transition arrays")
    if not np.isfinite(values).all() or not np.isfinite(rewards).all():
        raise ValueError("Non-finite transition")
    advantage = np.zeros_like(values)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(steps[rows], kind="stable")]
        if not done[rows[-1]] or done[rows[:-1]].any() or len(np.unique(steps[rows])) != len(rows):
            raise ValueError("Incomplete or inconsistent episode")
        following_value = following_advantage = 0.0
        for row in rows[::-1]:
            continuation = 0.0 if done[row] else 1.0
            delta = rewards[row] + gamma * continuation * following_value - values[row]
            following_advantage = delta + gamma * lam * continuation * following_advantage
            advantage[row] = following_advantage
            following_value = values[row]
    return advantage, advantage + values


def batch(
    data: dict[str, NDArray[Any]],
    indices: Sequence[int] | NDArray[np.integer[Any]],
    device: str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Collate full autoregressive decisions into a padded prefix tensor batch."""
    query_indices = []
    owners = []
    first = []
    for owner, index in enumerate(indices):
        begin, end = data["decision_offsets"][index : index + 2]
        first.append(len(query_indices))
        query_indices.extend(range(begin, end))
        owners.extend([owner] * (end - begin))
    sizes = data["sizes"][query_indices]
    width = int(sizes.max())
    x = np.zeros((len(sizes), width, data["x"].shape[1]), dtype=np.float32)
    mask = np.zeros((len(sizes), width), dtype=bool)
    for row, q in enumerate(query_indices):
        begin, end = data["query_offsets"][q : q + 2]
        x[row, : end - begin] = data["x"][begin:end]
        mask[row, : end - begin] = True
    return dict(
        x=torch.as_tensor(x, device=device, dtype=dtype),
        mask=torch.as_tensor(mask, device=device),
        targets=torch.as_tensor(data["targets"][query_indices], device=device, dtype=torch.long),
        owners=torch.tensor(owners, device=device),
        first=torch.tensor(first, device=device),
        count=len(indices),
    )


def joint_stats(
    model: ObservationTransformer, b: dict[str, Any], temperature: float, distribution: bool = False
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Sum prefix log probabilities and entropy; use the first prefix for Value."""
    if distribution:
        logits, values, value_logits = model.forward_details(b["x"], b["mask"])
    else:
        logits, values = model.forward_with_value(b["x"], b["mask"])
    if values is None:
        raise ValueError("PPO requires a Value head")
    logp = (logits / temperature).log_softmax(-1)
    query_logp = logp.gather(1, b["targets"][:, None]).squeeze(1)
    query_entropy = -(logp.exp() * logp.masked_fill(~torch.isfinite(logits), 0.0)).sum(-1)
    total = logits.new_zeros(b["count"])
    stats = (
        total.index_add(0, b["owners"], query_logp),
        values[b["first"]],
        total.index_add(0, b["owners"], query_entropy),
    )
    return (*stats, value_logits[b["first"]]) if distribution else stats


@torch.no_grad()
def audit_behavior(
    model: ObservationTransformer,
    data: dict[str, NDArray[Any]],
    temperature: float,
    batch_size: int = 512,
    threads: int = 1,
) -> dict[str, Any]:
    """Verify recorded log probabilities, Values, and entropy against the frozen policy."""
    if batch_size < 1 or threads < 1:
        raise ValueError("audit batch size and threads must be positive")
    torch.set_num_threads(threads)
    model = model.cpu().double().eval()
    errors = dict(logp=0.0, value=0.0, entropy=0.0)
    decisions = len(data["old_logp"])
    batches = (decisions + batch_size - 1) // batch_size
    print(
        f"PPO rollout behavior audit: verifying {decisions} decisions with {threads} CPU threads",
        flush=True,
    )
    for batch_index, start in enumerate(range(0, decisions, batch_size), start=1):
        indices = np.arange(start, min(start + batch_size, len(data["old_logp"])))
        stats = joint_stats(model, batch(data, indices, "cpu", torch.float64), temperature)
        for name, prediction in zip(errors, stats, strict=True):
            expected = data["old_" + name][indices]
            np.testing.assert_allclose(prediction.numpy(), expected, atol=2e-5, rtol=2e-5)
            errors[name] = max(errors[name], float(np.max(np.abs(prediction.numpy() - expected))))
        if batch_index == batches or batch_index % max(1, batches // 10) == 0:
            print(
                f"PPO rollout behavior audit progress: {batch_index}/{batches} batches "
                f"({100 * batch_index / batches:.1f}%)",
                flush=True,
            )
    return dict(max_abs_errors=errors, atol=2e-5, rtol=2e-5, decisions=decisions, threads=threads)
