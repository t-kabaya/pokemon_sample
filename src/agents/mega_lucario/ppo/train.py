"""One-pass clipped PPO update; behavior verification and complete per-seat GAE."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from agents.mega_lucario.ppo.rollout import audit_behavior, batch, compute_gae, joint_stats
from agents.mega_lucario.replay import sha256

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from agents.mega_lucario.transformer.model import ObservationTransformer


def load_rollout(
    directory: Path, contract_id: str, behavior_id: str
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Validate rollout identity and arrays before loading data for PPO."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["format"] != "ptcg-ppo-rollout-v1"
        or manifest["contract_id"] != contract_id
        or manifest["behavior_sha256"] != behavior_id
    ):
        raise ValueError("Rollout does not belong to this checkpoint/contract")
    if (
        sha256(directory / "rollout.npz") != manifest["arrays_sha256"]
        or sha256(directory / "decisions.json") != manifest["decisions_sha256"]
    ):
        raise ValueError("Rollout checksum mismatch")
    with np.load(directory / "rollout.npz", allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    if not np.isfinite(manifest["temperature"]) or manifest["temperature"] <= 0:
        raise ValueError("Invalid rollout temperature")
    for key in ("sizes", "targets", "decision_offsets", "query_offsets", "episode", "step"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError("Rollout indices must be integers")
    if data["done"].dtype != np.bool_:
        raise ValueError("Terminal mask must be boolean")
    n = len(data["old_logp"])
    q = len(data["sizes"])
    if manifest["decisions"] != n:
        raise ValueError("Rollout decision count mismatch")
    if (
        n < 1
        or data["decision_offsets"].shape != (n + 1,)
        or data["query_offsets"].shape != (q + 1,)
    ):
        raise ValueError("Invalid rollout offsets")
    if (
        data["decision_offsets"][0] != 0
        or data["decision_offsets"][-1] != q
        or np.any(np.diff(data["decision_offsets"]) < 1)
    ):
        raise ValueError("Invalid decision/query alignment")
    if (
        data["query_offsets"][0] != 0
        or data["query_offsets"][-1] != len(data["x"])
        or not np.array_equal(np.diff(data["query_offsets"]), data["sizes"])
    ):
        raise ValueError("Invalid feature/query alignment")
    if (
        data["targets"].shape != (q,)
        or np.any(data["sizes"] < 1)
        or np.any(data["targets"] < 0)
        or np.any(data["targets"] >= data["sizes"])
    ):
        raise ValueError("Invalid PPO action")
    for key in ("old_logp", "old_value", "old_entropy", "reward", "done", "episode", "step"):
        if data[key].shape != (n,):
            raise ValueError("Invalid decision scalar count")
    if any(
        not np.isfinite(data[key]).all()
        for key in ("x", "old_logp", "old_value", "old_entropy", "reward")
    ):
        raise ValueError("Non-finite PPO data")
    if (
        np.any(data["old_logp"] > 1e-8)
        or np.any(abs(data["old_value"]) > 1 + 1e-8)
        or np.any(data["old_entropy"] < 0)
    ):
        raise ValueError("Invalid behavior statistics")
    if (
        np.any(data["reward"][~data["done"]] != 0)
        or not np.isin(data["reward"][data["done"]], [-1, 0, 1]).all()
    ):
        raise ValueError("Invalid terminal rewards")
    return data, manifest


def clipped_loss(
    new_logp: torch.Tensor,
    value: torch.Tensor,
    entropy: torch.Tensor,
    old_logp: torch.Tensor,
    old_value: torch.Tensor,
    advantage: torch.Tensor,
    returns: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped policy and value objectives with entropy regularization."""
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantage, ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage
    ).mean()
    clipped_value = old_value + (value - old_value).clamp(
        -config["value_clip"], config["value_clip"]
    )
    value_loss = (
        0.5 * torch.maximum((value - returns).square(), (clipped_value - returns).square()).mean()
    )
    loss = (
        policy
        + config["value_coefficient"] * value_loss
        - config["entropy_coefficient"] * entropy.mean()
    )
    stats = dict(
        loss=loss,
        policy_loss=policy,
        value_loss=value_loss,
        entropy=entropy.mean(),
        approximate_kl=(ratio - 1 - log_ratio).mean(),
        clip_fraction=((ratio - 1).abs() > config["clip"]).float().mean(),
    )
    return loss, stats


def distribution_targets(returns: torch.Tensor) -> torch.Tensor:
    """Linear interpolation onto the 51-bin support, clamped at terminal bounds."""
    position = (returns.clamp(-1, 1) + 1) * 25
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    target = returns.new_zeros((len(returns), 51))
    target.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, upper[:, None], fraction[:, None])
    return target


def train(
    model: ObservationTransformer,
    payload: dict[str, Any],
    checkpoint: Path,
    rollout: Path,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Update PPO from a verified rollout, retaining held-out decisions for KL checks."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Update output must be empty")
    if config.get("epochs") != 1:
        raise ValueError("This PPO profile uses one epoch per fresh rollout")
    positive = ("batch_size", "learning_rate", "max_grad_norm", "updates")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive):
        raise ValueError("Invalid positive PPO settings")
    if (
        not 0 <= config["gamma"] <= 1
        or not 0 <= config["gae_lambda"] <= 1
        or not 0 < config["clip"] < 1
        or not 0 < config["value_clip"] < 1
    ):
        raise ValueError("Invalid PPO clip/GAE settings")
    if any(
        not np.isfinite(config[k]) or config[k] < 0
        for k in ("value_coefficient", "entropy_coefficient", "weight_decay")
    ):
        raise ValueError("Invalid loss coefficients")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    data, manifest = load_rollout(rollout, payload["contract_id"], sha256(checkpoint))
    if data["x"].ndim != 2 or data["x"].shape[1] != len(payload["feature_names"]):
        raise ValueError("PPO feature width mismatch")
    audit = audit_behavior(
        model, data, manifest["temperature"], threads=config.get("audit_threads", 1)
    )
    advantages, returns = compute_gae(
        data["old_value"],
        data["reward"],
        data["done"],
        data["episode"],
        data["step"],
        config["gamma"],
        config["gae_lambda"],
    )
    holdout_count = min(config.get("holdout_decisions", 10000), max(1, len(advantages) // 10))
    if len(advantages) < 2 or holdout_count < 1:
        raise ValueError("At least two decisions and positive holdout required")
    heldout = np.linspace(0, len(advantages) - 1, holdout_count, dtype=int)
    training_indices = np.setdiff1d(np.arange(len(advantages)), heldout)
    advantages = (advantages - advantages[training_indices].mean()) / max(
        float(advantages[training_indices].std()), 1e-8
    )
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # The PPO update runs on CUDA; keep PyTorch's CPU-side work from occupying audit threads.
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.float().to(device).train()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        for group in optimizer.param_groups:
            group.update(lr=config["learning_rate"], weight_decay=config["weight_decay"])
    order = rng.permutation(training_indices)
    reports = []
    used = 0
    max_updates = min(
        config["updates"], (len(order) + config["batch_size"] - 1) // config["batch_size"]
    )
    print(
        f"PPO update start: {max_updates} updates from {len(training_indices)} "
        f"training decisions on {device}",
        flush=True,
    )
    for start in range(0, len(order), config["batch_size"]):
        if len(reports) >= config["updates"]:
            break
        indices = order[start : start + config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        record = {}
        micro = config.get("micro_batch_size", 128)
        if micro < 1:
            raise ValueError("micro_batch_size must be positive")
        for chunk in range(0, len(indices), micro):
            rows = indices[chunk : chunk + micro]
            *stats, value_logits = joint_stats(
                model, batch(data, rows, device), manifest["temperature"], distribution=True
            )
            tensors = [
                torch.as_tensor(a[rows], device=device, dtype=torch.float32)
                for a in (data["old_logp"], data["old_value"], advantages, returns)
            ]
            loss, metrics = clipped_loss(*stats, *tensors, config)
            distribution_loss = (
                -(distribution_targets(tensors[-1]) * F.log_softmax(value_logits, -1))
                .sum(-1)
                .mean()
            )
            loss = loss + config["value_coefficient"] * distribution_loss
            metrics.update(loss=loss, distribution_loss=distribution_loss)
            if any(not torch.isfinite(v).all() for v in metrics.values()):
                raise ValueError("Non-finite PPO loss")
            fraction = len(rows) / len(indices)
            (loss * fraction).backward()
            for k, v in metrics.items():
                record[k] = record.get(k, 0.0) + float(v.detach()) * fraction
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        record["gradient_norm"] = float(norm)
        reports.append(record)
        used += len(indices)
        percent = 100 * len(reports) / max_updates
        print(
            f"PPO update progress: {len(reports)}/{max_updates} updates "
            f"({percent:.1f}%), decisions_used={used}/{len(training_indices)}",
            flush=True,
        )
    print("PPO heldout KL evaluation start", flush=True)
    with torch.no_grad():
        ratios = []
        log_deltas = []
        model.eval()
        for start in range(0, len(heldout), config.get("micro_batch_size", 128)):
            rows = heldout[start : start + config.get("micro_batch_size", 128)]
            new_logp = joint_stats(model, batch(data, rows, device), manifest["temperature"])[0]
            delta = new_logp - torch.as_tensor(
                data["old_logp"][rows], device=device, dtype=new_logp.dtype
            )
            ratios.extend((delta.exp() - 1 - delta).cpu().tolist())
            log_deltas.extend((-delta).cpu().tolist())
        global_kl = float(np.mean(ratios))
        sampled_forward_kl = float(np.mean(log_deltas))
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    delta = float(
        sum((state[k].double() - before[k].double()).square().sum() for k in state).sqrt()
    )
    if not reports or not np.isfinite(delta) or delta <= 0:
        raise ValueError("PPO did not make a finite weight update")
    report = dict(
        config=config,
        heldout_global_approximate_kl=global_kl,
        heldout_sampled_forward_kl=sampled_forward_kl,
        holdout_decisions=len(heldout),
        holdout_indices_sha256=hashlib.sha256(heldout.astype("<i8").tobytes()).hexdigest(),
        holdout_selection="uniform linspace in episode-step order; excluded from optimizer",
        training_decisions=len(training_indices),
        behavior_sha256=sha256(checkpoint),
        rollout_sha256=sha256(rollout / "manifest.json"),
        audit=audit,
        updates=len(reports),
        decisions_used=used,
        decisions_available=len(advantages),
        parameter_delta_l2=delta,
        optimizer_resumed="optimizer_state" in payload,
        mean_metrics={k: float(np.mean([r[k] for r in reports])) for k in reports[0]},
        last_metrics=reports[-1],
        opponent_distribution=manifest["collection"],
        hardware=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    )
    # Keep initialization provenance, but never present distillation metrics as PPO metrics.
    result = {
        k: payload[k]
        for k in (
            "format",
            "contract",
            "contract_id",
            "architecture",
            "feature_names",
            "deck_counts",
        )
    }
    result.update(
        stage="ppo",
        source_checkpoint_sha256=sha256(checkpoint),
        ppo=report,
        state_dict=state,
        optimizer_state=optimizer.state_dict(),
        total_updates=payload.get("total_updates", 0) + len(reports),
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, output / "model.pt")
    report["checkpoint_sha256"] = sha256(output / "model.pt")
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                updates=len(reports), parameter_delta_l2=delta, mean_metrics=report["mean_metrics"]
            )
        ),
        flush=True,
    )
    return report
