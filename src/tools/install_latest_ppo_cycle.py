"""Install the highest numbered PPO cycle from a Hugging Face model repo."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def replace_file(source: str, target: Path) -> None:
    """Copy source to target, replacing read-only files when needed."""
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    shutil.copy2(source, target)


def complete_cycles(repo_id: str, repo_type: str, remote_root: str) -> list[str]:
    """Return complete cycle ids sorted from oldest to newest."""
    pattern = re.compile(rf"^{re.escape(remote_root)}/cycle-(\d{{3}})/(model\.(?:pt|bin))$")
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    cycles: dict[str, set[str]] = {}
    for path in files:
        match = pattern.match(path)
        if match:
            cycles.setdefault(match.group(1), set()).add(match.group(2))
    complete = [cycle for cycle, names in cycles.items() if {"model.pt", "model.bin"} <= names]
    if not complete:
        raise ValueError(f"No complete PPO cycles found under {remote_root}/cycle-xxx")
    return sorted(complete, key=int)


def copy_cycle(
    repo_id: str,
    repo_type: str,
    remote_root: str,
    output_dir: Path,
    cycle: str,
    alias: str | None,
) -> Path:
    """Download one PPO cycle and optionally copy it to an alias directory."""
    target = output_dir / f"cycle-{cycle}" / "models"
    target.mkdir(parents=True, exist_ok=True)
    alias_target = output_dir / alias / "models" if alias else None
    if alias_target is not None:
        alias_target.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pt", "model.bin"):
        remote_path = f"{remote_root}/cycle-{cycle}/{filename}"
        downloaded = hf_hub_download(repo_id=repo_id, filename=remote_path, repo_type=repo_type)
        replace_file(downloaded, target / filename)
        if alias_target is not None:
            replace_file(downloaded, alias_target / filename)
    print(f"Installed PPO cycle-{cycle} to {target}")
    if alias_target is not None:
        print(f"Installed PPO {alias} alias to {alias_target}")
    return target


def install_cycle(
    repo_id: str,
    repo_type: str,
    remote_root: str,
    output_dir: Path,
    alias: str | None,
    previous_alias: str | None,
) -> Path:
    """Download the latest PPO cycle and optionally the previous cycle."""
    cycles = complete_cycles(repo_id, repo_type, remote_root)
    if previous_alias is not None and len(cycles) < 2:
        raise ValueError("At least two complete PPO cycles are required for previous alias")
    if previous_alias is not None:
        copy_cycle(repo_id, repo_type, remote_root, output_dir, cycles[-2], previous_alias)
    return copy_cycle(repo_id, repo_type, remote_root, output_dir, cycles[-1], alias)


def main() -> None:
    """Parse arguments and install the latest PPO cycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--remote-root", default="mega_lucario/ppo")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mega_lucario/ppo_training"))
    parser.add_argument("--alias", default="latest")
    parser.add_argument("--previous-alias")
    args = parser.parse_args()
    install_cycle(
        args.repo_id,
        args.repo_type,
        args.remote_root,
        args.output_dir,
        args.alias,
        args.previous_alias,
    )


if __name__ == "__main__":
    main()
