"""Upload the highest numbered local PPO cycle to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


def latest_local_cycle(output_dir: Path) -> tuple[str, Path]:
    """Return the highest complete local PPO cycle id and its models directory."""
    pattern = re.compile(r"^cycle-(\d{3})$")
    candidates: list[tuple[int, str, Path]] = []
    for path in output_dir.iterdir() if output_dir.exists() else []:
        match = pattern.match(path.name)
        models = path / "models"
        if (
            match
            and (models / "model.pt").is_file()
            and (models / "model.bin").is_file()
        ):
            cycle = match.group(1)
            candidates.append((int(cycle), cycle, models))
    if not candidates:
        raise ValueError(f"No complete PPO cycles found under {output_dir}/cycle-xxx/models")
    _, cycle, models = max(candidates)
    return cycle, models


def upload_cycle(repo_id: str, repo_type: str, remote_root: str, output_dir: Path) -> str:
    """Upload the latest complete local PPO cycle and return its cycle id."""
    cycle, models = latest_local_cycle(output_dir)
    api = HfApi()
    for filename in ("model.pt", "model.bin"):
        api.upload_file(
            path_or_fileobj=models / filename,
            path_in_repo=f"{remote_root}/cycle-{cycle}/{filename}",
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload PPO cycle-{cycle} {filename}",
        )
    print(f"Uploaded PPO cycle-{cycle} from {models} to {repo_id}/{remote_root}/cycle-{cycle}")
    return cycle


def main() -> None:
    """Parse arguments and upload the latest PPO cycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--remote-root", default="mega_lucario/ppo")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mega_lucario/ppo_training"))
    args = parser.parse_args()
    upload_cycle(args.repo_id, args.repo_type, args.remote_root, args.output_dir)


if __name__ == "__main__":
    main()
