"""Download the official Python API and optional source notebooks."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "src/agents"

CG_DATASET = "kiyotah/cg-lib"
CG_SHA256 = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "api.py": "593f1298e52a635f90f8f505a52113e9af114f444c293404e37906f18ee06ced",
    "game.py": "3bd3d4f4a369a11e6d2f5da9094cf15ebc410a2221835e6417b7cff4883f1fc2",
    "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
    "sim.py": "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655",
    "utils.py": "60f29665cee0a88525d6f0383bc45959a6262d16fe35ef380aece1e0ea13c49b",
}
OFFICIAL_AGENTS = (
    {
        "name": "mega_lucario",
        "deck_sha256": "406e2e9bd6ae82b8008b16ee64ffcbb58e4a50cd6bc36e33ae655456c6b9afee",
        "notebook": "kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck",
        "notebook_sha256": "8bd86060865cc6e9c225e49bd3276279ec2f00dddb27bb6b84a21955c26470b7",
    },
    {
        "name": "dragapult",
        "deck_sha256": "30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724",
        "notebook": "kiyotah/a-sample-rule-based-agent-dragapult-ex-deck",
        "notebook_sha256": "7fd3a2e6d6c36136bc2751bee8661d4bc66ac5a18135c31bfee6d50e02e6f698",
    },
    {
        "name": "iono",
        "deck_sha256": "e36d46c5bcafdef8a5d0e6caeb34dd8db09119c62d8fb67c99e89e7eed39f974",
        "notebook": "kiyotah/a-sample-rule-based-agent-iono-s-deck",
        "notebook_sha256": "05cbcb77ac5904e18ede4b489e6d7aed0d236edeb691ffac0a9c9ffc3eab2f58",
    },
    {
        "name": "mega_abomasnow",
        "deck_sha256": "7af2d7e111c084da535b89758730b3fd6cbb7c0543a9444499c5b61efdc8aecd",
        "notebook": "kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck",
        "notebook_sha256": "a1e82b219bec8905ad2593278c7b523c7d3c0ccdf8acf9f60d145b1de42a4468",
    },
)


def check(path: Path, expected: str) -> None:
    """Verify an external asset or bundled deck against its expected SHA-256 digest."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Asset SHA-256 mismatch: {path}")


def main() -> None:
    """Parse command-line arguments and run this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ROOT / "external/official_assets")
    parser.add_argument(
        "--notebooks", action="store_true", help="Also fetch source notebooks for parity checks"
    )
    parser.add_argument(
        "--require-hash-check",
        action="store_true",
        help="Require SHA-256 verification against the repository's fixed snapshot; default skips the check because Kaggle upstream files drift over time.",
    )
    args = parser.parse_args()
    kaggle = Path(sys.executable).parent / "kaggle"
    enforce_checks = args.require_hash_check
    for agent in OFFICIAL_AGENTS:
        if enforce_checks:
            check(AGENTS / agent["name"] / "deck.csv", agent["deck_sha256"])
    destination = args.asset_root / "cg-lib"
    subprocess.run(
        [str(kaggle), "datasets", "download", CG_DATASET, "--unzip", "--path", str(destination)],
        check=True,
    )
    if enforce_checks:
        for filename, expected in CG_SHA256.items():
            check(args.asset_root / "cg-lib/cg" / filename, expected)
    for agent in OFFICIAL_AGENTS:
        if args.notebooks:
            slug = agent["notebook"].split("/")[1]
            destination = ROOT / "external/official_notebooks" / slug
            subprocess.run(
                [
                    str(kaggle),
                    "kernels",
                    "pull",
                    agent["notebook"],
                    "--metadata",
                    "--path",
                    str(destination),
                ],
                check=True,
            )
            if enforce_checks:
                check(destination / f"{slug}.ipynb", agent["notebook_sha256"])
    if enforce_checks:
        print("Verified official Python API and bundled decks.")
    else:
        print("Downloaded official assets without SHA-256 verification; use --require-hash-check for fixed-snapshot validation.")


if __name__ == "__main__":
    main()
