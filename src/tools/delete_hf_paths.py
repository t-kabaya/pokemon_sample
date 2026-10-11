"""Delete selected files or folders from a Hugging Face Hub repository."""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def main() -> None:
    """Parse command-line arguments and delete requested Hub paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Repository id such as user/name.")
    parser.add_argument("paths", nargs="+", help="Files or folders to delete inside the repo.")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--commit-message", default="Delete generated artifact paths")
    args = parser.parse_args()

    api = HfApi()
    for path in args.paths:
        try:
            if path.endswith("/"):
                api.delete_folder(
                    path.rstrip("/"),
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    commit_message=args.commit_message,
                )
            else:
                api.delete_file(
                    path,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    commit_message=args.commit_message,
                )
        except HfHubHTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                print(f"Already absent: {path}")
                continue
            raise
        print(f"Deleted: {path}")


if __name__ == "__main__":
    main()
