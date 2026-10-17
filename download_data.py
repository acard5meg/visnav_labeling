import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default=Path("."),
        type=Path,
        help="Directory to use for the HuggingFace cache (defaults to the current working directory).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download the test dataset for labeling instead of the full dataset.",
    )
    args = parser.parse_args()

    if args.test:
        print("Downloading the test dataset for labeling")
        repo_id = "acard/navdata_test"
    else:
        print("Downloading the full dataset for labeling")
        repo_id = "acard/raw_navdata"

    snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=str(args.cache_dir.resolve()))


if __name__ == "__main__":
    main()