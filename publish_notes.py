"""
This script can be run with:
```
python publish_notes.py \
    --input-dir ~/Library/CloudStorage/Dropbox/obsidian/main-vaults/ \
    --output-dir ~/repositories/quartz/content/ \
    --dry-run
```
"""
import argparse
import subprocess
import argparse
import sys
from typing import Any


FOLDERS_TO_SYNC = ["AI-Notes", "Research-Papers", "Statistics"]
IGNORE_FILE_TYPES = ["*.csv", "*.pdf", "*.skim"]
RSYNC_FLAGS = [
    "--verbose",
    "--recursive",
    "--update",
]


def construct_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Input directory to sync from")
    parser.add_argument("--output-dir", help="Output directory to sync to")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run")
    return parser


def sync(input_dir: str, output_dir: str, dry_run: bool) -> None:
    """
    Syncs the input directory to the output directory
    """
    flags = " ".join(RSYNC_FLAGS)

    ignore_files = ""
    for file_type in IGNORE_FILE_TYPES:
        ignore_files += f"--exclude={file_type} "

    for folder in FOLDERS_TO_SYNC:
        command = f"rsync {flags} {ignore_files}{input_dir}/{folder} {output_dir}"
        if dry_run:
            command += " --dry-run"
        subprocess.run(command, shell=True)


def main(argv: Any) -> int:
    args = construct_parser().parse_args(argv)
    sync(args.input_dir, args.output_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
