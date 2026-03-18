#!/usr/bin/env python3
"""Clone PriMock57 repo (with LFS), copy 57 consultation WAVs to datasets/primock57_raw/audio_consultations/. Requires git and git-lfs."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/babylonhealth/primock57.git"
STAGING_DIR = ROOT / "datasets" / "primock57_repo_staging"
AUDIO_SOURCE = STAGING_DIR / "audio"
RAW_ROOT = ROOT / "datasets" / "primock57_raw"
AUDIO_DEST = RAW_ROOT / "audio_consultations"


def run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        env=env or None,
        capture_output=True,
        text=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Download PriMock57 audio via Git LFS clone.")
    parser.add_argument(
        "--no-keep",
        action="store_true",
        help="Remove the staging clone after copying (default: keep it).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=STAGING_DIR,
        help=f"Directory to clone into (default: {STAGING_DIR}).",
    )
    args = parser.parse_args()
    staging = args.staging_dir.resolve()

    # 1. Check git and git-lfs
    r = run(["git", "--version"])
    if r.returncode != 0:
        print("Error: git not found. Install git.", file=sys.stderr)
        sys.exit(1)
    r = run(["git", "lfs", "version"])
    if r.returncode != 0:
        print(
            "Error: git-lfs not found. Install it: https://git-lfs.github.com/\n"
            "  Fedora: dnf install git-lfs\n"
            "  macOS:  brew install git-lfs\n"
            "Then run: git lfs install",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Clone or pull
    if not staging.exists():
        staging.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {REPO_URL} into {staging} (this may take a few minutes)...")
        r = run(["git", "clone", REPO_URL, str(staging)])
        if r.returncode != 0:
            print(f"Clone failed: {r.stderr or r.stdout}", file=sys.stderr)
            sys.exit(1)
        print("Clone done. Pulling LFS files...")
        r = run(["git", "lfs", "pull"], cwd=staging)
        if r.returncode != 0:
            print(f"git lfs pull failed: {r.stderr or r.stdout}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Staging dir exists: {staging}. Pulling latest and LFS...")
        r = run(["git", "pull"], cwd=staging)
        run(["git", "lfs", "pull"], cwd=staging)

    audio_src = staging / "audio"
    if not audio_src.exists():
        print(f"Error: No audio/ folder in {staging}", file=sys.stderr)
        sys.exit(1)

    wavs = list(audio_src.glob("*.wav"))
    if not wavs:
        print("Warning: No .wav files in audio/. Did git lfs pull run correctly? Check for LFS pointer files.", file=sys.stderr)
    else:
        # Check for LFS pointer (real WAVs are much larger than 200 bytes)
        small = [f for f in wavs if f.stat().st_size < 200]
        if small:
            print(
                f"Warning: {len(small)} files are very small (likely LFS pointers). "
                "Run in the staging dir: git lfs pull",
                file=sys.stderr,
            )

    # 3. Copy to project layout
    AUDIO_DEST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in wavs:
        dest = AUDIO_DEST / f.name
        if not dest.exists() or dest.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dest)
            copied += 1
    print(f"Copied {copied} files to {AUDIO_DEST} (total WAVs in repo: {len(wavs)}).")

    # 4. Optional: remove staging
    if args.no_keep and staging.exists():
        print(f"Removing staging clone {staging}...")
        shutil.rmtree(staging)

    print("Done. You can run: python -m experiments.primock57_full_57.c1_whisper_57")


if __name__ == "__main__":
    main()
