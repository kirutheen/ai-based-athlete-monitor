#!/usr/bin/env python
"""Download a MediaPipe Pose Landmarker model bundle into models/.

Usage:
    python scripts/download_pose_model.py            # downloads the 'full' model
    python scripts/download_pose_model.py --variant lite
    python scripts/download_pose_model.py --variant heavy
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list(URLS), default="full")
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = URLS[args.variant]
    dest = out_dir / f"pose_landmarker_{args.variant}.task"

    print(f"Downloading {url}\n  -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 - want a clear message for any network failure
        print(f"\nDownload failed: {exc}")
        print("If your network blocks storage.googleapis.com, download the file manually")
        print(f"from the URL above and save it to: {dest}")
        sys.exit(1)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Done. Saved {size_mb:.1f} MB to {dest}")


if __name__ == "__main__":
    main()
