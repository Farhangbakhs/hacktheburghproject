"""Download the face-tracking model files used by face_tracker.py (idempotent).

    .venv/Scripts/python models/download_models.py          # into this folder
    .venv/Scripts/python models/download_models.py --force  # re-download

Files are checked by exact size plus a hash taken from the upstream repository
(sha256 from the opencv_zoo Git-LFS pointer; git blob sha1 from the GitHub API
for lbfmodel.yaml). A file that already passes is skipped. Downloads go to a
.part file first, so an interrupted run never leaves a bad model behind.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (url, size in bytes, hash kind, hex digest)
MODELS = {
    # YuNet face detector (MIT licence), opencv_zoo. raw.githubusercontent only
    # serves the LFS pointer, so use the media (LFS) host.
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        232589,
        "sha256",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # FacemarkLBF 68-point (iBUG 300-W) model, the file the OpenCV facemark
    # tutorial points to (GSoC 2017, Laksono Kurnianggoro).
    "lbfmodel.yaml": (
        "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
        56375857,
        "gitsha1",
        "417d6db318df9867ebd1f8a0a7cf3a2a82eb1610",
    ),
}


def _digest(path: Path, kind: str) -> str:
    if kind == "sha256":
        h = hashlib.sha256()
    else:  # git blob sha1: sha1(b"blob <size>\0" + content)
        h = hashlib.sha1()
        h.update(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path, size: int, kind: str, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and _digest(path, kind) == digest


def download(name: str, dest_dir: Path = HERE, force: bool = False) -> Path:
    url, size, kind, digest = MODELS[name]
    dest = dest_dir / name
    if not force and is_valid(dest, size, kind, digest):
        print(f"[models] ok        {name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[models] fetching  {name} ({size / 1e6:.1f} MB) from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "evm-face-tracker"})
    with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            print(f"\r[models]   {got / 1e6:6.1f} / {size / 1e6:.1f} MB", end="", flush=True)
    print()
    if not is_valid(part, size, kind, digest):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: size/hash mismatch after download")
    part.replace(dest)
    print(f"[models] verified  {name}")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=HERE)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in MODELS:
        try:
            download(name, a.dir, a.force)
        except Exception as e:  # keep going so one bad mirror does not block the rest
            print(f"[models] FAILED    {name}: {e}", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
