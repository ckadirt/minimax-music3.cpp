#!/usr/bin/env python3
"""Download and verify the pinned, inference-only MiniMax Music 3 snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

INVENTORY_PATH = Path(__file__).with_name("source_files.json")
CHUNK_BYTES = 8 * 1024 * 1024


def load_inventory() -> dict[str, Any]:
    with INVENTORY_PATH.open("r", encoding="utf-8") as handle:
        inventory = json.load(handle)
    paths = [item["path"] for item in inventory["files"]]
    if len(paths) != len(set(paths)):
        raise RuntimeError("source inventory contains duplicate paths")
    return inventory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, item: dict[str, Any]) -> bool:
    return path.is_file() and path.stat().st_size == item["size"] and sha256(path) == item["sha256"]


def request_headers(token: str | None, offset: int = 0) -> dict[str, str]:
    headers = {"User-Agent": "minimax-music3.cpp/0.1 checkpoint downloader"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return headers


def download_one(root: Path, repository: str, revision: str,
                 item: dict[str, Any], token: str | None) -> tuple[str, str]:
    destination = root / item["path"]
    if verified(destination, item):
        return item["path"], "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > item["size"]:
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    url = (f"https://huggingface.co/{repository}/resolve/{revision}/" +
           urllib.parse.quote(item["path"]))
    request = urllib.request.Request(url, headers=request_headers(token, offset))
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code == 416 and offset == item["size"]:
            response = None
        else:
            raise
    if response is not None:
        append = offset != 0 and getattr(response, "status", None) == 206
        with response, partial.open("ab" if append else "wb") as output:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
    if not partial.is_file() or partial.stat().st_size != item["size"]:
        actual = partial.stat().st_size if partial.exists() else 0
        raise RuntimeError(f"truncated {item['path']}: got {actual}, expected {item['size']}")
    actual_hash = sha256(partial)
    if actual_hash != item["sha256"]:
        raise RuntimeError(f"checksum mismatch for {item['path']}: {actual_hash}")
    os.replace(partial, destination)
    return item["path"], "downloaded"


def write_manifest(root: Path, inventory: dict[str, Any]) -> bool:
    files: list[dict[str, Any]] = []
    complete = True
    for item in inventory["files"]:
        path = root / item["path"]
        ok = verified(path, item)
        complete &= ok
        files.append({"path": item["path"], "size": item["size"],
                      "sha256": item["sha256"], "verified": ok})
    manifest = {"schema_version": 1, "repository": inventory["repository"],
                "revision": inventory["revision"], "complete": complete, "files": files}
    temporary = root / "source-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, root / "source-manifest.json")
    return complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Snapshot output directory")
    parser.add_argument("--jobs", type=int, default=3, help="Concurrent downloads (default: 3)")
    parser.add_argument("--metadata-only", action="store_true", help="Skip SafeTensors payloads")
    parser.add_argument("--verify-only", action="store_true", help="Do not access the network")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 16:
        raise ValueError("--jobs must be in [1, 16]")
    inventory = load_inventory()
    selected = [item for item in inventory["files"]
                if not args.metadata_only or not item["path"].endswith(".safetensors")]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not args.verify_only:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(download_one, args.output, inventory["repository"],
                                   inventory["revision"], item, token) for item in selected]
            for future in concurrent.futures.as_completed(futures):
                path, disposition = future.result()
                print(f"[{disposition}] {path}", flush=True)
    selected_ok = all(verified(args.output / item["path"], item) for item in selected)
    complete = write_manifest(args.output, inventory)
    print(f"[manifest] selected_verified={str(selected_ok).lower()} "
          f"complete={str(complete).lower()} elapsed={time.monotonic() - started:.1f}s")
    return 0 if selected_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
