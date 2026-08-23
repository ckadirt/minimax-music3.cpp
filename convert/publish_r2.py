#!/usr/bin/env python3
"""Validate and immutably upload the curated Cantor tiers to Cloudflare R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "publish" / "cloudflare-r2.json"
EXPECTED_ROLES = ("lm", "rvq", "condition", "dit", "vae")
EXPECTED_TIERS = {
    "1.0-fast": {"lm": "Q4_K_M", "rvq": "Q4_K_M", "condition": "F32", "dit": "Q4_K_M", "vae": "F16"},
    "1.0-balanced": {"lm": "Q6_K", "rvq": "Q6_K", "condition": "F32", "dit": "Q6_K", "vae": "F16"},
    "1.0-quality": {"lm": "BF16", "rvq": "BF16", "condition": "F32", "dit": "F32", "vae": "F32"},
}
CHUNK_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"manifest {key} must be a non-empty string")
    return value


def validate_manifest(path: Path, artifacts: Path | None = None,
                      verify_content: bool = False) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("model") != "minimax-music3":
        raise RuntimeError("unsupported Cloudflare manifest schema/model")
    if manifest.get("engine") != "minimax-music3" or manifest.get("version") != "1.0":
        raise RuntimeError("Cloudflare manifest engine/version mismatch")
    prefix = required_string(manifest, "prefix")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", prefix):
        raise RuntimeError("Cloudflare manifest prefix is not a safe immutable key")
    base = required_string(manifest, "public_base_url").rstrip("/")
    if not base.startswith("https://"):
        raise RuntimeError("Cloudflare public base URL must use HTTPS")
    if manifest.get("manifest_url") != f"{base}/{prefix}/manifest.json":
        raise RuntimeError("Cloudflare manifest URL does not match its prefix")

    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("Cloudflare manifest has no objects")
    by_filename: dict[str, dict[str, Any]] = {}
    for record in objects:
        if not isinstance(record, dict):
            raise RuntimeError("Cloudflare object record is not an object")
        filename = required_string(record, "filename")
        if Path(filename).name != filename or not filename.endswith(".gguf"):
            raise RuntimeError(f"unsafe Cloudflare object filename: {filename}")
        if filename in by_filename:
            raise RuntimeError(f"duplicate Cloudflare object: {filename}")
        digest = required_string(record, "blob")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"malformed Cloudflare object digest: {filename}")
        if record.get("url") != f"{base}/{prefix}/{filename}":
            raise RuntimeError(f"Cloudflare object URL mismatch: {filename}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise RuntimeError(f"Cloudflare object size is invalid: {filename}")
        if record.get("role") not in EXPECTED_ROLES or not isinstance(record.get("quant"), str):
            raise RuntimeError(f"Cloudflare object role/quant is invalid: {filename}")
        by_filename[filename] = record
        if artifacts is not None:
            gguf = artifacts / filename
            sidecar = artifacts / f"{filename}.manifest.json"
            if not gguf.is_file() or not sidecar.is_file():
                raise RuntimeError(f"missing staged Cloudflare object: {filename}")
            source = json.loads(sidecar.read_text(encoding="utf-8"))
            expected = source.get("gguf")
            actual = {"filename": filename, "size": record["bytes"], "sha256": digest[7:]}
            if expected != actual:
                raise RuntimeError(f"conversion manifest mismatch for Cloudflare object: {filename}")
            if gguf.stat().st_size != record["bytes"]:
                raise RuntimeError(f"staged Cloudflare object size mismatch: {filename}")
            if verify_content and sha256(gguf) != digest[7:]:
                raise RuntimeError(f"staged Cloudflare object digest mismatch: {filename}")

    variants = manifest.get("variants")
    if not isinstance(variants, list) or {item.get("tag") for item in variants} != set(EXPECTED_TIERS):
        raise RuntimeError("Cloudflare manifest must contain fast, balanced, and quality tiers")
    uses: dict[str, set[str]] = {filename: set() for filename in by_filename}
    for variant in variants:
        tag = required_string(variant, "tag")
        components = variant.get("components")
        if not isinstance(components, list) or tuple(item.get("role") for item in components) != EXPECTED_ROLES:
            raise RuntimeError(f"Cloudflare tier {tag} does not have the five ordered ABI roles")
        actual_quants = {item["role"]: item.get("quant") for item in components}
        if actual_quants != EXPECTED_TIERS[tag]:
            raise RuntimeError(f"Cloudflare tier {tag} does not match the approved quality ladder")
        total = 0
        for component in components:
            filename = required_string(component, "filename")
            source = by_filename.get(filename)
            if source is None:
                raise RuntimeError(f"Cloudflare tier {tag} references an unknown object: {filename}")
            for key in ("role", "quant", "blob", "url", "bytes"):
                if component.get(key) != source.get(key):
                    raise RuntimeError(f"Cloudflare tier {tag}/{filename} has inconsistent {key}")
            uses[filename].add(tag)
            total += component["bytes"]
        if variant.get("total_bytes") != total:
            raise RuntimeError(f"Cloudflare tier {tag} total byte count is incorrect")
    for filename, tags in uses.items():
        if by_filename[filename].get("used_by") != sorted(tags, key=list(EXPECTED_TIERS).index):
            raise RuntimeError(f"Cloudflare object use list is incorrect: {filename}")
    if manifest.get("unique_model_bytes") != sum(item["bytes"] for item in objects):
        raise RuntimeError("Cloudflare unique model byte count is incorrect")

    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, list) or [item.get("filename") for item in sidecars] != ["MODEL_LICENSE"]:
        raise RuntimeError("Cloudflare manifest must publish the model license")
    license_record = sidecars[0]
    if license_record.get("url") != f"{base}/{prefix}/MODEL_LICENSE":
        raise RuntimeError("Cloudflare model-license URL is incorrect")
    if re.fullmatch(r"[0-9a-f]{64}", str(license_record.get("sha256", ""))) is None:
        raise RuntimeError("Cloudflare model-license digest is malformed")
    if artifacts is not None:
        license_path = artifacts / "MODEL_LICENSE"
        if not license_path.is_file() or license_path.stat().st_size != license_record.get("bytes"):
            raise RuntimeError("staged Cloudflare model license size mismatch")
        if verify_content and sha256(license_path) != license_record["sha256"]:
            raise RuntimeError("staged Cloudflare model license digest mismatch")
    return manifest


def aws_environment() -> tuple[dict[str, str], str, str]:
    required = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "R2_BUCKET")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing Cloudflare R2 environment: " + ", ".join(missing))
    environment = os.environ.copy()
    environment.update({
        "AWS_ACCESS_KEY_ID": os.environ["R2_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["R2_SECRET_ACCESS_KEY"],
        "AWS_DEFAULT_REGION": "auto",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
    })
    return environment, os.environ["R2_ENDPOINT_URL"], os.environ["R2_BUCKET"]


def aws_json(arguments: list[str], environment: dict[str, str], allow_missing: bool = False) -> dict[str, Any] | None:
    completed = subprocess.run(
        ["aws", *arguments, "--output", "json"], env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode == 0:
        return json.loads(completed.stdout or "{}")
    if allow_missing and any(marker in completed.stderr for marker in ("(404)", "Not Found", "NoSuchKey")):
        return None
    raise RuntimeError("Cloudflare R2 request failed: " + completed.stderr.strip())


def head_object(key: str, environment: dict[str, str], endpoint: str, bucket: str) -> dict[str, Any] | None:
    return aws_json(["s3api", "head-object", "--endpoint-url", endpoint,
                     "--bucket", bucket, "--key", key], environment, allow_missing=True)


def upload_one(source: Path, key: str, size: int, digest: str, content_type: str,
               environment: dict[str, str], endpoint: str, bucket: str) -> str:
    existing = head_object(key, environment, endpoint, bucket)
    if existing is not None:
        metadata = {str(k).lower(): str(v) for k, v in existing.get("Metadata", {}).items()}
        if existing.get("ContentLength") != size or metadata.get("sha256") != digest:
            raise RuntimeError(f"refusing to replace non-matching immutable R2 object: {key}")
        print(f"[cached] {key} ({size} bytes)", flush=True)
        return "cached"
    completed = subprocess.run([
        "aws", "s3", "cp", str(source), f"s3://{bucket}/{key}",
        "--endpoint-url", endpoint, "--only-show-errors",
        "--cache-control", "public, max-age=31536000, immutable",
        "--content-type", content_type, "--metadata", f"sha256={digest}",
    ], env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"failed to upload Cloudflare R2 object: {key}")
    remote = head_object(key, environment, endpoint, bucket)
    metadata = {} if remote is None else {str(k).lower(): str(v) for k, v in remote.get("Metadata", {}).items()}
    if remote is None or remote.get("ContentLength") != size or metadata.get("sha256") != digest:
        raise RuntimeError(f"Cloudflare R2 post-upload verification failed: {key}")
    print(f"[uploaded] {key} ({size} bytes)", flush=True)
    return "uploaded"


def verify_public(records: list[tuple[str, int]]) -> None:
    pending = dict(records)
    for attempt in range(1, 7):
        for url, expected_size in list(pending.items()):
            try:
                request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "minimax-music3.cpp/r2-verifier"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    actual_size = int(response.headers.get("Content-Length", "-1"))
                    if response.status == 200 and actual_size == expected_size:
                        del pending[url]
            except (OSError, ValueError, urllib.error.URLError):
                pass
        if not pending:
            return
        if attempt != 6:
            print(f"[public] waiting for {len(pending)} object(s), attempt {attempt}/6", flush=True)
            time.sleep(5)
    raise RuntimeError("Cloudflare public verification failed: " + ", ".join(sorted(pending)))


def publish(manifest_path: Path, manifest: dict[str, Any], artifacts: Path) -> None:
    environment, endpoint, bucket = aws_environment()
    prefix = manifest["prefix"]
    public: list[tuple[str, int]] = []
    uploaded = 0
    cached = 0
    for record in manifest["objects"]:
        result = upload_one(artifacts / record["filename"], f"{prefix}/{record['filename']}",
                            record["bytes"], record["blob"][7:], "application/octet-stream",
                            environment, endpoint, bucket)
        uploaded += result == "uploaded"
        cached += result == "cached"
        public.append((record["url"], record["bytes"]))
    for record in manifest["sidecars"]:
        result = upload_one(artifacts / record["filename"], f"{prefix}/{record['filename']}",
                            record["bytes"], record["sha256"], "text/plain; charset=utf-8",
                            environment, endpoint, bucket)
        uploaded += result == "uploaded"
        cached += result == "cached"
        public.append((record["url"], record["bytes"]))
    manifest_bytes = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    result = upload_one(manifest_path, f"{prefix}/manifest.json", len(manifest_bytes), manifest_digest,
                        "application/json; charset=utf-8", environment, endpoint, bucket)
    uploaded += result == "uploaded"
    cached += result == "cached"
    public.append((manifest["manifest_url"], len(manifest_bytes)))
    verify_public(public)
    print(json.dumps({"status": "pass", "uploaded": uploaded, "cached": cached,
                      "public_verified": len(public), "manifest_url": manifest["manifest_url"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--upload", action="store_true", help="Mutate R2 after full local verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = validate_manifest(args.manifest, args.artifacts, verify_content=True)
    print(f"verified {len(manifest['objects'])} unique tier objects "
          f"({manifest['unique_model_bytes']} bytes) across {len(manifest['variants'])} variants")
    if not args.upload:
        print("dry-run: no Cloudflare R2 objects were created")
        return 0
    publish(args.manifest, manifest, args.artifacts)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
