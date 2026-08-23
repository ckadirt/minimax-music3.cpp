#!/usr/bin/env python3
"""Verify and immutably publish the complete GGUF matrix to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "ckadirt/MiniMax-Music3-GGUF"
SOURCE_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
GGML_REVISION = "70081fdfc8685b60477b54d9d11cd679c5a00cb1"
EXPECTED = {
    "lm": ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "rvq": ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "condition": ("F32",),
    "dit": ("F32", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "vae": ("F32", "F16"),
}
SIDECARS = (
    "MODEL_LICENSE",
    "config.json",
    "config/language_model.json",
    "config/rvq_depth_decoder.json",
    "config/condition_encoder.json",
    "config/transformer.json",
    "config/vocoder.json",
    "config/scheduler.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_ggufs() -> list[str]:
    return [f"{component}-{profile}.gguf"
            for component, profiles in EXPECTED.items() for profile in profiles]


def validate_artifacts(folder: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    expected = set(expected_ggufs())
    actual = {path.name for path in folder.glob("*.gguf")}
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise RuntimeError(f"GGUF matrix mismatch (missing: {missing}; extra: {extra})")
    if any(folder.glob("*.tmp")):
        raise RuntimeError("conversion temporary files are present")

    inventory_hash = sha256(ROOT / "convert" / "source_files.json")
    for component, profiles in EXPECTED.items():
        for profile in profiles:
            filename = f"{component}-{profile}.gguf"
            gguf = folder / filename
            manifest_path = folder / f"{filename}.manifest.json"
            checksum_path = folder / f"{filename}.sha256"
            if not manifest_path.is_file() or not checksum_path.is_file():
                raise RuntimeError(f"missing manifest/checksum sidecar for {filename}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or manifest.get("converter_version") != "1":
                raise RuntimeError(f"unsupported conversion manifest for {filename}")
            required = {
                "component": component,
                "profile": profile,
                "source_repository": "MiniMaxAI/MiniMax-Music3",
                "source_revision": SOURCE_REVISION,
                "source_inventory_sha256": inventory_hash,
                "ggml_revision": GGML_REVISION,
            }
            for key, expected_value in required.items():
                if manifest.get(key) != expected_value:
                    raise RuntimeError(f"{filename}: manifest {key} does not match the release contract")
            digest = sha256(gguf)
            gguf_record = manifest.get("gguf", {})
            if gguf_record != {"filename": filename, "size": gguf.stat().st_size, "sha256": digest}:
                raise RuntimeError(f"{filename}: manifest size/hash does not match GGUF bytes")
            checksum = checksum_path.read_text(encoding="ascii")
            if checksum != f"{digest}  {filename}\n":
                raise RuntimeError(f"{filename}: checksum sidecar is not canonical")
            manifests[filename] = manifest
    for name in SIDECARS:
        if not (folder / name).is_file():
            raise RuntimeError(f"required publication sidecar is missing: {name}")
    return manifests


def validate_release_matrix(path: Path) -> None:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != 1 or matrix.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("release matrix schema/source revision does not match the publisher")
    gates = matrix.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise RuntimeError("release matrix has no validation gates")
    incomplete = sorted(name for name, record in gates.items()
                        if not isinstance(record, dict) or record.get("status") != "pass" or
                        not isinstance(record.get("evidence"), str) or not record["evidence"].strip())
    if not matrix.get("release_approved") or incomplete:
        detail = ", ".join(incomplete) if incomplete else "release_approved=false"
        raise RuntimeError("model publication is gated by incomplete validation: " + detail)


def render_model_card(manifests: dict[str, dict[str, Any]]) -> str:
    lines = ["| Artifact | Bytes | SHA-256 |", "| --- | ---: | --- |"]
    for filename in expected_ggufs():
        record = manifests[filename]["gguf"]
        lines.append(f"| `{filename}` | {record['size']:,} | `{record['sha256']}` |")
    template = (ROOT / "publish" / "README.md").read_text(encoding="utf-8")
    marker = "{{ARTIFACT_TABLE}}"
    if template.count(marker) != 1:
        raise RuntimeError("model-card template must contain exactly one artifact-table marker")
    return template.replace(marker, "\n".join(lines))


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_publication(folder: Path, stage: Path, manifests: dict[str, dict[str, Any]]) -> set[str]:
    names: set[str] = {"README.md"}
    (stage / "README.md").write_text(render_model_card(manifests), encoding="utf-8")
    for filename in expected_ggufs():
        for relative in (filename, f"{filename}.manifest.json", f"{filename}.sha256"):
            link_or_copy(folder / relative, stage / relative)
            names.add(relative)
    for relative in SIDECARS:
        link_or_copy(folder / relative, stage / relative)
        names.add(relative)
    return names


def remote_lfs_sha(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def verify_remote(api: Any, repo_id: str, token: str, expected_names: set[str],
                  manifests: dict[str, dict[str, Any]], artifacts: Path) -> None:
    from huggingface_hub import hf_hub_download

    info = api.model_info(repo_id, revision="main", files_metadata=True, token=token)
    siblings = {item.rfilename: item for item in info.siblings}
    missing = expected_names - set(siblings)
    if missing:
        raise RuntimeError("published repository is missing: " + ", ".join(sorted(missing)))
    for filename, manifest in manifests.items():
        sibling = siblings[filename]
        expected = manifest["gguf"]
        if sibling.size != expected["size"]:
            raise RuntimeError(f"remote size mismatch for {filename}")
        remote_sha = remote_lfs_sha(sibling)
        if remote_sha is None or remote_sha != expected["sha256"]:
            raise RuntimeError(f"remote LFS SHA-256 mismatch for {filename}")

    with tempfile.TemporaryDirectory(prefix="minimax-hf-verify-") as temporary:
        verification_root = Path(temporary)
        verify_names = ["README.md", *SIDECARS]
        verify_names += [f"{name}.manifest.json" for name in expected_ggufs()]
        verify_names += [f"{name}.sha256" for name in expected_ggufs()]
        for relative in verify_names:
            downloaded = Path(hf_hub_download(
                repo_id=repo_id, filename=relative, repo_type="model", revision="main",
                local_dir=verification_root, force_download=True, token=token))
            if relative == "README.md":
                expected_bytes = render_model_card(manifests).encode("utf-8")
                if downloaded.read_bytes() != expected_bytes:
                    raise RuntimeError("remote model card does not match the staged model card")
            else:
                if sha256(downloaded) != sha256(artifacts / relative):
                    raise RuntimeError(f"remote sidecar verification failed for {relative}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--validation", type=Path, default=ROOT / "validation" / "release-matrix.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = validate_artifacts(args.artifacts)
    total = sum((args.artifacts / name).stat().st_size for name in expected_ggufs())
    print(f"verified {len(manifests)} GGUF artifacts ({total} bytes) for {args.repo}")
    if args.dry_run:
        print("dry-run: no repository or upload was created")
        return 0

    validate_release_matrix(args.validation)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError as error:
        raise RuntimeError("install huggingface_hub==1.1.7 with hf_xet support") from error

    api = HfApi(token=token)
    try:
        existing = set(api.list_repo_files(args.repo, repo_type="model", token=token))
    except RepositoryNotFoundError:
        existing = set()
    with tempfile.TemporaryDirectory(prefix="minimax-hf-stage-") as temporary:
        stage = Path(temporary)
        names = stage_publication(args.artifacts, stage, manifests)
        collisions = existing & names
        if collisions:
            raise RuntimeError("refusing to overwrite existing Hub files: " + ", ".join(sorted(collisions)))
        api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True, token=token)
        api.upload_folder(
            repo_id=args.repo, repo_type="model", folder_path=stage,
            commit_message="Publish verified MiniMax Music 3 GGUF matrix", token=token)
        verify_remote(api, args.repo, token, names, manifests, args.artifacts)
    print(f"published and verified https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
