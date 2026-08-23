#!/usr/bin/env python3
"""Convert the pinned MiniMax snapshot into the reviewed component GGUF matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_INVENTORY = ROOT / "convert" / "source_files.json"
MODEL_SPEC = ROOT / "model_specs" / "minimax_music3.json"
CONVERTER_VERSION = "1"

SIDECARS = {
    "config.json": "config.json",
    "language_model/config.json": "config/language_model.json",
    "rvq_depth_decoder/config.json": "config/rvq_depth_decoder.json",
    "condition_encoder/config.json": "config/condition_encoder.json",
    "transformer/config.json": "config/transformer.json",
    "vocoder/config.json": "config/vocoder.json",
    "scheduler/scheduler_config.json": "config/scheduler.json",
    "tokenizer/tokenizer.json": "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json": "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja": "tokenizer/chat_template.jinja",
    "LICENSE": "MODEL_LICENSE",
}


@dataclass(frozen=True)
class Component:
    name: str
    source: str
    output_stem: str


COMPONENTS = {
    "lm": Component("lm", "language_model/model.safetensors.index.json", "lm"),
    "rvq": Component("rvq", "rvq_depth_decoder/diffusion_pytorch_model.safetensors", "rvq"),
    "condition": Component("condition", "condition_encoder/diffusion_pytorch_model.safetensors", "condition"),
    "dit": Component("dit", "transformer/diffusion_pytorch_model.safetensors.index.json", "dit"),
    "vae": Component("vae", "vocoder/diffusion_pytorch_model.safetensors", "vae"),
}
PROFILES = ("F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
COMPONENT_PROFILES = {
    "lm": ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "rvq": ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "condition": ("F32",),
    "dit": ("F32", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"),
    "vae": ("F32", "F16"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_set(value: str, known: Iterable[str], label: str) -> list[str]:
    known_list = list(known)
    if value.lower() == "all":
        return known_list
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(requested) - set(known_list)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return [item for item in known_list if item in requested]


def promotion_overrides(component: str, profile: str) -> list[str]:
    target = {"Q8_0": "q8_0", "Q6_K": "q6_k", "Q5_K_M": "q6_k", "Q4_K_M": "q6_k"}[profile]
    output: list[str] = []
    if component == "lm":
        output.extend([f"model.embed_tokens.weight={target}", f"lm_head.weight={target}"])
        for layer in range(36):
            output.extend([
                f"model.layers.{layer}.self_attn.o_proj.weight={target}",
                f"model.layers.{layer}.mlp.down_proj.weight={target}",
            ])
    elif component == "rvq":
        output.extend(["pos_embedding.weight=f32", f"audio_embeddings.weight={target}"])
        output.extend(f"audio_heads.{head}.weight={target}" for head in range(7))
        for layer in range(4):
            output.extend([
                f"layers.{layer}.attn.to_out.weight={target}",
                f"layers.{layer}.down_proj.weight={target}",
            ])
    elif component == "dit":
        output.extend([
            "preprocess_conv*=f32",
            "postprocess_conv*=f32",
            "time_proj.weight=f32",
            "time_embed.linear_1.weight=f32",
            "time_embed.linear_2.weight=f32",
        ])
        for layer in range(36):
            output.extend([
                f"transformer_blocks.{layer}.attn.to_out.0.weight={target}",
                f"transformer_blocks.{layer}.ff_out.weight={target}",
            ])
    return output


def recipe(component: str, profile: str) -> tuple[str, list[str], str]:
    if component in {"lm", "rvq"}:
        if profile == "BF16":
            return "bf16", [], profile
        base = {"Q8_0": "q8_0", "Q6_K": "q6_k", "Q5_K_M": "q5_k", "Q4_K_M": "q4_k"}[profile]
        return base, promotion_overrides(component, profile), profile
    if component == "dit":
        if profile == "F32":
            return "orig", ["*=f32"], "F32"
        base = {"Q8_0": "q8_0", "Q6_K": "q6_k", "Q5_K_M": "q5_k", "Q4_K_M": "q4_k"}[profile]
        return base, promotion_overrides(component, profile), profile
    if component == "condition":
        return "orig", ["*=f32"], "F32"
    if component == "vae":
        if profile == "F32":
            return "orig", ["*=f32"], "F32"
        if profile == "F16":
            return "f16", ["dec_in_proj*=f32"], "F16"
    raise ValueError(component)


def validate_source(source: Path, dry_run: bool) -> dict:
    inventory = json.loads(SOURCE_INVENTORY.read_text(encoding="utf-8"))
    manifest_path = source / "source-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("repository") != inventory["repository"] or manifest.get("revision") != inventory["revision"]:
            raise RuntimeError("source manifest repository/revision does not match the pinned inventory")
        if not dry_run and not manifest.get("complete"):
            raise RuntimeError("source manifest is incomplete; run convert/download_model.py without --metadata-only")
    elif not dry_run:
        raise RuntimeError("source-manifest.json is missing; use the pinned downloader")
    return inventory


def copy_sidecars(source: Path, output: Path) -> None:
    for source_name, output_name in SIDECARS.items():
        src, dst = source / source_name, output / output_name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(dst) != sha256(src):
            raise RuntimeError(f"refusing to replace different sidecar {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)


def convert_one(args: argparse.Namespace, inventory: dict, component: str, profile: str) -> None:
    spec = COMPONENTS[component]
    storage, overrides, output_profile = recipe(component, profile)
    output_path = args.output / f"{spec.output_stem}-{output_profile}.gguf"
    input_path = args.source / spec.source
    command = [str(args.converter), "--input", str(input_path), "--root", str(args.source),
               "--output", str(output_path), "--type", storage, "--family", "minimax_music3",
               "--model-spec", str(MODEL_SPEC), "--allow-missing-model-spec"]
    if component != "lm":
        command.append("--no-sidecars")
    for source_name, destination in SIDECARS.items():
        command.extend(["--sidecar", f"{args.source / source_name}={destination}"])
    metadata = {
        "minimax.component": component,
        "minimax.profile": output_profile,
        "minimax.source.repository": inventory["repository"],
        "minimax.source.revision": inventory["revision"],
        "minimax.source.inventory_sha256": sha256(SOURCE_INVENTORY),
        "minimax.ggml.revision": "70081fdfc8685b60477b54d9d11cd679c5a00cb1",
        "minimax.converter.version": CONVERTER_VERSION,
    }
    for key, value in metadata.items():
        command.extend(["--metadata", f"{key}={value}"])
    for override in overrides:
        command.extend(["--keep-type", override])
    print("[plan]", component, output_profile, "->", output_path, flush=True)
    if args.dry_run:
        print("       ", " ".join(command), flush=True)
        return
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact {output_path}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    subprocess.run(command, cwd=ROOT, check=True)
    digest = sha256(output_path)
    source_inventory_hash = sha256(SOURCE_INVENTORY)
    artifact_manifest = {
        "schema_version": 1,
        "converter_version": CONVERTER_VERSION,
        "component": component,
        "profile": output_profile,
        "storage_type": storage,
        "type_overrides": overrides,
        "source_repository": inventory["repository"],
        "source_revision": inventory["revision"],
        "source_inventory_sha256": source_inventory_hash,
        "ggml_revision": "70081fdfc8685b60477b54d9d11cd679c5a00cb1",
        "gguf": {"filename": output_path.name, "size": output_path.stat().st_size, "sha256": digest},
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=ROOT / "build" / "bin" / "minimax-gguf-convert")
    parser.add_argument("--components", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components = parse_set(args.components, COMPONENTS, "component")
    profiles = parse_set(args.profiles, PROFILES, "profile")
    inventory = validate_source(args.source, args.dry_run)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        if not args.converter.is_file():
            raise FileNotFoundError(args.converter)
        copy_sidecars(args.source, args.output)
    for component in components:
        component_profiles = [profile for profile in profiles if profile in COMPONENT_PROFILES[component]]
        for profile in component_profiles:
            convert_one(args, inventory, component, profile)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
