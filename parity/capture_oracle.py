#!/usr/bin/env python3
"""Capture deterministic dense Diffusers component outputs for native parity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diffusers.models.autoencoders.minimax_music3_vocoder import MiniMaxMusic3Vocoder
from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import MiniMaxMusic3ConditionEncoder
from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "parity" / "fixture-spec.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype="<f4").tofile(path)


def unload(model: torch.nn.Module) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_model(cls, source: Path, subfolder: str, device: torch.device):
    model = cls.from_pretrained(
        source,
        subfolder=subfolder,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    return model.eval().to(device)


def verify_source(source: Path, revision: str) -> None:
    manifest_path = source / "source-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("source-manifest.json is missing; use convert/download_model.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != revision or not manifest.get("complete"):
        raise RuntimeError("source snapshot is not the complete pinned revision")


def capture_condition(source: Path, output: Path, rng: np.random.Generator, spec: dict, device: torch.device) -> dict:
    values = (rng.standard_normal(spec["input_shape"]) * spec["input_scale"]).astype("<f4")
    input_path = output / "condition-input.f32"
    oracle_path = output / "condition-oracle.f32"
    write_f32(input_path, values)
    model = load_model(MiniMaxMusic3ConditionEncoder, source, "condition_encoder", device)
    with torch.inference_mode():
        result = model(torch.from_numpy(values).to(device)).float().cpu().numpy()
    unload(model)
    if list(result.shape) != spec["output_shape"]:
        raise RuntimeError(f"condition output shape mismatch: {list(result.shape)}")
    write_f32(oracle_path, result)
    return {"inputs": [input_path.name], "oracle": oracle_path.name}


def capture_dit(source: Path, output: Path, rng: np.random.Generator, spec: dict, device: torch.device) -> dict:
    latent = (rng.standard_normal(spec["latent_shape"]) * spec["latent_scale"]).astype("<f4")
    condition = (rng.standard_normal(spec["condition_shape"]) * spec["condition_scale"]).astype("<f4")
    latent_path = output / "dit-latent.f32"
    condition_path = output / "dit-condition.f32"
    oracle_path = output / "dit-oracle.f32"
    write_f32(latent_path, latent)
    write_f32(condition_path, condition)
    model = load_model(MiniMaxMusic3Transformer1DModel, source, "transformer", device)
    with torch.inference_mode():
        latent_tensor = torch.from_numpy(latent).to(device)
        condition_tensor = torch.from_numpy(condition).to(device)
        result = model(
            hidden_states=latent_tensor.repeat(2, 1, 1),
            timestep=torch.full((2,), spec["timestep"], dtype=torch.float32, device=device),
            encoder_hidden_states=torch.cat((condition_tensor, torch.zeros_like(condition_tensor))),
        ).sample.float().cpu().numpy()
    unload(model)
    if list(result.shape) != spec["output_shape"]:
        raise RuntimeError(f"dit output shape mismatch: {list(result.shape)}")
    write_f32(oracle_path, result)
    return {"inputs": [latent_path.name, condition_path.name], "oracle": oracle_path.name}


def capture_vocoder(source: Path, output: Path, rng: np.random.Generator, spec: dict, device: torch.device) -> dict:
    latent = (rng.standard_normal(spec["latent_shape"]) * spec["latent_scale"]).astype("<f4")
    latent_path = output / "vocoder-latent.f32"
    oracle_path = output / "vocoder-oracle.f32"
    write_f32(latent_path, latent)
    model = load_model(MiniMaxMusic3Vocoder, source, "vocoder", device)
    with torch.inference_mode():
        result = model(torch.from_numpy(latent).to(device)).float().cpu().numpy()
    unload(model)
    if list(result.shape) != spec["output_shape"]:
        raise RuntimeError(f"vocoder output shape mismatch: {list(result.shape)}")
    write_f32(oracle_path, result)
    return {"inputs": [latent_path.name], "oracle": oracle_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    verify_source(args.source, spec["source_revision"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dense oracle capture requires an available CUDA device")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec["seed"])
    captures = {
        "condition": capture_condition(args.source, args.output, rng, spec["components"]["condition"], device),
        "dit": capture_dit(args.source, args.output, rng, spec["components"]["dit"], device),
        "vocoder": capture_vocoder(args.source, args.output, rng, spec["components"]["vocoder"], device),
    }
    files = sorted({name for item in captures.values() for name in item["inputs"] + [item["oracle"]]})
    manifest = {
        "schema_version": 1,
        "source_revision": spec["source_revision"],
        "diffusers_revision": spec["diffusers_revision"],
        "fixture_spec_sha256": sha256(SPEC_PATH),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "captures": captures,
        "files": {name: {"size": (args.output / name).stat().st_size, "sha256": sha256(args.output / name)} for name in files},
    }
    (args.output / "oracle-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
