#!/usr/bin/env python3
"""Run the pinned Diffusers oracle, native dense components, and metric gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "parity" / "fixture-spec.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "parity")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    oracle = args.output / "oracle"
    native = args.output / "native"
    native.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(ROOT / "parity" / "capture_oracle.py"),
         "--source", str(args.source), "--output", str(oracle), "--device", f"cuda:{args.device}"])
    common = [
        str(args.binary), "--model-dir", str(args.gguf), "--backend", args.backend,
        "--device", str(args.device), "--threads", str(args.threads),
    ]
    condition = spec["components"]["condition"]
    run(common + ["--component", "condition", "--input", str(oracle / "condition-input.f32"),
                  "--output", str(native / "condition-native.f32"), "--frames", str(condition["frames"])])
    dit = spec["components"]["dit"]
    run(common + ["--component", "dit", "--input", str(oracle / "dit-latent.f32"),
                  "--condition", str(oracle / "dit-condition.f32"),
                  "--output", str(native / "dit-native.f32"), "--frames", str(dit["frames"]),
                  "--timestep", str(dit["timestep"])])
    vocoder = spec["components"]["vocoder"]
    run(common + ["--component", "vocoder", "--input", str(oracle / "vocoder-latent.f32"),
                  "--output", str(native / "vocoder-native.f32"), "--frames", str(vocoder["frames"])])
    summary = args.output / "dense-summary.json"
    run([sys.executable, str(ROOT / "parity" / "compare.py"),
         "--oracle-dir", str(oracle), "--native-dir", str(native), "--summary", str(summary)])
    evidence = {
        "schema_version": 1,
        "backend": args.backend,
        "device_index": args.device,
        "binary_sha256": sha256(args.binary),
        "gguf_sha256": {
            name: sha256(args.gguf / filename)
            for name, filename in {
                "condition": "condition-F32.gguf", "dit": "dit-F32.gguf", "vocoder": "vae-F32.gguf"
            }.items()
        },
        "summary_sha256": sha256(summary),
    }
    (args.output / "run-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
