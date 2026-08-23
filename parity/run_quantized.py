#!/usr/bin/env python3
"""Run the complete mixed-component GGUF acceptance matrix."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

MATRIX = (
    {
        "name": "BF16-F32-F32VAE",
        "lm": "lm-BF16.gguf",
        "rvq": "rvq-BF16.gguf",
        "dit": "dit-F32.gguf",
        "vae": "vae-F32.gguf",
    },
    {
        "name": "BF16-F32-F16VAE",
        "lm": "lm-BF16.gguf",
        "rvq": "rvq-BF16.gguf",
        "dit": "dit-F32.gguf",
        "vae": "vae-F16.gguf",
    },
    *(
        {
            "name": profile,
            "lm": f"lm-{profile}.gguf",
            "rvq": f"rvq-{profile}.gguf",
            "dit": f"dit-{profile}.gguf",
            "vae": "vae-F16.gguf",
        }
        for profile in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_wav(path: Path, expected_rate: int) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        payload = wav.readframes(frames)
    if channels != 2 or rate != expected_rate or width != 2 or frames <= 0:
        raise RuntimeError(
            f"{path}: expected non-empty stereo PCM16 at {expected_rate} Hz, got "
            f"channels={channels} rate={rate} width={width} frames={frames}")
    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max(abs(value) for value in samples)
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    if peak == 0 or not math.isfinite(rms) or rms == 0:
        raise RuntimeError(f"{path}: output is silent or non-finite")
    return {
        "frames": frames,
        "peak_i16": peak,
        "rms_i16": rms,
        "sha256": sha256(path),
        "size": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip", "vulkan", "metal"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profiles", default="all",
                        help="comma-separated matrix names, or all")
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def selected_matrix(value: str) -> list[dict[str, str]]:
    if value == "all":
        return list(MATRIX)
    names = value.split(",")
    if any(not name or name.strip() != name for name in names) or len(names) != len(set(names)):
        raise ValueError("--profiles must be a unique comma-separated list without whitespace")
    records = {record["name"]: record for record in MATRIX}
    unknown = sorted(set(names) - set(records))
    if unknown:
        raise ValueError("unknown profile(s): " + ", ".join(unknown))
    return [records[name] for name in names]


def main() -> int:
    args = parse_args()
    if args.device < 0 or args.threads <= 0 or args.timeout <= 0:
        raise ValueError("device must be non-negative; threads and timeout must be positive")
    binary = args.binary.resolve()
    model = args.model.resolve()
    request_path = args.request.resolve()
    if not binary.is_file() or not model.is_dir() or not request_path.is_file():
        raise RuntimeError("binary, model directory, or request does not exist")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected_rate = request.get("output_sample_rate", 44100)
    if expected_rate not in (32000, 44100):
        raise RuntimeError("request output_sample_rate is unsupported")

    selected = selected_matrix(args.profiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for record in selected:
        for role in ("lm", "rvq", "dit", "vae"):
            if not (model / record[role]).is_file():
                raise RuntimeError(f"missing {role} artifact for {record['name']}: {record[role]}")
        output = (args.output_dir / f"{record['name']}.wav").resolve()
        command = [
            str(binary), "--generate", str(request_path),
            "--model", str(model), "--output", str(output),
            "--backend", args.backend, "--device", str(args.device),
            "--threads", str(args.threads),
            "--lm", record["lm"], "--rvq", record["rvq"],
            "--condition", "condition-F32.gguf",
            "--dit", record["dit"], "--vae", record["vae"],
        ]
        print(f"[run] {record['name']}", flush=True)
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=args.timeout, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
            raise RuntimeError(f"{record['name']} failed with exit {completed.returncode}: {detail}")
        metrics = inspect_wav(output, expected_rate)
        results.append({**record, **metrics})
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    summary = {
        "schema_version": 1,
        "backend": args.backend,
        "device": args.device,
        "request_sha256": sha256(request_path),
        "profiles": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
