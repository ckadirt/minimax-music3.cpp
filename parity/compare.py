#!/usr/bin/env python3
"""Compare raw little-endian F32 native captures against the dense oracle."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
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


def read_f32(path: Path) -> array.array:
    if path.stat().st_size % 4:
        raise RuntimeError(f"F32 file has a non-integral element count: {path}")
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // 4)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def metrics(reference: array.array, actual: array.array) -> dict[str, float | int]:
    if len(reference) != len(actual):
        raise RuntimeError(f"element count mismatch: {len(reference)} != {len(actual)}")
    if not reference:
        raise RuntimeError("cannot compare empty tensors")
    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    squared_actual = 0.0
    dot = 0.0
    for expected, observed in zip(reference, actual):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise RuntimeError("parity tensor contains a non-finite value")
        difference = float(observed) - float(expected)
        max_abs = max(max_abs, abs(difference))
        squared_error += difference * difference
        squared_reference += float(expected) * float(expected)
        squared_actual += float(observed) * float(observed)
        dot += float(expected) * float(observed)
    reference_rms = math.sqrt(squared_reference / len(reference))
    relative_rms = math.sqrt(squared_error / len(reference)) / max(reference_rms, 1.0e-30)
    cosine = dot / max(math.sqrt(squared_reference * squared_actual), 1.0e-30)
    return {"elements": len(reference), "max_abs": max_abs, "relative_rms": relative_rms, "cosine": cosine}


def compare_component(name: str, spec: dict, oracle: Path, native: Path) -> dict:
    reference = read_f32(oracle)
    actual = read_f32(native)
    expected_elements = math.prod(spec["output_shape"])
    if len(reference) != expected_elements or len(actual) != expected_elements:
        raise RuntimeError(
            f"{name} shape-size mismatch: expected {expected_elements}, "
            f"oracle has {len(reference)}, native has {len(actual)}"
        )
    result = metrics(reference, actual)
    failures = []
    if result["max_abs"] > spec["max_abs"]:
        failures.append(f"max_abs {result['max_abs']:.9g} > {spec['max_abs']:.9g}")
    if "max_relative_rms" in spec and result["relative_rms"] > spec["max_relative_rms"]:
        failures.append(
            f"relative_rms {result['relative_rms']:.9g} > {spec['max_relative_rms']:.9g}")
    return {
        **result,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "oracle_sha256": sha256(oracle),
        "native_sha256": sha256(native),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    components = {}
    for name, component_spec in spec["components"].items():
        components[name] = compare_component(
            name,
            component_spec,
            args.oracle_dir / f"{name}-oracle.f32",
            args.native_dir / f"{name}-native.f32",
        )
    passed = all(item["status"] == "pass" for item in components.values())
    summary = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "source_revision": spec["source_revision"],
        "diffusers_revision": spec["diffusers_revision"],
        "fixture_spec_sha256": sha256(SPEC_PATH),
        "components": components,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
