#!/usr/bin/env python3

import array
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fixture = json.loads((root / "parity" / "fixture-spec.json").read_text(encoding="utf-8"))
assert fixture["source_revision"] == "fbdf52fbaaca799592917417eb05f1899f1255ec"
assert fixture["diffusers_revision"] == "dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"
assert set(fixture["components"]) == {"condition", "dit", "vocoder"}
assert fixture["components"]["condition"]["output_shape"] == [1, 10, 2048]
assert fixture["components"]["vocoder"]["output_shape"] == [1, 2, 1024]

spec = importlib.util.spec_from_file_location("compare", root / "parity" / "compare.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

same = module.metrics(array.array("f", [1.0, -2.0, 0.5]), array.array("f", [1.0, -2.0, 0.5]))
assert same["max_abs"] == 0.0
assert same["relative_rms"] == 0.0
assert abs(same["cosine"] - 1.0) < 1e-12

different = module.metrics(array.array("f", [1.0, -2.0]), array.array("f", [1.1, -1.8]))
assert 0.19 < different["max_abs"] < 0.21
assert different["relative_rms"] > 0.0
