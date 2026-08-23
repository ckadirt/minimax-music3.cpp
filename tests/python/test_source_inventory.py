#!/usr/bin/env python3

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "convert" / "source_files.json").open("r", encoding="utf-8") as handle:
    inventory = json.load(handle)

assert inventory["repository"] == "MiniMaxAI/MiniMax-Music3"
assert inventory["revision"] == "fbdf52fbaaca799592917417eb05f1899f1255ec"
files = inventory["files"]
paths = [item["path"] for item in files]
assert len(files) == 25
assert len(paths) == len(set(paths))
assert all(not Path(path).is_absolute() and ".." not in Path(path).parts for path in paths)
assert all(item["size"] > 0 and len(item["sha256"]) == 64 for item in files)
assert all(bytes.fromhex(item["sha256"]) for item in files)
assert sum(item["size"] for item in files) == 28_517_620_805
assert sum(item["size"] for item in files if item["path"].endswith(".safetensors")) == 28_506_091_948
assert "language_model/model.safetensors.index.json" in paths
assert "transformer/diffusion_pytorch_model.safetensors.index.json" in paths
