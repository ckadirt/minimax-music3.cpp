#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("publish_r2", root / "convert" / "publish_r2.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

manifest = module.validate_manifest(root / "publish" / "cloudflare-r2.json")
assert manifest["unique_model_bytes"] == 46_148_926_304
assert len(manifest["objects"]) == 12
assert manifest["manifest_url"] == "https://cantor-ckpts.ckadirt.xyz/minimax-music3-1.0/manifest.json"

variants = {item["tag"]: item for item in manifest["variants"]}
assert variants["1.0-fast"]["total_bytes"] == 8_117_802_016
assert variants["1.0-balanced"]["total_bytes"] == 9_823_284_224
assert variants["1.0-quality"]["total_bytes"] == 28_517_696_672
assert [item["quant"] for item in variants["1.0-fast"]["components"]] == [
    "Q4_K_M", "Q4_K_M", "F32", "Q4_K_M", "F16"]
assert [item["quant"] for item in variants["1.0-balanced"]["components"]] == [
    "Q6_K", "Q6_K", "F32", "Q6_K", "F16"]
assert [item["quant"] for item in variants["1.0-quality"]["components"]] == [
    "BF16", "BF16", "F32", "F32", "F32"]
