#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("publish_hf", root / "convert" / "publish_hf.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

assert module.DEFAULT_REPO == "ckadirt/MiniMax-Music3-GGUF"
assert len(module.expected_ggufs()) == 18
assert len(set(module.expected_ggufs())) == 18
inventory_hash = module.sha256(root / "convert" / "source_files.json")

with tempfile.TemporaryDirectory(prefix="minimax-publish-test-") as temporary:
    folder = Path(temporary)
    for relative in module.SIDECARS:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    for component, profiles in module.EXPECTED.items():
        for profile in profiles:
            filename = f"{component}-{profile}.gguf"
            path = folder / filename
            path.write_bytes(f"GGUF fixture {component} {profile}\n".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 1,
                "converter_version": "1",
                "component": component,
                "profile": profile,
                "storage_type": "fixture",
                "type_overrides": [],
                "source_repository": "MiniMaxAI/MiniMax-Music3",
                "source_revision": module.SOURCE_REVISION,
                "source_inventory_sha256": inventory_hash,
                "ggml_revision": module.GGML_REVISION,
                "gguf": {"filename": filename, "size": path.stat().st_size, "sha256": digest},
            }
            (folder / f"{filename}.manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            (folder / f"{filename}.sha256").write_text(
                f"{digest}  {filename}\n", encoding="ascii")

    manifests = module.validate_artifacts(folder)
    card = module.render_model_card(manifests)
    assert "{{ARTIFACT_TABLE}}" not in card
    assert sum(line.startswith("| `") for line in card.splitlines()) == 18
    assert "MiniMax Music 3 Community License" in card

    broken = folder / "lm-BF16.gguf.sha256"
    original = broken.read_text(encoding="ascii")
    broken.write_text("0" * 64 + "  lm-BF16.gguf\n", encoding="ascii")
    try:
        module.validate_artifacts(folder)
        raise AssertionError("publisher accepted a non-canonical checksum")
    except RuntimeError as error:
        assert "checksum sidecar" in str(error)
    broken.write_text(original, encoding="ascii")

try:
    module.validate_release_matrix(root / "validation" / "release-matrix.json")
    raise AssertionError("publisher accepted the pending release matrix")
except RuntimeError as error:
    assert "incomplete validation" in str(error)
