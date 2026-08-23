#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("convert_model", root / "convert" / "convert_model.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

assert sum(len(value) for value in module.COMPONENT_PROFILES.values()) == 18
assert module.recipe("lm", "Q4_K_M")[0] == "q4_k"
assert module.recipe("rvq", "Q5_K_M")[0] == "q5_k"
assert module.recipe("dit", "F32") == ("orig", ["*=f32"], "F32")
assert module.recipe("vae", "F16") == ("f16", ["dec_in_proj*=f32"], "F16")

lm_mixed = module.promotion_overrides("lm", "Q4_K_M")
assert "model.embed_tokens.weight=q6_k" in lm_mixed
assert "lm_head.weight=q6_k" in lm_mixed
assert len(lm_mixed) == 74

rvq_mixed = module.promotion_overrides("rvq", "Q5_K_M")
assert "pos_embedding.weight=f32" in rvq_mixed
assert "audio_embeddings.weight=q6_k" in rvq_mixed
assert len(rvq_mixed) == 17

dit_mixed = module.promotion_overrides("dit", "Q4_K_M")
assert "time_proj.weight=f32" in dit_mixed
assert "transformer_blocks.35.ff_out.weight=q6_k" in dit_mixed
assert len(dit_mixed) == 77
