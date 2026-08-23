---
license: other
license_name: minimax-music-3-community-license
library_name: ggml
pipeline_tag: text-to-audio
tags:
  - music-generation
  - gguf
  - cantor
  - minimax-music-3
---

# MiniMax Music 3 GGUF

Portable GGUF components for
[MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
converted for [`minimax-music3.cpp`](https://github.com/ckadirt/minimax-music3.cpp)
and the Cantor engine ABI v1.

These files are checkpoint derivatives. They are governed by the included
`MODEL_LICENSE` (the MiniMax Music 3 Community License), not by the Apache-2.0
license used for the C++ source repository. Review the model license before
using or redistributing them.

## Components

Load exactly one file for each role: `lm`, `rvq`, `condition`, `dit`, and
`vae`. The condition encoder has one F32 artifact. The VAE has F32 and F16
storage variants. LM, RVQ, and DiT provide the following quality ladder:

- `BF16` for LM/RVQ or `F32` for DiT: reference storage
- `Q8_0`: high-quality quantized storage
- `Q6_K`: quality-oriented K-quant
- `Q5_K_M`: intermediate K-quant
- `Q4_K_M`: balanced/default K-quant

The recommended balanced set is `lm-Q4_K_M`, `rvq-Q4_K_M`,
`condition-F32`, `dit-Q4_K_M`, and `vae-F16`. Do not rename component files:
their exact bytes are part of Cantor pause/resume identity checks.

{{ARTIFACT_TABLE}}

Each GGUF has a `.manifest.json` and `.sha256` sidecar. The LM GGUFs also
embed the pinned tokenizer, all component configs, source inventory, model
license, and model specification. Every GGUF embeds its component, profile,
source commit, source-inventory hash, converter version, and GGML revision.

## Runtime

Native output is stereo 44.1 kHz; Cantor requests may explicitly select the
Torchaudio-compatible 32 kHz serving resampler. The same model implementation
targets CPU, CUDA, HIP, Vulkan, Metal, and Android through GGML.

This initial publication remains staging-only until the repository's committed
real-model parity matrix passes on the reference Python implementation and the
declared hardware backends.

