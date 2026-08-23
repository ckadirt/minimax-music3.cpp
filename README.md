# minimax-music3.cpp

Portable C++17/GGML inference for MiniMax Music 3, including its global
language model, RVQ depth decoder, acoustic condition encoder, flow-matching
transformer, and waveform decoder.

This repository is under active development. The first release is gated on
numerical parity with pinned official Python implementations, strict GGUF
conversion, Cantor ABI v1 pause/resume support, and real-backend validation.
Do not treat an untagged build or unpublished checkpoint as release quality.

## Planned runtime contract

- Lyrics and a music description produce up to five minutes of stereo audio.
- Native output is 44.1 kHz; explicit 32 kHz serving compatibility is planned.
- Model components are supplied through the Cantor roles `lm`, `rvq`,
  `condition`, `dit`, and `vae`.
- CPU, CUDA, Vulkan, Metal, and Android builds share the same model code.

The pinned sources and implementation order are recorded in
[`docs/plan.md`](docs/plan.md). Ongoing work and reproducible commands are
recorded in [`docs/implementation-report.md`](docs/implementation-report.md).

## Licenses

Repository source is Apache-2.0. Portions adapted from audio.cpp retain their
copyright and attribution. MiniMax Music 3 checkpoints and checkpoint-derived
GGUF artifacts are governed by the MiniMax Music 3 Community License; review
the upstream terms before using or distributing the model.
