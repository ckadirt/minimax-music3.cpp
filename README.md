# minimax-music3.cpp

Portable C++17/GGML inference for MiniMax Music 3, including its global
language model, RVQ depth decoder, acoustic condition encoder, flow-matching
transformer, and waveform decoder.

This repository is under active development. The first release is gated on
numerical parity with pinned official Python implementations, strict GGUF
conversion, Cantor ABI v1 pause/resume support, and real-backend validation.
Do not treat an untagged build or unpublished checkpoint as release quality.

## Runtime contract

- Lyrics and a music description produce up to five minutes of stereo audio.
- Native output is 44.1 kHz; explicit 32 kHz serving compatibility uses a
  Torchaudio-compatible sinc/Hann resampler.
- Model components are supplied through the Cantor roles `lm`, `rvq`,
  `condition`, `dit`, and `vae`.
- CPU, CUDA, Vulkan, Metal, and Android builds share the same model code.

## Build and conversion

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

python3 convert/download_model.py --output models/MiniMax-Music3-source
python3 convert/convert_model.py \
  --source models/MiniMax-Music3-source \
  --output artifacts/MiniMax-Music3-GGUF \
  --converter build/bin/minimax-gguf-convert
```

The downloader accepts `HF_TOKEN` from the environment, resumes `.part`
files, and verifies every byte against the pinned 25-file inventory. The full
conversion produces 18 artifacts across the five component roles. Existing
GGUFs are never overwritten.

Generation uses the Q4_K_M/F16 default component mix, or explicit role files:

```bash
build/bin/minimax-cli --generate request.json \
  --model artifacts/MiniMax-Music3-GGUF \
  --output song.wav --backend cuda --device 0
```

The pinned sources and implementation order are recorded in
[`docs/plan.md`](docs/plan.md). Ongoing work and reproducible commands are
recorded in [`docs/implementation-report.md`](docs/implementation-report.md).
The exact staged ABI and checkpoint ownership rules are documented in
[`docs/cantor.md`](docs/cantor.md).

Release workflows compile CPU/CUDA/Vulkan, Metal, Windows, and Android backend
archives and refuse to replace an existing GitHub Release asset. The model
publisher verifies the complete matrix and refuses network publication until
every gate in `validation/release-matrix.json` is explicitly passed with
evidence.

## Licenses

Repository source is Apache-2.0. Portions adapted from audio.cpp retain their
copyright and attribution. MiniMax Music 3 checkpoints and checkpoint-derived
GGUF artifacts are governed by the MiniMax Music 3 Community License; review
the upstream terms before using or distributing the model.
