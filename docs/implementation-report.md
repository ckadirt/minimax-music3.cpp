# Implementation report

This file is the append-only operational journal for the port. Entries record
what changed, exact commands and revisions, validation results, blockers, and
the next resumable step.

## 2026-08-23 — repository grounding and approved contract

- Confirmed the target repository contained only its initial README.
- Inspected `LeVo2.cpp` and `acestep.cpp` for strict conversion, native parity,
  quantization, Cantor ABI v1, staged resume, dynamic GGML backends, and release
  conventions.
- Pinned the official checkpoint and Python/native references listed in
  `docs/plan.md`.
- The full upstream snapshot is 57.4 GB, but contains duplicate legacy model
  layouts. The componentized inference subset is the sole conversion input.
- Current machine: four CPU cores, 14 GiB RAM, 348 GiB free workspace storage,
  and no GPU. The converter therefore must stream/mmap shards.
- Approved public choices: adapt and audit audio.cpp; publish to
  `ckadirt/MiniMax-Music3-GGUF`; native 44.1 kHz default; five Cantor roles;
  official RNG sequence on every backend; full LeVo-style quantization matrix;
  desktop plus Android builds; GitHub Releases only for backend binaries.
- Approved completed-CODES boundary: per-window BF16 projected conditions,
  approximately 69.5 MiB at five minutes rather than raw hidden states.

Next: add the pinned GGML/build scaffold and weightless contract tests.

## 2026-08-23 — pinned GGML and weightless core scaffold

- Added `ckadirt/ggml` at `70081fdfc8685b60477b54d9d11cd679c5a00cb1`
  as a submodule and a C++17 core/CLI build.
- Added backend discovery and a real GGML vector-add smoke. The current CPU
  backend returns the expected `5 5 5 5` result.
- Adapted the proven Qwen2 byte-level BPE and vendored Unicode/NFC support,
  retaining their licenses and modification notices.
- Ported the pinned caption cleanup and lyric normalization rules. In
  particular, text placed after a leading structure tag on the same line is
  dropped because that is the official prompt contract.
- Added the exact acoustic window geometry and completed-CODES projected-state
  calculation. The tests freeze 7 windows and 5,529,600 bytes at 750 frames,
  and 89 windows and 72,908,800 bytes at 9,000 frames.
- Validation command:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j 4
  ctest --test-dir build --output-on-failure
  ```

  All four tests pass: CPU backend smoke, chunking contract, prompt
  normalization, and tokenizer contract.

Next: add strict request/config parsing and the component GGUF conversion
inventory before importing weight-owning model graphs.
