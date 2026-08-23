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
