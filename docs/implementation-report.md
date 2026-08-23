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

## 2026-08-23 — strict Cantor request contract

- Added the closed Cantor v1 JSON schema and canonical serializer. The parser
  rejects unknown/duplicate keys, aliases, malformed Unicode, non-exact
  unsigned integers, unsupported output rates, and out-of-range generation
  settings.
- Preserved an absent flow seed so the runtime, rather than the parser, can
  apply the pinned SGLang seed derivation.
- Added `minimax-cli --validate-request REQUEST.json` for deployment-side
  validation and canonicalization.
- Rebuilt the CPU target and ran all five weightless tests successfully.

Next: freeze the component checkpoint inventory, deterministic source manifest,
and streaming GGUF conversion contract.

## 2026-08-23 — checkpoint pipeline and native graphs

- Froze all 25 componentized inference files (28,517,620,805 bytes) at the
  approved checkpoint commit, including exact SHA-256 values. Added a
  resumable stdlib-only downloader and verified the 16-file metadata subset on
  this CPU machine.
- Imported the audited MiniMax-only audio.cpp framework subset. It builds
  against the pinned GGML after lowering three post-pin/custom paths through
  portable public operations; unrelated models and SentencePiece are absent.
- Compiled the global Qwen3 LM, RVQ depth decoder, condition projection, 36
  layer flow transformer/Euler sampler, and stereo vocoder into
  `minimax-native`. Added a strict config fingerprint before weight loading.
- Added a converter and the 18-artifact matrix: BF16 LM/RVQ; F32 condition;
  F32 DiT; F32/F16 VAE; and Q8_0, Q6_K, Q5_K_M, and Q4_K_M for LM/RVQ/DiT.
  Mixed profiles promote lookup/output/down projections to Q6_K and keep
  convolution/time tensors F32. Every output receives deterministic manifest
  and checksum sidecars, and existing artifacts are never overwritten.
- Added `minimax-cli --generate` with explicit CPU/CUDA/HIP/Vulkan/Metal
  selection and all five component selectors. Native 44.1 kHz stereo and
  explicit 32 kHz output are wired.
- Built the full CPU runtime and converter, ran seven tests, and confirmed the
  GGML compute smoke returns `5 5 5 5`.

Next: finish official RNG/seed parity and implement the versioned Cantor ABI v1
stage boundaries before real-model GPU validation.

## 2026-08-23 — official position-addressed RNG contract

- Replaced the imported runtime's shared CUDA-style AR/RVQ random stream with
  SGLang 0.5.16's backend-independent MurmurHash3/Gumbel-max sampler. The
  public seed is first reduced with the pinned `minimax-ttm-ar` BLAKE2b
  namespace, and every draw is addressed by `(seed, frame * 8 + codebook,
  compact_vocab_column)`. This preserves results across batch size and GGML
  CPU/CUDA/HIP/Vulkan/Metal implementations.
- Matched SGLang's compact semantic column order (`audio_end` at column zero,
  then 16,384 semantic codes) even though the local readback buffer stores the
  token range first.
- Flow noise now uses the exact personalized BLAKE2b derivation
  `(base_seed, "dit", chunk_index)` and resets the PyTorch-compatible Philox
  cursor for each acoustic window. An explicitly supplied flow seed is the
  base; otherwise the request seed is used.
- Guidance scale zero is accepted consistently with the public schema and the
  pinned acoustic server.
- Added independent seed/hash test vectors generated from the pinned Python
  code. The full native CPU build, all eight tests, and the GGML compute smoke
  pass.

Next: expose replayable AR, projected-condition, resumable Euler, and decode
boundaries through the unchanged Cantor engine ABI v1.

## 2026-08-23 — Cantor ABI, durable execution, and publication scaffolding

- Added the unchanged 13-function Cantor engine ABI v1 with the five approved
  roles and only CODES, DIFFUSE, and DECODE advertised. The shared library
  exports no other implementation symbols on ELF.
- Added checksummed, versioned, little-endian durable envelopes with an exact
  section inventory, 2 GiB limit, component SHA-256 identities, shape/cursor
  validation, and strict source-boundary nesting.
- Made AR resumable by replaying `[T,8]` codes to rebuild LM/RVQ KV state while
  retaining the official position-addressed random coordinates. Completed
  CODES stores pre-nearest-resize BF16 condition projections.
- Made the Euler solver resumable after every update. Paused DIFFUSE stores
  finished windows, the active F32 latent, overlap carries, and its exact
  window/step cursor. Chunk noise is regenerated from the pinned seed
  derivation; DECODE uses retry-only cancellation from the complete DIFFUSE
  boundary.
- Added the strict Hugging Face publisher/model card and a committed pending
  release matrix. Publication is immutable and blocked until all real-model
  parity/backend gates contain passing evidence.
- Added CI and GitHub Release workflows for Linux CPU/CUDA/Vulkan, macOS Metal,
  Windows CPU/Vulkan, and Android CPU/Vulkan. Release jobs assert the ABI,
  include licenses/checksums, and refuse asset replacement.
- Validation command:

  ```bash
  cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
  cmake --build build-native -j 4
  ctest --test-dir build-native --output-on-failure
  ```

  All eleven current tests pass, including the C ABI smoke, checkpoint
  corruption vectors, deterministic RNG vectors, and publication policy.

Next: finish the real 18-artifact CPU conversion, record its checksums, then
move to one H100 80 GiB for dense Python/CUDA parity and resume-boundary tests.
