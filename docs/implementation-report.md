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

## 2026-08-23 — complete artifact matrix and real CPU validation

- Completed all 18 planned GGUFs with streaming conversion: five LM, five
  RVQ, one condition, five DiT, and two VAE profiles. The GGUF payload total is
  exactly 67,320,282,560 bytes. `convert/publish_hf.py --dry-run` re-hashed the
  entire matrix and accepted every manifest, canonical checksum, pinned source
  revision, and pinned GGML revision; it performed no upload.
- Added `minimax-cantor-smoke`, a real-weight release harness. It compares
  uninterrupted and paused/resumed CODES and DIFFUSE boundaries byte-for-byte,
  verifies cancelled DECODE retains no partial audio, retries the immutable
  DIFFUSE state, and requires byte-identical finite non-silent planar output.
- That harness exposed destructive reuse of the imported DiT GGML graph: an
  identical second forward changed its first velocity value from `0.122267008`
  to `-3.03268623`. DiT inference now rebuilds graph allocations for every
  Euler evaluation while keeping weights resident, and evicts the corresponding
  CUDA/HIP graph-cache entry. The dense parity runner repeats identical DiT
  inputs to prevent this regression.
- After the fix, the real Q4 Cantor harness reports exact CODES, DIFFUSE, and
  DECODE resume equivalence, 3,072 samples per stereo channel at 44,100 Hz,
  peak `0.00836946`, RMS `0.00224306`, and audio FNV-1a-64
  `18291441890909882771`.
- The standalone real Q4 CLI produced deterministic 44.1 kHz stereo twice with
  identical WAV SHA-256
  `7d7af5c688711f7ec1ad645d4df7348ffc5382e3d378cdda1dff0cb69053c5b9`
  (1,536 frames, peak PCM16 453, RMS 145.119989). The 32 kHz serving path also
  produced finite non-silent stereo (1,115 frames) with SHA-256
  `aa61fd18b4e1cdd8fd29ea4e07c7b4c443de37f1917fd097be677d057b17c658`.
- All 12 weightless/native contract tests pass locally. Cross-compilation has
  separately reached green on Linux CPU/Vulkan, macOS Metal, Windows CPU, and
  Android CPU/Vulkan; the final post-fix CI run is still required before a
  release is approved.

Validation commands:

```bash
python3 convert/publish_hf.py --artifacts artifacts/gguf \
  --repo ckadirt/MiniMax-Music3-GGUF --dry-run
build-minimal-make/bin/minimax-cli --generate artifacts/cpu-smoke-request.json \
  --model artifacts/gguf --output artifacts/cpu-smoke.wav --backend cpu --threads 4
build-minimal-make/bin/minimax-cantor-smoke --model artifacts/gguf \
  --request artifacts/cantor-smoke-request.json --threads 4
```

Next: run the final portable CI matrix, then move to one H100 80 GiB for dense
Diffusers/CUDA parity, all quantized-profile smokes, and CUDA resume evidence.

## 2026-08-23 — final portable build matrix green

- GitHub Actions run
  [32656223224](https://github.com/ckadirt/minimax-music3.cpp/actions/runs/32656223224)
  passed on commit `747bdc36e2254170e65a15d03b3387627d6a3fb5`.
- Green jobs: Linux CPU, Linux Vulkan, Linux CUDA, macOS arm64 Metal,
  Windows x64 CPU, Android arm64 CPU, and Android arm64 Vulkan. Each job built
  the requested backend and asserted the Cantor ABI/backend surface; host jobs
  ran all weightless tests through their available runtime or CPU fallback.
- The matrix proves portable compilation, linking, plugin discovery, and ABI
  packaging. It does not claim real-model CUDA/Vulkan/Metal/Android numerical
  validation; those remain explicit release gates.
- The repository is ready for the GPU validation phase with dense oracle,
  repeatability, Cantor resume, and complete quantized-profile runners already
  committed.

Next: provision one NVIDIA H100 with 80 GiB VRAM and retain at least 100 GiB of
workspace storage for the existing source/GGUF artifacts plus parity fixtures.

## 2026-08-23 — H100 dense CUDA parity green

- Built the dynamic CUDA backend with CUDA 13.0.88 on an NVIDIA H100 80GB and
  passed all 12 native/Python contract tests. The runtime discovered the H100
  as CUDA device zero with compute capability 9.0.
- Corrected the GPU parity environment to use NumPy 1.26 because the previous
  unconstrained NumPy 2 requirement was incompatible with the oracle's SciPy
  import path.
- The first dense run exposed three precision lowerings hidden by CPU-only
  validation: F32 Conv1d used F16 im2col columns, dense DiT selected flash
  attention, and CUDA ignored `GGML_PREC_F32` when choosing its custom matmul
  kernel and cuBLAS math mode.
- F32 Conv1d now preserves F32 columns, dense DiT uses explicit attention, and
  the pinned GGML fork routes explicitly strict F32 CUDA matmuls through
  pedantic cuBLAS. Quantized and default operations retain their accelerated
  paths. The GGML fix is commit
  `2f50f0164352c6aa301da49a4e344c2379c82afd`.
- The fresh end-to-end dense runner passed without relaxing a threshold:
  condition max error `3.16650e-7`, DiT `6.67572e-6`, and vocoder
  `2.21223e-4`. Compact evidence is committed at
  `validation/evidence/h100-dense.json`; raw F32 fixtures remain ignored.

Next: rebuild the CUDA CLI/Cantor harness, run exact resume and rate smokes,
then execute all six mixed-component quantized profiles covering 18 GGUFs.

## 2026-08-23 — real CUDA runtime and quantized matrix green

- The H100 Q4 CLI produced byte-identical 44.1 kHz stereo on two runs with
  SHA-256 `cd59d414cd9c717e4950dd671d4595df82f2967b23730f30bdbee3379b360ddb`
  (1,536 frames, peak PCM16 425, RMS 136.915909031). The 32 kHz path produced
  1,115 finite non-silent stereo frames with SHA-256
  `3841ed99a1cad624d66964aa586263484c8752fbda9badcb941af378942c19de`.
- The CUDA Cantor harness preserved exact CODES, DIFFUSE, and DECODE results
  across interruption/resumption. It returned 3,072 samples per channel at
  44.1 kHz with FNV-1a-64 `4549339077756924446`.
- `parity/run_quantized.py` passed all six mixed-component cases: two BF16/F32
  baselines plus Q8_0, Q6_K, Q5_K_M, and Q4_K_M. Together they loaded every
  one of the 18 staged GGUFs and produced finite, non-silent 88,064-frame
  stereo PCM16 output.
- Compact release evidence is in `validation/evidence/h100-cuda-runtime.json`
  and `validation/evidence/h100-quantized.json`. The dense, quantized-profile,
  Cantor-resume, CPU, and CUDA gates are now passing. Publication remains
  blocked on real Vulkan, Metal, and Android model runs.

Next: probe this H100 host for a usable Vulkan ICD. If none is exposed, retain
the Vulkan gate as pending and proceed to portable CI for the CUDA changes.

## 2026-08-23 — Vulkan build green, host runtime unavailable

- The pinned LunarG Vulkan SDK 1.4.309.0 configured and built the complete
  release tree, including generated shaders, the dynamic `libggml-vulkan.so`
  plugin, the CLI, the Cantor backend, parity runners, and all test targets.
- All 12 tests pass with that Vulkan build. Backend discovery exposes CPU only,
  and `minimax-cli --smoke vulkan` correctly rejects the unavailable device.
- The host NVIDIA ICD advertises API 1.4.312, but its loader cannot resolve
  `vkCreateInstance`; `vulkaninfo --summary` exits with
  `ERROR_INCOMPATIBLE_DRIVER`. Consequently, this machine cannot supply a real
  Vulkan model run even though the portable backend compiles successfully.
- Reproducible probe details and binary hashes are captured in
  `validation/evidence/h100-vulkan-probe.json`. The release gate remains
  pending by design; build success is not being substituted for runtime proof.

Next: verify the pushed revision through the portable GitHub Actions matrix.
Real Vulkan, Metal, and Android devices remain the final publication gates.

## 2026-08-23 — Cantor fast/balanced/quality tiers on Cloudflare R2

- Selected three monotonic storage tiers following the existing Cantor/LeVo2
  convention: Q4_K_M `1.0-fast`, Q6_K `1.0-balanced`, and reference-precision
  BF16/F32 `1.0-quality`. Every tier contains the five ABI roles `lm`, `rvq`,
  `condition`, `dit`, and `vae`.
- The bundle totals are 8,117,802,016, 9,823,284,224, and 28,517,696,672
  bytes. Shared condition/VAE objects reduce the union to 12 GGUFs and
  46,148,926,304 bytes.
- `convert/publish_r2.py` rehashed the complete union, uploaded it immutably to
  `minimax-music3-1.0/`, verified R2 byte counts and SHA-256 metadata, then
  verified all 14 public URLs (12 GGUFs, license, and manifest). The local and
  public manifest SHA-256 is
  `2143899066fb37700da284b7426b0d046bf34af6ec32c2221e46799d88b62600`.
- The final machine-readable bundle is public at
  `https://cantor-ckpts.ckadirt.xyz/minimax-music3-1.0/manifest.json`; compact
  publication evidence is in `validation/evidence/cloudflare-r2.json`.
- Added all three variants to Cantor commit `186983e` on its existing
  `levo2-engine-family` branch. Relay tests, type checks, asset checks, and the
  Wrangler deploy dry-run pass. The live catalog deployment remains staged
  until `backends/v1.json` has matching `minimax-music3` ABI v1 archives, so
  nodes are not offered a checkpoint without an executable engine.

Next: complete portable CI on the updated publisher policy and build the
immutable MiniMax Cantor backend archives before activating the catalog entry.
