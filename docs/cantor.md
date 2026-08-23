# Cantor engine ABI v1

The release library is `libcantor_engine.so`, `libcantor_engine.dylib`, or
`cantor_engine.dll`. Its only public native symbols are the 13 unchanged
`cantor_engine_*` functions declared by `include/cantor_engine.h`.

## Loading

The model name is `minimax-music3`; the engine advertises CODES, DIFFUSE, and
DECODE. PLAN is intentionally absent. Supply exactly one GGUF for each role:

| Role | Component |
| --- | --- |
| `lm` | 36-layer global Qwen3 language model |
| `rvq` | four-layer residual depth decoder |
| `condition` | hidden-state acoustic condition projection |
| `dit` | 36-layer flow transformer |
| `vae` | stereo waveform decoder/vocoder |

Unknown and duplicate roles are rejected. All five component paths are
required before a stage executes. The engine hashes their exact bytes once per
context; every durable boundary carries all five SHA-256 values, so a state
cannot silently resume against a different quantization mix.

GGML chooses the best loaded device backend and keeps CPU as the fallback.
`n_threads` is honored. The remaining ABI v1 load fields are accepted for node
compatibility; MiniMax's fixed acoustic geometry does not currently expose a
VAE window override, and the imported graphs do not currently switch a
separate flash-attention or batched-CFG implementation.

## Stage boundaries

Every non-audio output is a private version-1 envelope with an eight-byte
magic, Cantor stage, section directory, explicit little-endian lengths, and a
SHA-256 over the entire envelope. The parser caps a blob at 2 GiB, rejects
truncation/overlap/checksum errors, requires the exact section set for its
state kind, and validates internal cursor/shape relationships before loading
weights.

| Boundary | Magic | Durable contents |
| --- | --- | --- |
| paused CODES | `MMXCOD01` | canonical request, `[T,8]` codes, target/replay cursor, component hashes |
| complete CODES | `MMXCON01` | canonical request, per-window pre-resize BF16 projected conditions, window cursor, hashes |
| paused DIFFUSE | `MMXFLW01` | complete CODES source, finished latents, current F32 Euler latent, overlap carries, window/step cursor, hashes |
| complete DIFFUSE | `MMXLAT01` | complete CODES source, all F32 latent windows, window count, hashes |

CODES cancellation occurs after an AR frame or before a condition window.
Resume replays the stored discrete prefix through LM/RVQ to rebuild KV caches
and hidden states while preserving the official position-addressed sampler.
A complete maximum-duration CODES boundary stores 89 projected windows and
72,908,800 condition bytes (about 69.5 MiB).

DIFFUSE cancellation occurs after an Euler update. Resume regenerates the
window's exact chunk-derived Philox noise and continues from the stored F32
latent and overlap state. Finished windows are not recomputed.

DECODE does not serialize a partially decoded waveform. On cancellation it
returns `CANTOR_PAUSED` with a null state output; the caller retries the same
immutable DIFFUSE boundary. On success, `cantor_engine_audio` returns planar
stereo F32 (`left[n_samples]` followed by `right[n_samples]`) at 44,100 Hz or
the explicitly requested 32,000 Hz.

## Ownership and errors

Stage blobs are allocated by the engine and released with
`cantor_engine_free_blob`. Audio remains owned by the context until the next
stage call or context destruction. All ABI entry points contain exceptions;
failure details are available through the thread-local error code/string.
`cantor_engine_resident_modules` reports the current stage module count and
`cantor_engine_resident_bytes` reports the selected backend's used-memory
snapshot when that backend exposes one.
