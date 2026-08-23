# MiniMax Music 3 port plan

## Frozen inputs

| Source | Revision | Purpose |
| --- | --- | --- |
| `MiniMaxAI/MiniMax-Music3` | `fbdf52fbaaca799592917417eb05f1899f1255ec` | Checkpoint, tokenizer, configs, license |
| `huggingface/diffusers` | `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d` | Componentized Python oracle |
| `sgl-project/sglang-omni` | `5b2d8852fbed855fc07ce04eb1f105fa4294f34f` | Serving prompt, RNG, chunking oracle |
| `0xShug0/audio.cpp` | `62735eafd96294c52d6c4607f5f38ac55be54f06` | Audited native implementation seed |
| `ckadirt/ggml` | `70081fdfc8685b60477b54d9d11cd679c5a00cb1` | Portable compute backends |

The converter downloads only the componentized inference files. The legacy
`qwen_7B`, `dav.pth`, `flowmatching_vae.pth`, demo assets, and training-only
files are excluded.

## Milestones

1. Establish repository safety, licenses, architecture/GGUF/parity contracts,
   and an append-only implementation report.
2. Add the pinned GGML build, backend enumeration, model-independent tests,
   tokenizer, prompt normalization, and strict request parsing.
3. Add a shard-streaming converter with an exact tensor inventory,
   deterministic manifests, checksums, and strict GGUF loaders.
4. Port and gate the global Qwen3 LM and RVQ depth decoder, including official
   seeded sampling and incremental KV-cache execution.
5. Port and gate the condition projection, flow transformer/Euler solver,
   overlapping acoustic windows, vocoder, and WAV/provenance writers.
6. Implement Cantor ABI v1 with five roles and durable CODES, DIFFUSE, and
   DECODE boundaries.
7. Produce the F32/BF16 baseline and Q8_0, Q6_K, Q5_K_M, and Q4_K_M component
   matrix. Benchmark and publish only artifacts that pass their gates.
8. Build immutable GitHub Release backends for Linux, Windows, macOS, and
   Android; publish GGUF components to `ckadirt/MiniMax-Music3-GGUF`.

## Commit policy

Each milestone is committed only after its relevant tests pass. Generated
weights, audio, large oracle tensors, credentials, and local build trees remain
outside Git. The implementation report records the command, source revision,
artifact hashes, validation result, and next step so work can resume after an
interruption.

## Hardware transition

CPU work covers scaffolding, conversion inventory, strict loading, primitives,
and small graphs. Dense real-model parity begins on one H100 with 80 GiB VRAM.
CUDA is not required before that gate.
