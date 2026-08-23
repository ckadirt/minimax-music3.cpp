# Parity and acceptance policy

## Oracles

Diffusers is the component-level oracle. SGLang-Omni is the prompt, seeded
sampling, chunking, overlap, and 32 kHz serving oracle. Oracle scripts run in
evaluation mode with fixed inputs, fixed noise, explicit dtypes, and TF32
disabled for F32 correctness captures.

Large checkpoints and raw oracle tensors are ignored. Compact metric summaries,
fixture hashes, commands, and source revisions are committed.

The dense component gate is executable, rather than a prose-only checklist:

```bash
# Install a CUDA-matched PyTorch >= 2.6 wheel first.
python3 -m pip install -r parity/requirements-gpu.txt
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DMINIMAX_DYNAMIC_BACKENDS=ON \
  -DMINIMAX_BUILD_PARITY_TOOLS=ON
cmake --build build-cuda --config Release -j
python3 parity/run_dense.py \
  --source models/MiniMax-Music3-source \
  --gguf artifacts/gguf \
  --binary build-cuda/bin/minimax-parity

# The request must use at least two Euler steps so the harness can pause after
# a completed update and resume from the serialized F32 latent.
build-cuda/bin/minimax-cantor-smoke \
  --model artifacts/gguf \
  --request artifacts/cantor-smoke-request.json \
  --threads 4
```

`fixture-spec.json` freezes the random-input generator, shapes, revisions, and
thresholds. The oracle loads one F32 Diffusers component at a time, disables
TF32, and records hashes before unloading it. The native runner then executes
the dense condition, two-branch DiT velocity, and planar stereo vocoder
boundaries through the selected GGML backend. `compare.py` rejects shape,
non-finite, maximum-error, or relative-RMS failures and emits a compact hashed
summary. Raw `.f32` files remain under the ignored `artifacts/` tree.
`minimax-cantor-smoke` separately proves that CODES and DIFFUSE resumed
boundaries are byte-identical to uninterrupted execution, and that a cancelled
DECODE retains no partial audio while a retry returns byte-identical planar
F32 output. It rejects non-finite or silent audio and prints a compact result
record suitable for release evidence.
The DiT component runner also executes every fixture twice and requires an
exact match before comparing with Python. This guards the GGML graph against
destructive reuse and stale backend graph-cache state.

## Initial numerical gates

| Boundary | Gate |
| --- | --- |
| F32 primitive/block | `atol=1e-4`, `rtol=1e-4` |
| BF16 AR hidden/logits | cosine >= `0.9999`, normalized RMSE <= `1e-2` |
| Final AR logits | maximum absolute error <= `0.1` |
| Condition projection | maximum absolute error <= `5e-5` |
| Flow block 0 | maximum error <= `2e-3`, cosine > `0.99999` |
| Full velocity | maximum error <= `5e-3`, relative RMS < `3e-3` |
| Final latent | maximum error <= `2e-2`, relative RMS < `5e-3` |
| Vocoder waveform | maximum error <= `3e-3`, relative RMS < `1e-3` |

Tokenizer IDs, prompt normalization, special-token placement, official random
draws, and `top_k=1` dense codes must match exactly. Thresholds may not be
silently relaxed; a change needs earliest-divergence evidence and a dedicated
reviewable commit.

## Release cases

- Minimal prompt and invalid/empty/instrumental prompt cases.
- Two seconds, ten seconds, and thirty seconds with exact dense AR fixtures.
- One-window, exact-window, overlap-boundary, and multiple-window synthesis.
- One Euler step for diagnostics and thirty steps for release.
- Early EOS and the 9,000-frame maximum-duration path.
- Pause/resume before and after every durable boundary and at representative
  AR frame, flow window, and Euler-step positions.
- CPU, CUDA, Vulkan, Metal, and Android backend smoke. A backend is described
  as real-model validated only after a hardware run, not merely compilation.

Quantized output is approximate. Each profile must load strictly, complete
short CPU and CUDA smokes, produce finite non-silent stereo audio, and publish
its isolated token/latent/waveform metrics plus listening notes.
