# Parity and acceptance policy

## Oracles

Diffusers is the component-level oracle. SGLang-Omni is the prompt, seeded
sampling, chunking, overlap, and 32 kHz serving oracle. Oracle scripts run in
evaluation mode with fixed inputs, fixed noise, explicit dtypes, and TF32
disabled for F32 correctness captures.

Large checkpoints and raw oracle tensors are ignored. Compact metric summaries,
fixture hashes, commands, and source revisions are committed.

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
