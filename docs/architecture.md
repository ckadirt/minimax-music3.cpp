# Architecture contract

## Model stages

MiniMax Music 3 generates one autoregressive music frame at 25 frames per
second. A 36-layer Qwen3 model predicts the semantic code, then a four-layer
depth decoder predicts seven residual RVQ codes. Synthesis consumes the eight
4096-wide final hidden states, not only the discrete codes.

For each acoustic window, the condition encoder learns a weighted sum over the
eight hidden states, applies a 4096-to-2048 convolution, and nearest-neighbor
resizes the result to the 44.1 kHz latent timeline. A 36-layer flow transformer
integrates 128-channel latents. The vocoder upsamples each latent frame by 512
samples and produces stereo 44.1 kHz audio.

| Property | Value |
| --- | ---: |
| Prompt limit | 5,000 tokens |
| AR context | 10,240 positions |
| Maximum audio frames | 9,000 |
| Maximum duration | 300 seconds |
| RVQ codebooks | 8 |
| Semantic vocabulary | 16,384 entries |
| Residual vocabulary | 1,024 entries |
| Acoustic window/hop | 200 / 100 AR frames |
| Default solver | 30 Euler steps |
| Default AR/flow CFG | 1.5 / 1.7 |
| Native output | 44,100 Hz stereo |

## Residency and durable boundaries

The runtime owns five independently selectable components: `lm`, `rvq`,
`condition`, `dit`, and `vae`.

- CODES loads LM/RVQ and, after AR completion, condition projection weights.
  A paused prefix stores codes and RNG state and rebuilds KV state by replay.
- A completed CODES state stores window-specific, pre-resize BF16 condition
  projections. This is mathematically equivalent to rounding the resized
  nearest-neighbor output and reduces the five-minute boundary from roughly
  562.5 MiB of raw BF16 hidden states to roughly 69.5 MiB.
- DIFFUSE loads only the DiT and stores completed latents, the current Euler
  position, overlap carry, RNG state, settings, and component identities.
- DECODE loads only the VAE. Cancellation retries from the durable DIFFUSE
  boundary rather than serializing a partial waveform graph.

Every boundary is versioned, little-endian, length-delimited, checksummed, and
stamped with the exact component SHA-256 values and canonical request.

## Cantor request v1

The JSON request is a closed schema. Unknown keys, duplicate keys, invalid
UTF-8, lossy integer spellings, and non-finite numbers are rejected. Lyrics,
description, and `duration_seconds` are required; all other values use the
defaults shown here. The optional flow seed remains absent until the runtime
applies the pinned upstream seed derivation.

```json
{
  "lyrics": "[Verse]\n...",
  "description": "warm synth-pop, female vocal",
  "duration_seconds": 20,
  "seed": 0,
  "cfg_scale": 1.5,
  "sampling": { "top_k": 50 },
  "flow": { "seed": 0, "euler_steps": 30, "cfg_scale": 1.7 },
  "output_sample_rate": 44100
}
```

Duration is limited to 300 seconds. The supported output rates are the native
44,100 Hz and the Cantor serving rate of 32,000 Hz.
