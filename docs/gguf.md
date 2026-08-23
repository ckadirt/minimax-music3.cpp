# GGUF contract

## Component artifacts

The Hugging Face repository contains five independently loadable component
families:

- LM and RVQ: BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M
- Condition encoder: F32
- DiT: F32, Q8_0, Q6_K, Q5_K_M, Q4_K_M
- VAE/vocoder: F32 and F16 storage; execution remains F32

Each GGUF embeds its architecture/config fields, source repository and pinned
revision, complete source-file SHA-256 inventory, converter and policy
revisions, logical tensor shapes, physical padding, and tokenizer assets when
owned by the LM.

The converter refuses unclassified tensors, shape/dtype changes, unsafe
payloads, non-finite values, in-place output, and existing output paths unless
an explicit staging-only overwrite flag is supplied. A strict loader verifies
the complete tensor name/type policy before allocating backend buffers.

## Quantization policy

Rank-zero/one tensors, normalization, learned time/Fourier embeddings,
convolutions, and numerically sensitive non-block projections remain F32.
Q8_0 and Q6_K use their base type for eligible matrices. Q5_K_M and Q4_K_M
promote embeddings, output heads, attention output projections, and FFN down
projections to Q6_K; other eligible matrices use Q5_K or Q4_K.

Quantized input axes are padded to the GGML block width in storage while
retaining the logical tensor shape in metadata. Runtime loaders reject an
artifact whose padding or tensor routing differs from its declared profile.

Every artifact has deterministic `.manifest.json` and `.sha256` sidecars.
Quantized manifests also record the exact baseline GGUF SHA-256 from which they
were derived.
