# Audited audio.cpp source subset

This directory contains the framework and MiniMax Music 3 implementation
adapted from `0xShug0/audio.cpp` at revision
`62735eafd96294c52d6c4607f5f38ac55be54f06` (Apache-2.0). It is copied source,
not a runtime dependency or fetched build input.

Only the framework, MiniMax model, and small tokenizer/config dependencies are
retained. Unrelated models and SentencePiece are excluded. Local portability
changes are deliberately small and searchable:

- graph compaction uses `ggml_graph_clear`/`ggml_graph_add_node`;
- packed CUDA projection lowers through portable `ggml_mul_mat`;
- the custom fast 1-D convolution lowers through portable `ggml_conv_1d`;
- unrelated BigVGAN and relative-attention paths requiring post-pin GGML ops
  are not compiled.

See the root `NOTICE`, `THIRD_PARTY_NOTICES.md`, and this directory's `LICENSE`.
