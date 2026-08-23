# Third-party notices

## MiniMax Music 3

The model architecture, configuration, tokenizer, and weights originate from
MiniMax. Checkpoint-derived artifacts remain governed by the
[MiniMax Music 3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/fbdf52fbaaca799592917417eb05f1899f1255ec/LICENSE).

## audio.cpp

The native model implementation is adapted from
[`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) at revision
`62735eafd96294c52d6c4607f5f38ac55be54f06`, Copyright 2026 ShugoAI LLC,
under Apache-2.0. Adapted files must retain a modification notice.
The compiled, pruned source snapshot and its portability modifications are
documented in `third_party/audiocpp/README.md`; its license is retained there.

## Hugging Face Diffusers and SGLang-Omni

The numerical oracle and architecture review use Hugging Face Diffusers at
`dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d` and SGLang-Omni at
`5b2d8852fbed855fc07ce04eb1f105fa4294f34f`. Their code is not bundled unless
an adapted source file explicitly says otherwise.

## GGML

GGML is consumed as a pinned submodule from `ckadirt/ggml` at
`70081fdfc8685b60477b54d9d11cd679c5a00cb1` and retains its upstream license.
