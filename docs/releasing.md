# Release contract

## Source and backend releases

The first release is `v0.1.0`. A tag builds separate CPU, CUDA, Vulkan, and
Metal/Android archives as applicable. GPU archives include the portable CPU
fallback. Each archive contains the Cantor engine library, GGML runtime
libraries, runnable CLI tools where supported, NOTICE/license files, a build
manifest, and SHA-256 sidecar.

Release workflows refuse to replace an existing GitHub asset. The Cantor
backend manifest uses immutable GitHub Release URLs and exact checksums. R2 is
not part of this repository's v0.1 release path.

## Model publication

GGUFs are staged outside Git and uploaded to
`ckadirt/MiniMax-Music3-GGUF` only after their committed validation matrix is
complete. Publication refuses an existing remote filename. After upload, the
publisher re-downloads public metadata/sidecars and verifies filename, byte
count, SHA-256, source revision, and quantization policy.

Credentials come from local environment variables or GitHub secrets. Their
values are never printed, written into manifests, or committed.

## Automation

`.github/workflows/ci.yml` builds weightless CPU, CUDA, Vulkan, Metal, Windows,
and Android configurations. GPU jobs prove compilation and exercise the CPU
fallback unless the runner has the corresponding device; this does not count
as real-model backend validation.

`.github/workflows/release.yml` runs for a published Release or an explicit
existing tag. It produces GitHub assets for Linux CPU/CUDA/Vulkan, macOS Metal,
Windows CPU/Vulkan, and Android CPU/Vulkan. Each job checks the 13-symbol ABI,
packages licenses and the public header, writes a SHA-256 sidecar, and aborts
if an asset of the same name already exists.

`.github/workflows/publish-model.yml` is manual and restricted to the labelled
self-hosted publisher because the pinned source plus conversion scratch needs
at least 140 GiB free. It downloads the exact source inventory, rebuilds all
18 GGUFs, validates the local publication set, and defaults to a no-mutation
dry run. A real upload additionally requires every evidence-bearing gate in
`validation/release-matrix.json` to be `pass` and `release_approved` to be
true. `convert/publish_hf.py` refuses remote filename collisions and verifies
remote sizes, LFS SHA-256 metadata, manifests, checksums, license, and model
card after upload.
