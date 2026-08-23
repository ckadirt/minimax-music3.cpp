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
