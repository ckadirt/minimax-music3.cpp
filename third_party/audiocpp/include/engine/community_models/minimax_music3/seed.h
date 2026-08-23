#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::minimax_music3 {

// These functions mirror the pinned SGLang-Omni MiniMax Music 3 serving
// implementation. Keeping sampling position-addressed makes a request stable
// across batching and all GGML backends.
std::uint32_t derive_ar_sampling_seed(std::uint64_t public_seed);
std::uint64_t derive_dit_chunk_seed(std::uint64_t public_seed, std::size_t chunk_index);
std::uint32_t murmur_sampling_hash(
    std::uint64_t sampling_seed,
    std::uint32_t position,
    std::uint32_t column);

std::size_t seeded_gumbel_argmax(
    const std::vector<float> & scores,
    std::uint64_t sampling_seed,
    std::uint32_t position,
    const std::vector<std::uint32_t> * columns = nullptr);

}  // namespace engine::models::minimax_music3
