#include "engine/community_models/minimax_music3/seed.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace engine::models::minimax_music3 {
namespace {

constexpr std::array<std::uint64_t, 8> kBlake2bIv = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

constexpr std::uint8_t kBlake2bSigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

std::uint64_t load_le64(const std::uint8_t * input) {
    std::uint64_t out = 0;
    for (unsigned index = 0; index != 8; ++index) {
        out |= static_cast<std::uint64_t>(input[index]) << (8U * index);
    }
    return out;
}

void store_le64(std::uint8_t * output, std::uint64_t value) {
    for (unsigned index = 0; index != 8; ++index) {
        output[index] = static_cast<std::uint8_t>(value >> (8U * index));
    }
}

std::uint64_t rotate_right(std::uint64_t value, unsigned amount) {
    return (value >> amount) | (value << (64U - amount));
}

void blake2b_mix(
    std::uint64_t & a,
    std::uint64_t & b,
    std::uint64_t & c,
    std::uint64_t & d,
    std::uint64_t x,
    std::uint64_t y) {
    a = a + b + x;
    d = rotate_right(d ^ a, 32);
    c += d;
    b = rotate_right(b ^ c, 24);
    a = a + b + y;
    d = rotate_right(d ^ a, 16);
    c += d;
    b = rotate_right(b ^ c, 63);
}

void blake2b_compress(
    std::array<std::uint64_t, 8> & state,
    const std::uint8_t block[128],
    std::uint64_t byte_count,
    bool last) {
    std::array<std::uint64_t, 16> message{};
    std::array<std::uint64_t, 16> work{};
    for (std::size_t index = 0; index != message.size(); ++index) {
        message[index] = load_le64(block + index * 8);
    }
    for (std::size_t index = 0; index != 8; ++index) {
        work[index] = state[index];
        work[index + 8] = kBlake2bIv[index];
    }
    work[12] ^= byte_count;
    if (last) work[14] = ~work[14];
    for (std::size_t round = 0; round != 12; ++round) {
        const auto & sigma = kBlake2bSigma[round];
        blake2b_mix(work[0], work[4], work[8], work[12], message[sigma[0]], message[sigma[1]]);
        blake2b_mix(work[1], work[5], work[9], work[13], message[sigma[2]], message[sigma[3]]);
        blake2b_mix(work[2], work[6], work[10], work[14], message[sigma[4]], message[sigma[5]]);
        blake2b_mix(work[3], work[7], work[11], work[15], message[sigma[6]], message[sigma[7]]);
        blake2b_mix(work[0], work[5], work[10], work[15], message[sigma[8]], message[sigma[9]]);
        blake2b_mix(work[1], work[6], work[11], work[12], message[sigma[10]], message[sigma[11]]);
        blake2b_mix(work[2], work[7], work[8], work[13], message[sigma[12]], message[sigma[13]]);
        blake2b_mix(work[3], work[4], work[9], work[14], message[sigma[14]], message[sigma[15]]);
    }
    for (std::size_t index = 0; index != state.size(); ++index) {
        state[index] ^= work[index] ^ work[index + 8];
    }
}

std::uint64_t blake2b_64(
    const std::vector<std::uint8_t> & input,
    const std::array<std::uint8_t, 16> & personal) {
    std::array<std::uint8_t, 64> parameter{};
    parameter[0] = 8;
    parameter[2] = 1;
    parameter[3] = 1;
    std::copy(personal.begin(), personal.end(), parameter.begin() + 48);
    auto state = kBlake2bIv;
    for (std::size_t index = 0; index != state.size(); ++index) {
        state[index] ^= load_le64(parameter.data() + index * 8);
    }

    std::size_t offset = 0;
    std::uint64_t total = 0;
    while (input.size() - offset > 128) {
        total += 128;
        blake2b_compress(state, input.data() + offset, total, false);
        offset += 128;
    }
    std::array<std::uint8_t, 128> final_block{};
    const std::size_t final_size = input.size() - offset;
    if (final_size != 0) {
        std::memcpy(final_block.data(), input.data() + offset, final_size);
    }
    total += static_cast<std::uint64_t>(final_size);
    blake2b_compress(state, final_block.data(), total, true);
    return state[0];
}

void append_le32(std::vector<std::uint8_t> & output, std::uint32_t value) {
    for (unsigned index = 0; index != 4; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    }
}

void append_le64(std::vector<std::uint8_t> & output, std::uint64_t value) {
    for (unsigned index = 0; index != 8; ++index) {
        output.push_back(static_cast<std::uint8_t>(value >> (8U * index)));
    }
}

std::uint32_t rotate_left32(std::uint32_t value, unsigned amount) {
    return (value << amount) | (value >> (32U - amount));
}

std::uint32_t murmur_mix(std::uint32_t hash, std::uint32_t key) {
    key *= 0xcc9e2d51U;
    key = rotate_left32(key, 15);
    key *= 0x1b873593U;
    hash ^= key;
    hash = rotate_left32(hash, 13);
    return hash * 5U + 0xe6546b64U;
}

std::uint32_t murmur_final(std::uint32_t hash) {
    hash ^= hash >> 16U;
    hash *= 0x85ebca6bU;
    hash ^= hash >> 13U;
    hash *= 0xc2b2ae35U;
    hash ^= hash >> 16U;
    return hash;
}

double gumbel_from_hash(std::uint32_t hash) {
    const double uniform = static_cast<double>(hash) /
        static_cast<double>(std::numeric_limits<std::uint32_t>::max());
    double log_uniform = std::log(uniform);
    if (log_uniform < std::numeric_limits<double>::lowest()) {
        log_uniform = std::numeric_limits<double>::lowest();
    }
    return -std::log(-log_uniform);
}

}  // namespace

std::uint32_t derive_ar_sampling_seed(std::uint64_t public_seed) {
    const std::string material = "minimax-ttm-ar:" + std::to_string(public_seed);
    std::vector<std::uint8_t> input(material.begin(), material.end());
    const std::array<std::uint8_t, 16> no_personal{};
    return static_cast<std::uint32_t>(blake2b_64(input, no_personal) & 0x7fffffffULL);
}

std::uint64_t derive_dit_chunk_seed(std::uint64_t public_seed, std::size_t chunk_index) {
    std::vector<std::uint8_t> input;
    append_le64(input, public_seed);
    const std::string label = "dit";
    append_le32(input, static_cast<std::uint32_t>(label.size()));
    input.insert(input.end(), label.begin(), label.end());
    const std::string index = std::to_string(chunk_index);
    append_le32(input, static_cast<std::uint32_t>(index.size()));
    input.insert(input.end(), index.begin(), index.end());
    std::array<std::uint8_t, 16> personal{};
    constexpr char kPersonal[] = "minimax-ttm";
    std::copy(kPersonal, kPersonal + sizeof(kPersonal) - 1, personal.begin());
    return blake2b_64(input, personal) & 0x7fffffffffffffffULL;
}

std::uint32_t murmur_sampling_hash(
    std::uint64_t sampling_seed,
    std::uint32_t position,
    std::uint32_t column) {
    std::uint32_t hash = 0;
    hash = murmur_mix(hash, static_cast<std::uint32_t>(sampling_seed));
    hash = murmur_mix(hash, static_cast<std::uint32_t>(sampling_seed >> 32U));
    hash = murmur_mix(hash, position);
    hash = murmur_mix(hash, column);
    return murmur_final(hash ^ 16U);
}

std::size_t seeded_gumbel_argmax(
    const std::vector<float> & scores,
    std::uint64_t sampling_seed,
    std::uint32_t position,
    const std::vector<std::uint32_t> * columns) {
    if (scores.empty()) throw std::runtime_error("seeded Gumbel sampler requires scores");
    if (columns != nullptr && columns->size() != scores.size()) {
        throw std::runtime_error("seeded Gumbel sampler column shape mismatch");
    }
    std::size_t selected = 0;
    double best = -std::numeric_limits<double>::infinity();
    bool found = false;
    for (std::size_t index = 0; index != scores.size(); ++index) {
        if (std::isnan(scores[index])) return index;
        if (!std::isfinite(scores[index])) continue;
        const std::uint32_t column = columns == nullptr
            ? static_cast<std::uint32_t>(index)
            : (*columns)[index];
        const double ranked = static_cast<double>(scores[index]) +
            gumbel_from_hash(murmur_sampling_hash(sampling_seed, position, column));
        if (!found || ranked > best) {
            found = true;
            best = ranked;
            selected = index;
        }
    }
    if (!found) throw std::runtime_error("seeded Gumbel sampler has no finite scores");
    return selected;
}

}  // namespace engine::models::minimax_music3
