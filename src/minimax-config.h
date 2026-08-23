#pragma once

#include <cstdint>
#include <vector>

namespace minimax {

struct qwen_config {
    std::int64_t vocab_size = 200000;
    std::int64_t hidden_size = 4096;
    std::int64_t intermediate_size = 12288;
    std::int64_t layers = 36;
    std::int64_t attention_heads = 32;
    std::int64_t kv_heads = 8;
    std::int64_t head_dim = 128;
    std::int64_t max_positions = 10240;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct rvq_config {
    std::int64_t vocab_size = 1024;
    std::int64_t hidden_size = 4096;
    std::int64_t intermediate_size = 6144;
    std::int64_t max_positions = 16;
    std::int64_t attention_heads = 16;
    std::int64_t codebooks = 8;
    std::int64_t layers = 4;
    float rms_norm_eps = 1.0e-6F;
};

struct condition_config {
    std::int64_t hidden_size = 4096;
    std::int64_t hidden_layers = 8;
    std::int64_t output_size = 2048;
    std::int64_t input_sample_rate = 24000;
    std::int64_t input_hop = 960;
    std::int64_t output_sample_rate = 44100;
    std::int64_t output_hop = 512;
};

struct flow_config {
    std::int64_t channels = 128;
    std::int64_t condition_size = 2048;
    std::int64_t layers = 36;
    std::int64_t attention_heads = 32;
    std::int64_t head_dim = 64;
    std::int64_t ffn_size = 8192;
    std::int64_t rotary_dim = 32;
    std::int64_t fourier_dim = 256;
};

struct vocoder_config {
    std::int64_t latent_channels = 128;
    std::int64_t input_size = 1024;
    std::int64_t hidden_size = 1536;
    int sample_rate = 44100;
    std::int64_t hop = 512;
    std::vector<std::int64_t> upsample_ratios{8, 8, 4, 2};
};

struct model_config {
    qwen_config qwen;
    rvq_config rvq;
    condition_config condition;
    flow_config flow;
    vocoder_config vocoder;
    std::int64_t max_prompt_tokens = 5000;
    std::int64_t max_audio_frames = 9000;
    std::int64_t frame_rate = 25;
    std::int64_t window_frames = 200;
    std::int64_t window_hop = 100;
    std::int64_t overlap_latents = 172;
};

inline const model_config default_model_config{};

} // namespace minimax
