#include "engine/community_models/minimax_music3/ar_runtime.h"
#include "engine/community_models/minimax_music3/seed.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;

struct ArBranchState {
    std::vector<float> hidden;
};

struct ArStepState {
    std::vector<float> logits;
    ArBranchState cond;
    ArBranchState uncond;
};

float top_k_threshold(
    const std::vector<float> & values,
    std::vector<float> & window,
    int64_t offset,
    int64_t vocab_size,
    int64_t start,
    int64_t count,
    int64_t end_token,
    int64_t top_k) {
    if (offset < 0 || vocab_size <= 0 || static_cast<int64_t>(values.size()) < offset + vocab_size) {
        throw std::runtime_error("MiniMax Music 3 AR logits shape mismatch");
    }
    const int64_t allowed = count + (end_token >= 0 ? 1 : 0);
    if (top_k <= 0 || top_k >= allowed) {
        return -std::numeric_limits<float>::infinity();
    }
    window.clear();
    window.reserve(static_cast<size_t>(allowed));
    for (int64_t i = 0; i < count; ++i) {
        window.push_back(values[static_cast<size_t>(offset + start + i)]);
    }
    if (end_token >= 0 && end_token < vocab_size) {
        window.push_back(values[static_cast<size_t>(offset + end_token)]);
    }
    std::nth_element(
        window.begin(),
        window.begin() + static_cast<std::ptrdiff_t>(top_k - 1),
        window.end(),
        std::greater<float>());
    return window[static_cast<size_t>(top_k - 1)];
}

std::vector<int32_t> semantic_logits_readback_token_ids(const MiniMaxMusic3Prompt & prompt) {
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(prompt.semantic_vocab_size + 1));
    for (int64_t i = 0; i < prompt.semantic_vocab_size; ++i) {
        out.push_back(static_cast<int32_t>(prompt.audio_code_offset + i));
    }
    out.push_back(prompt.audio_end_token_id);
    return out;
}

int32_t compact_semantic_token_id(const MiniMaxMusic3Prompt & prompt, int64_t compact_index) {
    if (compact_index < 0 || compact_index > prompt.semantic_vocab_size) {
        throw std::runtime_error("MiniMax Music 3 compact semantic token index mismatch");
    }
    if (compact_index == prompt.semantic_vocab_size) {
        return prompt.audio_end_token_id;
    }
    return static_cast<int32_t>(prompt.audio_code_offset + compact_index);
}

int32_t sample_compact_semantic_token(
    const MiniMaxMusic3Prompt & prompt,
    const std::vector<float> & batch2_logits,
    int64_t vocab_size,
    std::vector<float> & logits,
    std::vector<float> & topk_window,
    std::vector<int32_t> & candidates,
    std::vector<double> & weights,
    const MiniMaxMusic3Request & request,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks) {
    const int64_t compact_size = prompt.semantic_vocab_size + 1;
    if (vocab_size <= 0 || static_cast<int64_t>(batch2_logits.size()) != 2 * compact_size) {
        throw std::runtime_error("MiniMax Music 3 compact AR logits branch size mismatch");
    }
    logits.assign(static_cast<size_t>(compact_size), -std::numeric_limits<float>::infinity());
    constexpr int64_t kCondRow = 0;
    const int64_t uncond_row = compact_size;
    const float threshold = top_k_threshold(
        batch2_logits,
        topk_window,
        kCondRow,
        compact_size,
        0,
        prompt.semantic_vocab_size,
        compact_size - 1,
        request.top_k);
    for (int64_t index = 0; index < compact_size; ++index) {
        const float cond = batch2_logits[static_cast<size_t>(index)];
        if (cond >= threshold) {
            const float uncond = batch2_logits[static_cast<size_t>(uncond_row + index)];
            logits[static_cast<size_t>(index)] = uncond + (cond - uncond) * request.ar_guidance_scale;
        }
    }
    const float combined_threshold = top_k_threshold(
        logits,
        topk_window,
        0,
        compact_size,
        0,
        compact_size,
        -1,
        request.top_k);
    candidates.clear();
    candidates.reserve(logits.size());
    weights.clear();
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < logits.size(); ++index) {
        if (std::isfinite(logits[index]) && logits[index] >= combined_threshold) {
            candidates.push_back(static_cast<int32_t>(index));
            max_score = std::max(max_score, logits[index]);
        }
    }
    if (candidates.empty() || !std::isfinite(max_score)) {
        throw std::runtime_error("MiniMax Music 3 compact semantic sampler has no finite logits");
    }
    weights.reserve(candidates.size());
    std::vector<std::uint32_t> columns;
    columns.reserve(candidates.size());
    for (const int32_t index : candidates) {
        weights.push_back(static_cast<double>(logits[static_cast<size_t>(index)]));
        columns.push_back(index == prompt.semantic_vocab_size
            ? 0U
            : static_cast<std::uint32_t>(index + 1));
    }
    std::vector<float> scores(weights.begin(), weights.end());
    const auto selected = seeded_gumbel_argmax(
        scores,
        derive_ar_sampling_seed(request.seed),
        static_cast<std::uint32_t>(sample_call_index),
        &columns);
    const int32_t selected_compact = candidates[selected];
    ++sample_call_index;
    (void)rng_offset_blocks;
    (void)vocab_size;
    return compact_semantic_token_id(prompt, selected_compact);
}

int32_t sample_semantic_token(
    const MiniMaxMusic3Prompt & prompt,
    const std::vector<float> & batch2_logits,
    int64_t vocab_size,
    std::vector<float> & logits,
    std::vector<float> & topk_window,
    std::vector<int32_t> & compact_candidates,
    const MiniMaxMusic3Request & request,
    uint64_t & sample_call_index,
    uint64_t & rng_offset_blocks,
    sampling::HfSamplerScratch & scratch,
    std::vector<double> & weights) {
    const int64_t compact_size = prompt.semantic_vocab_size + 1;
    if (static_cast<int64_t>(batch2_logits.size()) == 2 * compact_size) {
        return sample_compact_semantic_token(
            prompt,
            batch2_logits,
            vocab_size,
            logits,
            topk_window,
            compact_candidates,
            weights,
            request,
            sample_call_index,
            rng_offset_blocks);
    }
    if (vocab_size <= 0 || static_cast<int64_t>(batch2_logits.size()) != 2 * vocab_size) {
        throw std::runtime_error("MiniMax Music 3 AR logits branch size mismatch");
    }
    logits.assign(static_cast<size_t>(vocab_size), -std::numeric_limits<float>::infinity());
    const int64_t start = prompt.audio_code_offset;
    const int64_t count = prompt.semantic_vocab_size;
    constexpr int64_t kCondRow = 0;
    const int64_t uncond_row = vocab_size;
    const float threshold = top_k_threshold(
        batch2_logits,
        topk_window,
        kCondRow,
        vocab_size,
        start,
        count,
        prompt.audio_end_token_id,
        request.top_k);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t token = start + i;
        const float cond = batch2_logits[static_cast<size_t>(token)];
        if (cond >= threshold) {
            const float uncond = batch2_logits[static_cast<size_t>(uncond_row + token)];
            logits[static_cast<size_t>(token)] = uncond + (cond - uncond) * request.ar_guidance_scale;
        }
    }
    if (prompt.audio_end_token_id >= 0 && prompt.audio_end_token_id < vocab_size) {
        const float cond = batch2_logits[static_cast<size_t>(prompt.audio_end_token_id)];
        const float uncond = batch2_logits[static_cast<size_t>(uncond_row + prompt.audio_end_token_id)];
        if (cond >= threshold) {
            logits[static_cast<size_t>(prompt.audio_end_token_id)] =
                uncond + (cond - uncond) * request.ar_guidance_scale;
        }
    }
    sampling::HfLogitsProcessor::apply_top_k(logits, request.top_k, 1, scratch);
    compact_candidates.clear();
    weights.clear();
    std::vector<std::uint32_t> columns;
    for (int64_t code = 0; code < count; ++code) {
        const int64_t token = start + code;
        if (!std::isfinite(logits[static_cast<std::size_t>(token)])) continue;
        compact_candidates.push_back(static_cast<int32_t>(token));
        weights.push_back(static_cast<double>(logits[static_cast<std::size_t>(token)]));
        columns.push_back(static_cast<std::uint32_t>(code + 1));
    }
    if (prompt.audio_end_token_id >= 0 && prompt.audio_end_token_id < vocab_size &&
        std::isfinite(logits[static_cast<std::size_t>(prompt.audio_end_token_id)])) {
        compact_candidates.push_back(prompt.audio_end_token_id);
        weights.push_back(static_cast<double>(logits[static_cast<std::size_t>(prompt.audio_end_token_id)]));
        columns.push_back(0U);
    }
    std::vector<float> scores(weights.begin(), weights.end());
    const auto selected = seeded_gumbel_argmax(
        scores,
        derive_ar_sampling_seed(request.seed),
        static_cast<std::uint32_t>(sample_call_index),
        &columns);
    const int32_t token = compact_candidates[selected];
    ++sample_call_index;
    (void)rng_offset_blocks;
    return token;
}

void assign_batch2_row(const std::vector<float> & values, int64_t row, int64_t width, std::vector<float> & out) {
    if (row < 0 || row >= 2 || width <= 0 || static_cast<int64_t>(values.size()) != 2 * width) {
        throw std::runtime_error("MiniMax Music 3 batch-2 output shape mismatch");
    }
    out.assign(
        values.begin() + static_cast<std::ptrdiff_t>(row * width),
        values.begin() + static_cast<std::ptrdiff_t>((row + 1) * width));
}

void duplicate_feedback_batch2(const std::vector<float> & feedback, std::vector<float> & out) {
    out.resize(feedback.size() * 2);
    std::copy(feedback.begin(), feedback.end(), out.begin());
    std::copy(feedback.begin(), feedback.end(), out.begin() + static_cast<std::ptrdiff_t>(feedback.size()));
}

std::vector<int32_t> batch2_prompt_ids(const MiniMaxMusic3Prompt & prompt) {
    if (prompt.conditional_ids.empty() ||
        prompt.conditional_ids.size() != prompt.unconditional_ids.size()) {
        throw std::runtime_error("MiniMax Music 3 prompt branch token shape mismatch");
    }
    std::vector<int32_t> out;
    out.reserve(prompt.conditional_ids.size() + prompt.unconditional_ids.size());
    out.insert(out.end(), prompt.conditional_ids.begin(), prompt.conditional_ids.end());
    out.insert(out.end(), prompt.unconditional_ids.begin(), prompt.unconditional_ids.end());
    return out;
}

}  // namespace

struct MiniMaxMusic3ArRuntime::Impl {
    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & input_execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          execution(input_execution),
          prompt_builder(assets),
          global_weights(load_minimax_music3_global_lm_weights(*assets, execution, weight_context_bytes, storage_type)),
          depth(std::make_unique<MiniMaxMusic3DepthDecoderRuntime>(
              assets,
              global_weights.token_embedding,
              execution,
              graph_arena_bytes,
              weight_context_bytes,
              storage_type)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 AR runtime requires assets");
        }
        auto qwen_config = make_minimax_music3_global_lm_runtime_config(
            assets->config,
            execution.backend_type(),
            graph_arena_bytes,
            graph_arena_bytes);
        qwen_config.return_hidden = true;
        qwen_config.logits_readback_token_ids = semantic_logits_readback_token_ids(MiniMaxMusic3Prompt{});
        global_runtime = std::make_unique<modules::QwenCausalDecodeRuntime>(
            execution,
            qwen_config,
            global_weights.qwen);
        semantic_scratch.reserve_vocab(static_cast<size_t>(assets->config.qwen.vocab_size));
    }

    MiniMaxMusic3Prompt prompt_from_request(const MiniMaxMusic3Request & request) const {
        return prompt_builder.build(request.prompt, request.lyrics);
    }

    void assign_step_outputs(
        std::vector<float> logits,
        const std::vector<float> & hidden,
        ArStepState & state) const {
        const int64_t compact_size = MiniMaxMusic3Prompt{}.semantic_vocab_size + 1;
        const int64_t logits_width = static_cast<int64_t>(logits.size()) / 2;
        if (logits.size() % 2 != 0 ||
            (logits_width != assets->config.qwen.vocab_size && logits_width != compact_size)) {
            throw std::runtime_error("MiniMax Music 3 AR logits batch shape mismatch");
        }
        state.logits = std::move(logits);
        assign_batch2_row(hidden, 0, assets->config.qwen.hidden_size, state.cond.hidden);
        assign_batch2_row(hidden, 1, assets->config.qwen.hidden_size, state.uncond.hidden);
    }

    MiniMaxMusic3ArResult generate_frame_hiddens_resumable(
        const MiniMaxMusic3Request & request,
        int64_t target_frames,
        const std::vector<int32_t> & replay_codes,
        const std::function<bool(std::size_t, std::size_t)> & should_pause,
        uint64_t & rng_offset_blocks) {
        const std::size_t codebooks = static_cast<std::size_t>(assets->config.depth.codebooks);
        if (replay_codes.size() % codebooks != 0 || replay_codes.size() / codebooks >
            static_cast<std::size_t>(target_frames + 1)) {
            throw std::runtime_error("MiniMax Music 3 replay code shape mismatch");
        }
        const auto ar_start = Clock::now();
        const auto prompt = prompt_from_request(request);
        const int64_t prompt_steps = static_cast<int64_t>(prompt.conditional_ids.size());
        const int64_t required_cache_steps = std::max<int64_t>(1, prompt_steps + target_frames);
        auto prefill = global_runtime->prefill_tokens_batched(batch2_prompt_ids(prompt), 2, prompt_steps);
        global_runtime->start_decode_embeddings_batched(prefill.state, required_cache_steps);

        ArStepState state;
        assign_step_outputs(std::move(prefill.logits), prefill.hidden, state);

        MiniMaxMusic3ArResult result;
        result.frame_hiddens.reserve(static_cast<size_t>(
            target_frames * assets->config.condition.condition_layers * assets->config.qwen.hidden_size));
        result.codes.reserve(static_cast<std::size_t>(target_frames + 1) * codebooks);
        uint64_t sample_call_index = 0;
        for (int64_t frame = 0; frame <= target_frames; ++frame) {
            if (should_pause && should_pause(
                    static_cast<std::size_t>(frame),
                    static_cast<std::size_t>(target_frames + 1))) {
                result.completed = false;
                break;
            }
            const bool replay = static_cast<std::size_t>(frame) < replay_codes.size() / codebooks;
            const int32_t token = replay
                ? prompt.audio_code_offset + replay_codes[static_cast<std::size_t>(frame) * codebooks]
                : sample_semantic_token(
                      prompt,
                      state.logits,
                      assets->config.qwen.vocab_size,
                      semantic_logits,
                      topk_window,
                      compact_candidates,
                      request,
                      sample_call_index,
                      rng_offset_blocks,
                      semantic_scratch,
                      semantic_weights);
            if (replay) ++sample_call_index;
            if (token == prompt.audio_end_token_id) {
                result.completed = true;
                break;
            }
            if (token < prompt.audio_code_offset ||
                token >= prompt.audio_code_offset + prompt.semantic_vocab_size) {
                throw std::runtime_error("MiniMax Music 3 sampled token outside semantic audio range");
            }
            const int32_t semantic_code = token - prompt.audio_code_offset;
            std::vector<int32_t> forced;
            if (replay) {
                const auto begin = replay_codes.begin() + static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(frame) * codebooks);
                forced.assign(begin, begin + static_cast<std::ptrdiff_t>(codebooks));
            }
            auto depth_codes = depth->generate(
                state.cond.hidden,
                state.uncond.hidden,
                semantic_code,
                request.ar_guidance_scale,
                request.top_k,
                request.seed,
                sample_call_index,
                rng_offset_blocks,
                replay ? &forced : nullptr);
            result.codes.insert(result.codes.end(), depth_codes.codes.begin(), depth_codes.codes.end());
            if (frame > 0) {
                result.frame_hiddens.insert(
                    result.frame_hiddens.end(), state.cond.hidden.begin(), state.cond.hidden.end());
                result.frame_hiddens.insert(
                    result.frame_hiddens.end(), depth_codes.hidden.begin(), depth_codes.hidden.end());
            }
            if (frame == target_frames) {
                result.completed = true;
                break;
            }
            const auto feedback = depth->feedback_embedding(depth_codes.codes);
            duplicate_feedback_batch2(feedback, feedback_batch);
            auto step = global_runtime->decode_embeddings_batched(feedback_batch, 2);
            assign_step_outputs(std::move(step.logits), step.hidden, state);
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.ar.total_ms",
            engine::debug::elapsed_ms(ar_start, Clock::now()));
        return result;
    }

    void release_runtime_graphs() {
        if (global_runtime != nullptr) {
            global_runtime->release_runtime_graphs();
        }
        if (depth != nullptr) {
            depth->release_runtime_graphs();
        }
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    core::ExecutionContext & execution;
    MiniMaxMusic3PromptBuilder prompt_builder;
    MiniMaxMusic3GlobalLMWeights global_weights;
    std::unique_ptr<modules::QwenCausalDecodeRuntime> global_runtime;
    std::unique_ptr<MiniMaxMusic3DepthDecoderRuntime> depth;
    sampling::HfSamplerScratch semantic_scratch;
    std::vector<float> semantic_logits;
    std::vector<float> topk_window;
    std::vector<int32_t> compact_candidates;
    std::vector<double> semantic_weights;
    std::vector<float> feedback_batch;
};

MiniMaxMusic3ArRuntime::MiniMaxMusic3ArRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3ArRuntime::~MiniMaxMusic3ArRuntime() = default;

std::vector<float> MiniMaxMusic3ArRuntime::generate_frame_hiddens(
    const MiniMaxMusic3Request & request,
    int64_t target_frames,
    uint64_t & rng_offset_blocks) {
    auto result = impl_->generate_frame_hiddens_resumable(
        request, target_frames, {}, {}, rng_offset_blocks);
    if (!result.completed) {
        throw std::runtime_error("MiniMax Music 3 blocking AR run paused unexpectedly");
    }
    return std::move(result.frame_hiddens);
}

MiniMaxMusic3ArResult MiniMaxMusic3ArRuntime::generate_frame_hiddens_resumable(
    const MiniMaxMusic3Request & request,
    int64_t target_frames,
    const std::vector<int32_t> & replay_codes,
    const std::function<bool(std::size_t, std::size_t)> & should_pause,
    uint64_t & rng_offset_blocks) {
    return impl_->generate_frame_hiddens_resumable(
        request, target_frames, replay_codes, should_pause, rng_offset_blocks);
}

void MiniMaxMusic3ArRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
