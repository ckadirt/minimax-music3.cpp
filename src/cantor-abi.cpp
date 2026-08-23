#include "cantor_engine.h"

#include "minimax-checkpoint.h"
#include "minimax-request.h"

#include "engine/community_models/minimax_music3/ar_runtime.h"
#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/flow_sampler.h"
#include "engine/community_models/minimax_music3/seed.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mm3 = engine::models::minimax_music3;

struct cantor_ctx {
    std::string lm_path;
    std::string rvq_path;
    std::string condition_path;
    std::string dit_path;
    std::string vae_path;
    cantor_load_opts options{};
    std::shared_ptr<const mm3::MiniMaxMusic3Assets> assets;
    std::unique_ptr<engine::core::ExecutionContext> execution;
    std::array<std::array<std::uint8_t, 32>, 5> identities{};
    bool identities_ready = false;
    std::vector<float> audio;
    int audio_samples = 0;
    int audio_rate = 0;
    int resident_modules = 0;
};

namespace {

constexpr std::array<char, 8> codes_pause_magic{{'M', 'M', 'X', 'C', 'O', 'D', '0', '1'}};
constexpr std::array<char, 8> codes_done_magic{{'M', 'M', 'X', 'C', 'O', 'N', '0', '1'}};
constexpr std::array<char, 8> flow_pause_magic{{'M', 'M', 'X', 'F', 'L', 'W', '0', '1'}};
constexpr std::array<char, 8> flow_done_magic{{'M', 'M', 'X', 'L', 'A', 'T', '0', '1'}};
constexpr std::size_t graph_arena_bytes = 32U * 1024U * 1024U;
constexpr std::size_t weight_context_bytes = 32U * 1024U * 1024U;
constexpr int64_t crop_left_latent = 86;
constexpr int64_t crop_right_latent = 344 - crop_left_latent;

class model_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class backend_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

thread_local cantor_error last_error_code = CANTOR_OK;
thread_local std::string last_error;

void clear_error() {
    last_error_code = CANTOR_OK;
    last_error.clear();
}

void set_error(cantor_error code, const std::string & message) {
    last_error_code = code;
    last_error = message;
}

template <typename Function>
cantor_status guarded(Function && function) {
    try {
        return function();
    } catch (const std::bad_alloc &) {
        set_error(CANTOR_ERR_OOM, "[MiniMax ABI] allocation failed");
    } catch (const model_error & error) {
        set_error(CANTOR_ERR_MODEL, std::string("[MiniMax ABI] ") + error.what());
    } catch (const backend_error & error) {
        set_error(CANTOR_ERR_BACKEND, std::string("[MiniMax ABI] ") + error.what());
    } catch (const std::invalid_argument & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[MiniMax ABI] invalid argument: ") + error.what());
    } catch (const std::exception & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[MiniMax ABI] ") + error.what());
    } catch (...) {
        set_error(CANTOR_ERR_OTHER, "[MiniMax ABI] unknown exception");
    }
    return CANTOR_ERR;
}

bool has_magic(const std::uint8_t * input, std::size_t size, const std::array<char, 8> & magic) {
    return size >= magic.size() &&
        std::equal(magic.begin(), magic.end(), reinterpret_cast<const char *>(input));
}

bool is_cancelled(cantor_cancel_fn callback, void * userdata) {
    return callback != nullptr && callback(userdata) != 0;
}

void progress(cantor_progress_fn callback, void * userdata, cantor_stage stage,
              std::size_t completed, std::size_t total) {
    if (callback == nullptr) return;
    if (completed > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("progress counter exceeds Cantor ABI range");
    }
    callback(stage, static_cast<int>(completed), static_cast<int>(total), userdata);
}

bool allocate_blob(const std::vector<std::uint8_t> & source, std::uint8_t ** output, std::size_t * output_size) {
    auto * result = static_cast<std::uint8_t *>(std::malloc(source.empty() ? 1U : source.size()));
    if (result == nullptr) {
        set_error(CANTOR_ERR_OOM, "[MiniMax ABI] cannot allocate stage output");
        return false;
    }
    if (!source.empty()) std::memcpy(result, source.data(), source.size());
    *output = result;
    *output_size = source.size();
    return true;
}

void append_u32(std::vector<std::uint8_t> & output, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t> & output, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) output.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_f32(std::vector<std::uint8_t> & output, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "MiniMax state requires binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(output, bits);
}

std::uint32_t take_u32(const std::vector<std::uint8_t> & input, std::size_t & offset, const char * label) {
    if (offset > input.size() || input.size() - offset < 4) throw std::runtime_error(std::string("truncated ") + label);
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) result |= static_cast<std::uint32_t>(input[offset++]) << shift;
    return result;
}

std::uint64_t take_u64(const std::vector<std::uint8_t> & input, std::size_t & offset, const char * label) {
    if (offset > input.size() || input.size() - offset < 8) throw std::runtime_error(std::string("truncated ") + label);
    std::uint64_t result = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) result |= static_cast<std::uint64_t>(input[offset++]) << shift;
    return result;
}

float take_f32(const std::vector<std::uint8_t> & input, std::size_t & offset, const char * label) {
    const std::uint32_t bits = take_u32(input, offset, label);
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    if (!std::isfinite(result)) throw std::runtime_error(std::string(label) + " contains a non-finite float");
    return result;
}

const minimax::checkpoint::section & one_section(
    const minimax::checkpoint::decoded_blob & blob,
    minimax::checkpoint::section_kind kind) {
    const minimax::checkpoint::section * result = nullptr;
    for (const auto & section : blob.sections) {
        if (section.kind != kind) continue;
        if (result != nullptr) throw std::runtime_error("durable state contains a duplicate section");
        result = &section;
    }
    if (result == nullptr) throw std::runtime_error("durable state is missing a required section");
    return *result;
}

void require_exact_sections(
    const minimax::checkpoint::decoded_blob & blob,
    std::initializer_list<minimax::checkpoint::section_kind> expected) {
    if (blob.sections.size() != expected.size()) {
        throw std::runtime_error("durable state has an unexpected section count");
    }
    for (const auto kind : expected) (void)one_section(blob, kind);
}

std::string section_string(const minimax::checkpoint::section & section) {
    return std::string(section.bytes.begin(), section.bytes.end());
}

std::vector<std::uint8_t> encode_codes(const std::vector<std::int32_t> & codes) {
    std::vector<std::uint8_t> output;
    append_u64(output, codes.size());
    output.reserve(8U + codes.size() * 4U);
    for (const auto code : codes) append_u32(output, static_cast<std::uint32_t>(code));
    return output;
}

std::vector<std::int32_t> decode_codes(const std::vector<std::uint8_t> & input) {
    std::size_t offset = 0;
    const auto count = take_u64(input, offset, "AR code payload");
    if (count > (input.size() - offset) / 4U || count * 4U != input.size() - offset) {
        throw std::runtime_error("AR code payload has an invalid length");
    }
    std::vector<std::int32_t> result(static_cast<std::size_t>(count));
    for (auto & code : result) code = static_cast<std::int32_t>(take_u32(input, offset, "AR code payload"));
    if (result.size() % 8U != 0) throw std::runtime_error("AR code payload is not [T,8]");
    return result;
}

struct condition_chunk {
    std::int64_t frames = 0;
    std::vector<float> projected;
};

struct latent_chunk {
    std::int64_t frames = 0;
    std::vector<float> values;
};

std::vector<std::uint8_t> encode_conditions(const std::vector<condition_chunk> & chunks) {
    std::vector<std::uint8_t> output;
    append_u32(output, static_cast<std::uint32_t>(chunks.size()));
    for (const auto & chunk : chunks) {
        if (chunk.frames <= 0 || chunk.projected.size() != static_cast<std::size_t>(chunk.frames * 2048)) {
            throw std::runtime_error("projected condition shape mismatch");
        }
        append_u32(output, static_cast<std::uint32_t>(chunk.frames));
        append_u64(output, chunk.projected.size());
        for (const float value : chunk.projected) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint16_t bf16 = static_cast<std::uint16_t>(bits >> 16U);
            output.push_back(static_cast<std::uint8_t>(bf16));
            output.push_back(static_cast<std::uint8_t>(bf16 >> 8U));
        }
    }
    return output;
}

std::vector<condition_chunk> decode_conditions(const std::vector<std::uint8_t> & input) {
    std::size_t offset = 0;
    const auto count = take_u32(input, offset, "condition payload");
    if (count == 0 || count > 10000U) throw std::runtime_error("condition payload has an invalid window count");
    std::vector<condition_chunk> result;
    result.reserve(count);
    for (std::uint32_t index = 0; index != count; ++index) {
        condition_chunk chunk;
        chunk.frames = take_u32(input, offset, "condition payload");
        const auto values = take_u64(input, offset, "condition payload");
        if (chunk.frames <= 0 || values != static_cast<std::uint64_t>(chunk.frames * 2048) ||
            values > (input.size() - offset) / 2U) {
            throw std::runtime_error("condition payload has an invalid shape");
        }
        chunk.projected.resize(static_cast<std::size_t>(values));
        for (auto & value : chunk.projected) {
            const std::uint16_t bits = static_cast<std::uint16_t>(input[offset]) |
                static_cast<std::uint16_t>(input[offset + 1U] << 8U);
            offset += 2;
            const std::uint32_t f32 = static_cast<std::uint32_t>(bits) << 16U;
            std::memcpy(&value, &f32, sizeof(value));
        }
        result.push_back(std::move(chunk));
    }
    if (offset != input.size()) throw std::runtime_error("condition payload has trailing bytes");
    return result;
}

std::vector<std::uint8_t> encode_latents(const std::vector<latent_chunk> & chunks) {
    std::vector<std::uint8_t> output;
    append_u32(output, static_cast<std::uint32_t>(chunks.size()));
    for (const auto & chunk : chunks) {
        if (chunk.frames <= 0 || chunk.values.size() != static_cast<std::size_t>(chunk.frames * 128)) {
            throw std::runtime_error("latent chunk shape mismatch");
        }
        append_u32(output, static_cast<std::uint32_t>(chunk.frames));
        append_u64(output, chunk.values.size());
        for (const auto value : chunk.values) append_f32(output, value);
    }
    return output;
}

std::vector<latent_chunk> decode_latents(const std::vector<std::uint8_t> & input) {
    std::size_t offset = 0;
    const auto count = take_u32(input, offset, "latent payload");
    if (count > 10000U) throw std::runtime_error("latent payload has too many windows");
    std::vector<latent_chunk> result;
    result.reserve(count);
    for (std::uint32_t index = 0; index != count; ++index) {
        latent_chunk chunk;
        chunk.frames = take_u32(input, offset, "latent payload");
        const auto values = take_u64(input, offset, "latent payload");
        if (chunk.frames <= 0 || values != static_cast<std::uint64_t>(chunk.frames * 128) ||
            values > (input.size() - offset) / 4U) {
            throw std::runtime_error("latent payload has an invalid shape");
        }
        chunk.values.resize(static_cast<std::size_t>(values));
        for (auto & value : chunk.values) value = take_f32(input, offset, "latent payload");
        result.push_back(std::move(chunk));
    }
    if (offset != input.size()) throw std::runtime_error("latent payload has trailing bytes");
    return result;
}

std::vector<std::uint8_t> encode_float_vectors(const std::vector<std::vector<float>> & vectors) {
    std::vector<std::uint8_t> output;
    append_u32(output, static_cast<std::uint32_t>(vectors.size()));
    for (const auto & vector : vectors) {
        append_u64(output, vector.size());
        for (const auto value : vector) append_f32(output, value);
    }
    return output;
}

std::vector<std::vector<float>> decode_float_vectors(const std::vector<std::uint8_t> & input) {
    std::size_t offset = 0;
    const auto count = take_u32(input, offset, "float vector payload");
    if (count > 8U) throw std::runtime_error("float vector payload has too many vectors");
    std::vector<std::vector<float>> result(count);
    for (auto & vector : result) {
        const auto values = take_u64(input, offset, "float vector payload");
        if (values > (input.size() - offset) / 4U) throw std::runtime_error("float vector payload is truncated");
        vector.resize(static_cast<std::size_t>(values));
        for (auto & value : vector) value = take_f32(input, offset, "float vector payload");
    }
    if (offset != input.size()) throw std::runtime_error("float vector payload has trailing bytes");
    return result;
}

void ensure_components(cantor_ctx & context) {
    if (context.lm_path.empty() || context.rvq_path.empty() || context.condition_path.empty() ||
        context.dit_path.empty() || context.vae_path.empty()) {
        throw model_error("lm, rvq, condition, dit, and vae components are required");
    }
    if (!context.identities_ready) {
        const std::array<std::string, 5> paths{{context.lm_path, context.rvq_path, context.condition_path,
                                               context.dit_path, context.vae_path}};
        try {
            for (std::size_t index = 0; index != paths.size(); ++index) {
                context.identities[index] = minimax::checkpoint::sha256_file(paths[index]);
            }
        } catch (const std::exception & error) {
            throw model_error(error.what());
        }
        context.identities_ready = true;
    }
    if (context.assets == nullptr) {
        try {
            context.assets = mm3::load_minimax_music3_component_assets(
                context.lm_path, context.rvq_path, context.condition_path, context.dit_path, context.vae_path);
        } catch (const std::exception & error) {
            throw model_error(error.what());
        }
    }
    if (context.execution == nullptr) {
        try {
            const int threads = context.options.n_threads > 0 ? context.options.n_threads : 1;
            context.execution = std::make_unique<engine::core::ExecutionContext>(
                engine::core::BackendConfig{engine::core::BackendType::BestAvailable, 0, threads});
        } catch (const std::exception & error) {
            throw backend_error(error.what());
        }
    }
}

std::vector<std::uint8_t> identity_metadata(const cantor_ctx & context, std::uint32_t first, std::uint32_t second) {
    if (!context.identities_ready) throw std::runtime_error("component identities are unavailable");
    std::vector<std::uint8_t> output;
    append_u32(output, 1U);
    append_u32(output, first);
    append_u32(output, second);
    append_u32(output, 0U);
    for (const auto & digest : context.identities) output.insert(output.end(), digest.begin(), digest.end());
    return output;
}

std::pair<std::uint32_t, std::uint32_t> verify_identity_metadata(
    const cantor_ctx & context,
    const std::vector<std::uint8_t> & input) {
    if (input.size() != 16U + 5U * 32U) throw std::runtime_error("durable metadata has an invalid size");
    std::size_t offset = 0;
    if (take_u32(input, offset, "durable metadata") != 1U) throw std::runtime_error("unsupported durable metadata version");
    const auto first = take_u32(input, offset, "durable metadata");
    const auto second = take_u32(input, offset, "durable metadata");
    (void)take_u32(input, offset, "durable metadata");
    for (const auto & digest : context.identities) {
        if (!std::equal(digest.begin(), digest.end(), input.begin() + static_cast<std::ptrdiff_t>(offset))) {
            throw model_error("durable state component SHA-256 does not match loaded components");
        }
        offset += digest.size();
    }
    return {first, second};
}

mm3::MiniMaxMusic3Request native_request(const minimax::generation_request & request) {
    mm3::MiniMaxMusic3Request output;
    output.prompt = request.description;
    output.lyrics = request.lyrics;
    output.duration_sec = request.duration_seconds;
    output.num_inference_steps = static_cast<std::int64_t>(request.euler_steps);
    output.guidance_scale = request.flow_cfg_scale;
    output.ar_guidance_scale = request.ar_cfg_scale;
    output.top_k = static_cast<std::int64_t>(request.top_k);
    output.seed = request.seed;
    output.flow_seed = request.flow_seed;
    output.flow_seed_present = request.flow_seed_present;
    return output;
}

std::vector<std::int64_t> chunk_starts(std::int64_t frames) {
    if (frames <= 0) throw std::runtime_error("cannot window an empty AR result");
    if (frames <= 200) return {0};
    std::vector<std::int64_t> result;
    for (std::int64_t start = 0; start < frames - 100; start += 100) result.push_back(start);
    return result;
}

std::vector<float> slice_hiddens(const std::vector<float> & input, std::int64_t start, std::int64_t frames) {
    constexpr std::int64_t width = 8 * 4096;
    if (start < 0 || frames <= 0 || static_cast<std::uint64_t>((start + frames) * width) > input.size()) {
        throw std::runtime_error("AR hidden slice is out of range");
    }
    return std::vector<float>(
        input.begin() + static_cast<std::ptrdiff_t>(start * width),
        input.begin() + static_cast<std::ptrdiff_t>((start + frames) * width));
}

std::int64_t condition_output_frames(const condition_chunk & chunk) {
    const auto output_frames = static_cast<std::int64_t>(
        static_cast<double>(chunk.frames) * 44100.0 / 24000.0 * 960.0 / 512.0);
    if (output_frames <= 0 || chunk.projected.size() != static_cast<std::size_t>(chunk.frames * 2048)) {
        throw std::runtime_error("completed condition window has an invalid shape");
    }
    return output_frames;
}

std::vector<float> resize_condition(const condition_chunk & chunk) {
    const std::int64_t output_frames = condition_output_frames(chunk);
    std::vector<float> result(static_cast<std::size_t>(output_frames * 2048));
    for (std::int64_t frame = 0; frame < output_frames; ++frame) {
        const std::int64_t source = std::min<std::int64_t>(chunk.frames - 1, frame * chunk.frames / output_frames);
        std::copy_n(
            chunk.projected.begin() + static_cast<std::ptrdiff_t>(source * 2048),
            2048U,
            result.begin() + static_cast<std::ptrdiff_t>(frame * 2048));
    }
    return result;
}

void validate_condition_windows(
    const std::vector<condition_chunk> & conditions,
    std::uint32_t generated_frames) {
    const auto starts = chunk_starts(generated_frames);
    if (conditions.size() != starts.size()) {
        throw std::runtime_error("completed CODES metadata does not match condition windows");
    }
    for (std::size_t index = 0; index != conditions.size(); ++index) {
        const auto expected_frames = std::min<std::int64_t>(200, generated_frames - starts[index]);
        if (conditions[index].frames != expected_frames) {
            throw std::runtime_error("completed condition window shape does not match CODES metadata");
        }
    }
}

std::vector<float> crop_audio(const engine::runtime::AudioBuffer & audio, std::int64_t left, std::int64_t right) {
    if (audio.channels != 2 || audio.samples.size() % 2U != 0) throw std::runtime_error("vocoder returned invalid stereo audio");
    const auto frames = static_cast<std::int64_t>(audio.samples.size() / 2U);
    const auto begin = std::min(std::max<std::int64_t>(0, left), frames);
    const auto end = std::max(begin, frames - std::max<std::int64_t>(0, right));
    return std::vector<float>(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(begin * 2),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end * 2));
}

cantor_status pause_with_blob(
    const std::vector<std::uint8_t> & blob,
    const char * message,
    std::uint8_t ** output,
    std::size_t * output_size) {
    if (!allocate_blob(blob, output, output_size)) return CANTOR_ERR;
    set_error(CANTOR_ERR_CANCEL, message);
    return CANTOR_PAUSED;
}

cantor_status run_codes(cantor_ctx & context, const std::uint8_t * input, std::size_t input_size,
                        std::uint8_t ** output, std::size_t * output_size,
                        cantor_progress_fn on_progress, cantor_cancel_fn should_cancel, void * userdata) {
    minimax::generation_request request;
    std::vector<std::int32_t> replay_codes;
    bool resuming = has_magic(input, input_size, codes_pause_magic);
    minimax::checkpoint::decoded_blob resume_blob;
    if (resuming) {
        resume_blob = minimax::checkpoint::decode(input, input_size, codes_pause_magic);
        if (resume_blob.stage != CANTOR_STAGE_CODES) throw std::runtime_error("CODES checkpoint has the wrong stage");
        require_exact_sections(resume_blob, {
            minimax::checkpoint::section_kind::request_json,
            minimax::checkpoint::section_kind::ar_codes_i32,
            minimax::checkpoint::section_kind::metadata,
        });
        request = minimax::request_io::parse(section_string(one_section(
            resume_blob, minimax::checkpoint::section_kind::request_json)));
        replay_codes = decode_codes(one_section(
            resume_blob, minimax::checkpoint::section_kind::ar_codes_i32).bytes);
    } else {
        request = minimax::request_io::parse(std::string(reinterpret_cast<const char *>(input), input_size));
    }
    ensure_components(context);

    const auto native = native_request(request);
    const auto target_frames = std::min<std::int64_t>(
        context.assets->config.max_audio_frames,
        static_cast<std::int64_t>(request.duration_seconds * 25.0));
    if (resuming) {
        const auto cursor = verify_identity_metadata(context, one_section(
            resume_blob, minimax::checkpoint::section_kind::metadata).bytes);
        if (cursor.first != static_cast<std::uint32_t>(target_frames) ||
            cursor.second != replay_codes.size() / 8U || cursor.second > cursor.first + 1U) {
            throw std::runtime_error("CODES checkpoint cursor does not match its request or code payload");
        }
    }
    std::uint64_t unused_rng_offset = 0;
    context.resident_modules = 2;
    mm3::MiniMaxMusic3ArRuntime ar(
        context.assets, *context.execution, graph_arena_bytes, weight_context_bytes,
        engine::assets::TensorStorageType::Native);
    auto ar_result = ar.generate_frame_hiddens_resumable(
        native,
        target_frames,
        replay_codes,
        [&](std::size_t completed, std::size_t total) {
            progress(on_progress, userdata, CANTOR_STAGE_CODES, completed, total);
            return is_cancelled(should_cancel, userdata);
        },
        unused_rng_offset);
    ar.release_runtime_graphs();
    context.resident_modules = 0;
    if (!ar_result.completed) {
        const std::string canonical = minimax::request_io::serialize(request);
        const auto blob = minimax::checkpoint::encode(codes_pause_magic, CANTOR_STAGE_CODES, {
            {minimax::checkpoint::section_kind::request_json, {canonical.begin(), canonical.end()}},
            {minimax::checkpoint::section_kind::ar_codes_i32, encode_codes(ar_result.codes)},
            {minimax::checkpoint::section_kind::metadata,
             identity_metadata(context, static_cast<std::uint32_t>(target_frames),
                               static_cast<std::uint32_t>(ar_result.codes.size() / 8U))},
        });
        return pause_with_blob(blob, "[MiniMax ABI] CODES paused at an AR-frame boundary", output, output_size);
    }

    const std::int64_t generated_frames = static_cast<std::int64_t>(ar_result.frame_hiddens.size()) / (8 * 4096);
    const auto starts = chunk_starts(generated_frames);
    context.resident_modules = 1;
    mm3::MiniMaxMusic3ConditionEncoderRuntime condition(
        context.assets, *context.execution, graph_arena_bytes, weight_context_bytes,
        engine::assets::TensorStorageType::Native);
    std::vector<condition_chunk> conditions;
    conditions.reserve(starts.size());
    for (std::size_t index = 0; index != starts.size(); ++index) {
        if (is_cancelled(should_cancel, userdata)) {
            condition.release_runtime_graphs();
            context.resident_modules = 0;
            const std::string canonical = minimax::request_io::serialize(request);
            const auto blob = minimax::checkpoint::encode(codes_pause_magic, CANTOR_STAGE_CODES, {
                {minimax::checkpoint::section_kind::request_json, {canonical.begin(), canonical.end()}},
                {minimax::checkpoint::section_kind::ar_codes_i32, encode_codes(ar_result.codes)},
                {minimax::checkpoint::section_kind::metadata,
                 identity_metadata(context, static_cast<std::uint32_t>(target_frames),
                                   static_cast<std::uint32_t>(ar_result.codes.size() / 8U))},
            });
            return pause_with_blob(blob, "[MiniMax ABI] CODES paused before condition projection", output, output_size);
        }
        const auto frames = std::min<std::int64_t>(200, generated_frames - starts[index]);
        auto projected = condition.project(slice_hiddens(ar_result.frame_hiddens, starts[index], frames), frames);
        engine::core::round_f32_to_bf16_in_place(projected);
        conditions.push_back({frames, std::move(projected)});
    }
    condition.release_runtime_graphs();
    context.resident_modules = 0;
    const std::string canonical = minimax::request_io::serialize(request);
    const auto blob = minimax::checkpoint::encode(codes_done_magic, CANTOR_STAGE_CODES, {
        {minimax::checkpoint::section_kind::request_json, {canonical.begin(), canonical.end()}},
        {minimax::checkpoint::section_kind::conditions_bf16, encode_conditions(conditions)},
        {minimax::checkpoint::section_kind::metadata,
         identity_metadata(context, static_cast<std::uint32_t>(generated_frames),
                           static_cast<std::uint32_t>(conditions.size()))},
    });
    if (!allocate_blob(blob, output, output_size)) return CANTOR_ERR;
    progress(on_progress, userdata, CANTOR_STAGE_CODES,
             static_cast<std::size_t>(target_frames + 1), static_cast<std::size_t>(target_frames + 1));
    return CANTOR_DONE;
}

struct flow_resume_state {
    std::vector<std::uint8_t> codes_boundary;
    std::vector<latent_chunk> completed;
    std::vector<float> active;
    std::vector<float> carry_condition;
    std::vector<float> carry_latent;
    std::uint32_t chunk_index = 0;
    std::uint32_t completed_steps = 0;
};

flow_resume_state decode_flow_pause(const minimax::checkpoint::decoded_blob & blob) {
    require_exact_sections(blob, {
        minimax::checkpoint::section_kind::source_boundary,
        minimax::checkpoint::section_kind::completed_latents_f32,
        minimax::checkpoint::section_kind::euler_state_f32,
        minimax::checkpoint::section_kind::carry_f32,
        minimax::checkpoint::section_kind::metadata,
    });
    flow_resume_state result;
    result.codes_boundary = one_section(blob, minimax::checkpoint::section_kind::source_boundary).bytes;
    result.completed = decode_latents(one_section(blob, minimax::checkpoint::section_kind::completed_latents_f32).bytes);
    const auto active = decode_float_vectors(one_section(blob, minimax::checkpoint::section_kind::euler_state_f32).bytes);
    if (active.size() != 1U) throw std::runtime_error("flow checkpoint active Euler state is invalid");
    result.active = active.front();
    const auto carry = decode_float_vectors(one_section(blob, minimax::checkpoint::section_kind::carry_f32).bytes);
    if (carry.size() != 2U) throw std::runtime_error("flow checkpoint carry state is invalid");
    result.carry_condition = carry[0];
    result.carry_latent = carry[1];
    return result;
}

cantor_status run_flow(cantor_ctx & context, const std::uint8_t * input, std::size_t input_size,
                       std::uint8_t ** output, std::size_t * output_size,
                       cantor_progress_fn on_progress, cantor_cancel_fn should_cancel, void * userdata) {
    flow_resume_state state;
    minimax::checkpoint::decoded_blob paused;
    const bool resuming = has_magic(input, input_size, flow_pause_magic);
    if (resuming) {
        paused = minimax::checkpoint::decode(input, input_size, flow_pause_magic);
        if (paused.stage != CANTOR_STAGE_DIFFUSE) throw std::runtime_error("DIFFUSE checkpoint has the wrong stage");
        state = decode_flow_pause(paused);
    } else {
        state.codes_boundary.assign(input, input + input_size);
    }
    const auto codes_blob = minimax::checkpoint::decode(state.codes_boundary, codes_done_magic);
    if (codes_blob.stage != CANTOR_STAGE_CODES) throw std::runtime_error("DIFFUSE input is not a completed CODES boundary");
    require_exact_sections(codes_blob, {
        minimax::checkpoint::section_kind::request_json,
        minimax::checkpoint::section_kind::conditions_bf16,
        minimax::checkpoint::section_kind::metadata,
    });
    const auto request = minimax::request_io::parse(section_string(one_section(
        codes_blob, minimax::checkpoint::section_kind::request_json)));
    const auto conditions = decode_conditions(one_section(
        codes_blob, minimax::checkpoint::section_kind::conditions_bf16).bytes);

    ensure_components(context);
    const auto codes_cursor = verify_identity_metadata(
        context, one_section(codes_blob, minimax::checkpoint::section_kind::metadata).bytes);
    if (codes_cursor.first == 0U || codes_cursor.second != conditions.size()) {
        throw std::runtime_error("completed CODES metadata does not match its condition payload");
    }
    validate_condition_windows(conditions, codes_cursor.first);
    if (resuming) {
        const auto indexes = verify_identity_metadata(context, one_section(
            paused, minimax::checkpoint::section_kind::metadata).bytes);
        state.chunk_index = indexes.first;
        state.completed_steps = indexes.second;
    }
    if (state.chunk_index > conditions.size() || state.completed.size() != state.chunk_index) {
        throw std::runtime_error("flow checkpoint window cursor is inconsistent");
    }
    if (resuming && (state.chunk_index >= conditions.size() ||
                     state.completed_steps > request.euler_steps || state.active.empty())) {
        throw std::runtime_error("flow checkpoint Euler cursor is inconsistent");
    }

    auto native = native_request(request);
    const auto policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        context.execution->backend_type(), context.execution->config().device,
        "minimax_music3.cantor.flow", "MiniMax Music 3 Cantor flow",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);
    context.resident_modules = 1;
    mm3::MiniMaxMusic3FlowSamplerRuntime flow(
        context.assets, *context.execution, graph_arena_bytes, weight_context_bytes,
        engine::assets::TensorStorageType::Native);
    for (std::size_t index = state.chunk_index; index != conditions.size(); ++index) {
        const auto resized = resize_condition(conditions[index]);
        const auto frames = static_cast<std::int64_t>(resized.size() / 2048U);
        auto chunk_request = native;
        const auto base_seed = request.flow_seed_present ? request.flow_seed : request.seed;
        chunk_request.seed = mm3::derive_dit_chunk_seed(base_seed, index);
        const std::int64_t starting_step = index == state.chunk_index ? state.completed_steps : 0;
        const std::vector<float> active = index == state.chunk_index ? state.active : std::vector<float>{};
        auto result = flow.denoise_chunk_resumable(
            resized,
            frames,
            state.carry_latent,
            state.carry_condition,
            chunk_request,
            0,
            policy,
            active,
            starting_step,
            [&](std::int64_t completed, std::int64_t total) {
                progress(on_progress, userdata, CANTOR_STAGE_DIFFUSE,
                         index * static_cast<std::size_t>(total) + static_cast<std::size_t>(completed),
                         conditions.size() * static_cast<std::size_t>(total));
                return is_cancelled(should_cancel, userdata);
            });
        if (!result.completed) {
            flow.release_runtime_graphs();
            context.resident_modules = 0;
            const auto blob = minimax::checkpoint::encode(flow_pause_magic, CANTOR_STAGE_DIFFUSE, {
                {minimax::checkpoint::section_kind::source_boundary, state.codes_boundary},
                {minimax::checkpoint::section_kind::completed_latents_f32, encode_latents(state.completed)},
                {minimax::checkpoint::section_kind::euler_state_f32, encode_float_vectors({result.latents})},
                {minimax::checkpoint::section_kind::carry_f32,
                 encode_float_vectors({state.carry_condition, state.carry_latent})},
                {minimax::checkpoint::section_kind::metadata,
                 identity_metadata(context, static_cast<std::uint32_t>(index),
                                   static_cast<std::uint32_t>(result.completed_steps))},
            });
            return pause_with_blob(blob, "[MiniMax ABI] DIFFUSE paused at an Euler boundary", output, output_size);
        }
        state.completed.push_back({frames, std::move(result.latents)});
        state.carry_condition = std::move(result.carry_condition);
        state.carry_latent = std::move(result.carry_latent);
        state.active.clear();
        state.completed_steps = 0;
    }
    flow.release_runtime_graphs();
    context.resident_modules = 0;
    const auto blob = minimax::checkpoint::encode(flow_done_magic, CANTOR_STAGE_DIFFUSE, {
        {minimax::checkpoint::section_kind::source_boundary, state.codes_boundary},
        {minimax::checkpoint::section_kind::completed_latents_f32, encode_latents(state.completed)},
        {minimax::checkpoint::section_kind::metadata,
         identity_metadata(context, static_cast<std::uint32_t>(state.completed.size()), 0U)},
    });
    if (!allocate_blob(blob, output, output_size)) return CANTOR_ERR;
    progress(on_progress, userdata, CANTOR_STAGE_DIFFUSE,
             conditions.size() * request.euler_steps, conditions.size() * request.euler_steps);
    return CANTOR_DONE;
}

cantor_status run_decode(cantor_ctx & context, const std::uint8_t * input, std::size_t input_size,
                         cantor_progress_fn on_progress, cantor_cancel_fn should_cancel, void * userdata) {
    const auto flow_blob = minimax::checkpoint::decode(input, input_size, flow_done_magic);
    if (flow_blob.stage != CANTOR_STAGE_DIFFUSE) throw std::runtime_error("DECODE input is not a completed DIFFUSE boundary");
    require_exact_sections(flow_blob, {
        minimax::checkpoint::section_kind::source_boundary,
        minimax::checkpoint::section_kind::completed_latents_f32,
        minimax::checkpoint::section_kind::metadata,
    });
    const auto codes_bytes = one_section(flow_blob, minimax::checkpoint::section_kind::source_boundary).bytes;
    const auto codes_blob = minimax::checkpoint::decode(codes_bytes, codes_done_magic);
    if (codes_blob.stage != CANTOR_STAGE_CODES) throw std::runtime_error("DECODE source is not a completed CODES boundary");
    require_exact_sections(codes_blob, {
        minimax::checkpoint::section_kind::request_json,
        minimax::checkpoint::section_kind::conditions_bf16,
        minimax::checkpoint::section_kind::metadata,
    });
    const auto request = minimax::request_io::parse(section_string(one_section(
        codes_blob, minimax::checkpoint::section_kind::request_json)));
    auto latents = decode_latents(one_section(
        flow_blob, minimax::checkpoint::section_kind::completed_latents_f32).bytes);
    if (latents.empty()) throw std::runtime_error("DECODE boundary contains no latent windows");
    ensure_components(context);
    const auto flow_cursor = verify_identity_metadata(
        context, one_section(flow_blob, minimax::checkpoint::section_kind::metadata).bytes);
    if (flow_cursor.first != latents.size() || flow_cursor.second != 0U) {
        throw std::runtime_error("completed DIFFUSE metadata does not match its latent payload");
    }
    const auto conditions = decode_conditions(one_section(
        codes_blob, minimax::checkpoint::section_kind::conditions_bf16).bytes);
    const auto codes_cursor = verify_identity_metadata(
        context, one_section(codes_blob, minimax::checkpoint::section_kind::metadata).bytes);
    if (codes_cursor.first == 0U || codes_cursor.second != conditions.size() ||
        conditions.size() != latents.size()) {
        throw std::runtime_error("DECODE boundaries have inconsistent window counts");
    }
    validate_condition_windows(conditions, codes_cursor.first);
    for (std::size_t index = 0; index != latents.size(); ++index) {
        if (latents[index].frames != condition_output_frames(conditions[index])) {
            throw std::runtime_error("DECODE latent shape does not match its condition window");
        }
    }

    context.audio.clear();
    context.audio_samples = 0;
    context.audio_rate = 0;
    context.resident_modules = 1;
    mm3::MiniMaxMusic3VocoderRuntime vocoder(
        context.assets, *context.execution, graph_arena_bytes, weight_context_bytes,
        engine::assets::TensorStorageType::Native);
    std::vector<float> interleaved;
    for (std::size_t index = 0; index != latents.size(); ++index) {
        progress(on_progress, userdata, CANTOR_STAGE_DECODE, index, latents.size());
        if (is_cancelled(should_cancel, userdata)) {
            vocoder.release_runtime_graphs();
            context.resident_modules = 0;
            set_error(CANTOR_ERR_CANCEL, "[MiniMax ABI] DECODE paused; retry the durable DIFFUSE boundary");
            return CANTOR_PAUSED;
        }
        auto audio = vocoder.decode(latents[index].values, latents[index].frames);
        const auto left = index == 0 ? 0 : crop_left_latent * 512;
        const auto right = index + 1U == latents.size() ? 0 : crop_right_latent * 512;
        auto cropped = crop_audio(audio, left, right);
        interleaved.insert(interleaved.end(), cropped.begin(), cropped.end());
    }
    vocoder.release_runtime_graphs();
    context.resident_modules = 0;

    std::vector<float> left(interleaved.size() / 2U);
    std::vector<float> right(interleaved.size() / 2U);
    for (std::size_t frame = 0; frame != left.size(); ++frame) {
        left[frame] = std::clamp(interleaved[2U * frame], -1.0F, 1.0F);
        right[frame] = std::clamp(interleaved[2U * frame + 1U], -1.0F, 1.0F);
    }
    if (request.output_sample_rate != 44100) {
        const auto settings = engine::audio::torchaudio_sinc_hann_float32_options();
        left = engine::audio::resample_mono_torchaudio_sinc_hann(
            left, 44100, request.output_sample_rate, settings);
        right = engine::audio::resample_mono_torchaudio_sinc_hann(
            right, 44100, request.output_sample_rate, settings);
    }
    context.audio_samples = static_cast<int>(std::min(left.size(), right.size()));
    context.audio_rate = request.output_sample_rate;
    context.audio.resize(static_cast<std::size_t>(context.audio_samples) * 2U);
    std::copy_n(left.begin(), static_cast<std::size_t>(context.audio_samples), context.audio.begin());
    std::copy_n(right.begin(), static_cast<std::size_t>(context.audio_samples),
                context.audio.begin() + static_cast<std::ptrdiff_t>(context.audio_samples));
    progress(on_progress, userdata, CANTOR_STAGE_DECODE, latents.size(), latents.size());
    return CANTOR_DONE;
}

}  // namespace

extern "C" std::uint32_t cantor_engine_abi_version(void) { return CANTOR_ENGINE_ABI; }
extern "C" const char * cantor_engine_model(void) { return "minimax-music3"; }
extern "C" const char * cantor_engine_version(void) { return MINIMAX_VERSION; }
extern "C" std::uint32_t cantor_engine_stages(void) {
    return (1U << CANTOR_STAGE_CODES) | (1U << CANTOR_STAGE_DIFFUSE) | (1U << CANTOR_STAGE_DECODE);
}
extern "C" cantor_error cantor_engine_last_error_code(void) { return last_error_code; }
extern "C" const char * cantor_engine_last_error(void) { return last_error.c_str(); }

extern "C" cantor_ctx * cantor_engine_load(
    const cantor_component * components,
    std::size_t count,
    const cantor_load_opts * options) {
    clear_error();
    try {
        if (count != 0 && components == nullptr) throw std::invalid_argument("component array is null");
        auto context = std::make_unique<cantor_ctx>();
        if (options != nullptr) context->options = *options;
        if (context->options.n_threads < 0 || context->options.vae_chunk < 0 || context->options.vae_overlap < 0) {
            throw std::invalid_argument("load options cannot be negative");
        }
        for (std::size_t index = 0; index != count; ++index) {
            if (components[index].role == nullptr || components[index].path == nullptr ||
                components[index].path[0] == '\0') {
                throw std::invalid_argument("component role/path is empty");
            }
            const std::string role = components[index].role;
            std::string * destination = nullptr;
            if (role == "lm") destination = &context->lm_path;
            else if (role == "rvq") destination = &context->rvq_path;
            else if (role == "condition") destination = &context->condition_path;
            else if (role == "dit") destination = &context->dit_path;
            else if (role == "vae") destination = &context->vae_path;
            else throw std::invalid_argument("unknown component role '" + role + "'");
            if (!destination->empty()) throw std::invalid_argument("duplicate component role '" + role + "'");
            *destination = components[index].path;
        }
        return context.release();
    } catch (const std::bad_alloc &) {
        set_error(CANTOR_ERR_OOM, "[MiniMax ABI] allocation failed");
    } catch (const std::exception & error) {
        set_error(CANTOR_ERR_OTHER, std::string("[MiniMax ABI] ") + error.what());
    }
    return nullptr;
}

extern "C" void cantor_engine_free(cantor_ctx * context) { delete context; }
extern "C" void cantor_engine_free_blob(std::uint8_t * blob) { std::free(blob); }

extern "C" cantor_status cantor_engine_run_stage(
    cantor_ctx * context,
    cantor_stage stage,
    const std::uint8_t * input,
    std::size_t input_size,
    std::uint8_t ** output,
    std::size_t * output_size,
    cantor_progress_fn on_progress,
    cantor_cancel_fn should_cancel,
    void * userdata) {
    clear_error();
    if (output != nullptr) *output = nullptr;
    if (output_size != nullptr) *output_size = 0;
    if (context == nullptr || output == nullptr || output_size == nullptr ||
        input == nullptr || input_size == 0) {
        set_error(CANTOR_ERR_OTHER, "[MiniMax ABI] stage arguments are invalid");
        return CANTOR_ERR;
    }
    context->audio.clear();
    context->audio_samples = 0;
    context->audio_rate = 0;
    switch (stage) {
        case CANTOR_STAGE_CODES:
            return guarded([&] { return run_codes(*context, input, input_size, output, output_size,
                                                   on_progress, should_cancel, userdata); });
        case CANTOR_STAGE_DIFFUSE:
            return guarded([&] { return run_flow(*context, input, input_size, output, output_size,
                                                  on_progress, should_cancel, userdata); });
        case CANTOR_STAGE_DECODE:
            return guarded([&] { return run_decode(*context, input, input_size,
                                                    on_progress, should_cancel, userdata); });
        default:
            set_error(CANTOR_ERR_OTHER, "[MiniMax ABI] unsupported stage");
            return CANTOR_ERR;
    }
}

extern "C" const float * cantor_engine_audio(cantor_ctx * context, int * samples, int * sample_rate) {
    if (samples != nullptr) *samples = context == nullptr ? 0 : context->audio_samples;
    if (sample_rate != nullptr) *sample_rate = context == nullptr ? 0 : context->audio_rate;
    return context == nullptr || context->audio.empty() ? nullptr : context->audio.data();
}

extern "C" std::uint64_t cantor_engine_resident_bytes(cantor_ctx * context) {
    if (context == nullptr || context->execution == nullptr) return 0;
    const auto memory = context->execution->memory_snapshot();
    return memory.available && memory.used_bytes > 0 ? static_cast<std::uint64_t>(memory.used_bytes) : 0;
}

extern "C" int cantor_engine_resident_modules(cantor_ctx * context) {
    return context == nullptr ? 0 : context->resident_modules;
}
