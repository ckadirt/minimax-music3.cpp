#include "cantor_engine.h"
#include "minimax-checkpoint.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct blob_deleter {
    void operator()(std::uint8_t * value) const noexcept {
        cantor_engine_free_blob(value);
    }
};

struct blob {
    std::unique_ptr<std::uint8_t, blob_deleter> data;
    std::size_t size = 0;
};

struct context_deleter {
    void operator()(cantor_ctx * value) const noexcept {
        cantor_engine_free(value);
    }
};

using context_ptr = std::unique_ptr<cantor_ctx, context_deleter>;

struct one_shot_cancel {
    std::size_t calls = 0;
    std::size_t pause_on_call = 0;
    bool fired = false;
};

int cancel_once(void * userdata) {
    auto & state = *static_cast<one_shot_cancel *>(userdata);
    const bool cancel = !state.fired && state.calls++ == state.pause_on_call;
    state.fired = state.fired || cancel;
    return cancel ? 1 : 0;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string engine_error(const std::string & action) {
    const char * detail = cantor_engine_last_error();
    return action + " failed: " + (detail == nullptr ? "unknown Cantor error" : detail);
}

blob run_stage(
    cantor_ctx * context,
    cantor_stage stage,
    const std::uint8_t * input,
    std::size_t input_size,
    cantor_status expected,
    one_shot_cancel * cancel = nullptr) {
    std::uint8_t * output = nullptr;
    std::size_t output_size = 0;
    const auto status = cantor_engine_run_stage(
        context,
        stage,
        input,
        input_size,
        &output,
        &output_size,
        nullptr,
        cancel == nullptr ? nullptr : cancel_once,
        cancel);
    blob result{std::unique_ptr<std::uint8_t, blob_deleter>(output), output_size};
    if (status != expected) {
        throw std::runtime_error(engine_error("Cantor stage " + std::to_string(stage)));
    }
    if (cantor_engine_resident_modules(context) != 0) {
        throw std::runtime_error("Cantor stage left a model module resident");
    }
    if ((stage == CANTOR_STAGE_DECODE) != (result.data == nullptr)) {
        throw std::runtime_error("Cantor stage returned an invalid blob ownership shape");
    }
    return result;
}

bool equal_blob(const blob & left, const blob & right) {
    return left.size == right.size &&
        std::equal(left.data.get(), left.data.get() + left.size, right.data.get());
}

std::string completed_flow_difference(const blob & left, const blob & right) {
    constexpr std::array<char, 8> magic{{'M', 'M', 'X', 'L', 'A', 'T', '0', '1'}};
    const auto left_blob = minimax::checkpoint::decode(left.data.get(), left.size, magic);
    const auto right_blob = minimax::checkpoint::decode(right.data.get(), right.size, magic);
    if (left_blob.sections.size() != right_blob.sections.size()) {
        return "section count differs";
    }
    for (std::size_t section_index = 0; section_index != left_blob.sections.size(); ++section_index) {
        const auto & lhs = left_blob.sections[section_index];
        const auto & rhs = right_blob.sections[section_index];
        if (lhs.kind != rhs.kind || lhs.bytes.size() != rhs.bytes.size()) {
            return "section kind or size differs at index " + std::to_string(section_index);
        }
        const auto mismatch = std::mismatch(lhs.bytes.begin(), lhs.bytes.end(), rhs.bytes.begin());
        if (mismatch.first == lhs.bytes.end()) continue;
        const auto offset = static_cast<std::size_t>(mismatch.first - lhs.bytes.begin());
        std::string detail = "section " + std::to_string(static_cast<std::uint32_t>(lhs.kind)) +
            " first differs at byte " + std::to_string(offset);
        if (lhs.kind == minimax::checkpoint::section_kind::completed_latents_f32 &&
            offset >= 16 && offset + 4 <= lhs.bytes.size()) {
            const auto float_offset = 16 + ((offset - 16) / 4) * 4;
            float lhs_value = 0.0F;
            float rhs_value = 0.0F;
            std::memcpy(&lhs_value, lhs.bytes.data() + float_offset, sizeof(float));
            std::memcpy(&rhs_value, rhs.bytes.data() + float_offset, sizeof(float));
            detail += " (F32 " + std::to_string(lhs_value) + " versus " +
                std::to_string(rhs_value) + ")";
        }
        return detail;
    }
    return "payloads match but envelope bytes differ";
}

std::uint64_t fnv1a64(const std::uint8_t * bytes, std::size_t size) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t index = 0; index != size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::uint64_t audio_fingerprint(const std::vector<float> & audio) {
    return fnv1a64(
        reinterpret_cast<const std::uint8_t *>(audio.data()),
        audio.size() * sizeof(float));
}

void usage(const char * program) {
    std::cerr << "Usage: " << program
              << " --model DIR --request REQUEST.json [--threads N]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        std::filesystem::path model;
        std::filesystem::path request_path;
        int threads = 1;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) {
                usage(argv[0]);
                return 2;
            }
            const std::string option = argv[index];
            if (option == "--model") model = argv[index + 1];
            else if (option == "--request") request_path = argv[index + 1];
            else if (option == "--threads") threads = std::stoi(argv[index + 1]);
            else throw std::invalid_argument("unknown option: " + option);
        }
        if (model.empty() || request_path.empty() || threads <= 0) {
            usage(argv[0]);
            return 2;
        }
        model = std::filesystem::absolute(model);
        const auto request = read_bytes(request_path);
        if (request.empty()) throw std::runtime_error("request is empty");

        const std::vector<std::string> paths{
            (model / "lm-Q4_K_M.gguf").string(),
            (model / "rvq-Q4_K_M.gguf").string(),
            (model / "condition-F32.gguf").string(),
            (model / "dit-Q4_K_M.gguf").string(),
            (model / "vae-F16.gguf").string(),
        };
        const cantor_component components[] = {
            {"lm", paths[0].c_str()},
            {"rvq", paths[1].c_str()},
            {"condition", paths[2].c_str()},
            {"dit", paths[3].c_str()},
            {"vae", paths[4].c_str()},
        };
        cantor_load_opts options{};
        options.n_threads = threads;
        context_ptr context(cantor_engine_load(components, 5, &options));
        if (!context) throw std::runtime_error(engine_error("Cantor load"));

        const auto direct_codes = run_stage(
            context.get(), CANTOR_STAGE_CODES, request.data(), request.size(), CANTOR_DONE);
        one_shot_cancel codes_cancel{0, 1, false};
        const auto paused_codes = run_stage(
            context.get(), CANTOR_STAGE_CODES, request.data(), request.size(),
            CANTOR_PAUSED, &codes_cancel);
        if (!codes_cancel.fired || paused_codes.size == 0) {
            throw std::runtime_error("CODES did not produce a durable paused boundary");
        }
        const auto resumed_codes = run_stage(
            context.get(), CANTOR_STAGE_CODES,
            paused_codes.data.get(), paused_codes.size, CANTOR_DONE);
        if (!equal_blob(direct_codes, resumed_codes)) {
            throw std::runtime_error("resumed CODES boundary differs from uninterrupted execution");
        }

        const auto direct_latents = run_stage(
            context.get(), CANTOR_STAGE_DIFFUSE,
            direct_codes.data.get(), direct_codes.size, CANTOR_DONE);
        const auto repeated_latents = run_stage(
            context.get(), CANTOR_STAGE_DIFFUSE,
            direct_codes.data.get(), direct_codes.size, CANTOR_DONE);
        if (!equal_blob(direct_latents, repeated_latents)) {
            throw std::runtime_error(
                "repeated uninterrupted DIFFUSE boundary differs: " +
                completed_flow_difference(direct_latents, repeated_latents));
        }
        one_shot_cancel diffuse_cancel{0, 1, false};
        const auto paused_latents = run_stage(
            context.get(), CANTOR_STAGE_DIFFUSE,
            direct_codes.data.get(), direct_codes.size, CANTOR_PAUSED, &diffuse_cancel);
        if (!diffuse_cancel.fired || paused_latents.size == 0) {
            throw std::runtime_error("DIFFUSE did not produce a durable paused boundary");
        }
        const auto resumed_latents = run_stage(
            context.get(), CANTOR_STAGE_DIFFUSE,
            paused_latents.data.get(), paused_latents.size, CANTOR_DONE);
        if (!equal_blob(direct_latents, resumed_latents)) {
            throw std::runtime_error(
                "resumed DIFFUSE boundary differs from uninterrupted execution: " +
                completed_flow_difference(direct_latents, resumed_latents));
        }

        (void)run_stage(
            context.get(), CANTOR_STAGE_DECODE,
            direct_latents.data.get(), direct_latents.size, CANTOR_DONE);
        int direct_samples = 0;
        int direct_rate = 0;
        const float * direct_audio_ptr = cantor_engine_audio(
            context.get(), &direct_samples, &direct_rate);
        if (direct_audio_ptr == nullptr || direct_samples <= 0 || direct_rate <= 0) {
            throw std::runtime_error("uninterrupted DECODE returned no audio");
        }
        const std::vector<float> direct_audio(
            direct_audio_ptr, direct_audio_ptr + 2LL * direct_samples);

        one_shot_cancel decode_cancel{0, 0, false};
        (void)run_stage(
            context.get(), CANTOR_STAGE_DECODE,
            direct_latents.data.get(), direct_latents.size,
            CANTOR_PAUSED, &decode_cancel);
        int cancelled_samples = -1;
        int cancelled_rate = -1;
        if (!decode_cancel.fired || cantor_engine_audio(
                context.get(), &cancelled_samples, &cancelled_rate) != nullptr ||
            cancelled_samples != 0 || cancelled_rate != 0) {
            throw std::runtime_error("cancelled DECODE retained partial audio");
        }
        (void)run_stage(
            context.get(), CANTOR_STAGE_DECODE,
            direct_latents.data.get(), direct_latents.size, CANTOR_DONE);
        int retry_samples = 0;
        int retry_rate = 0;
        const float * retry_audio_ptr = cantor_engine_audio(
            context.get(), &retry_samples, &retry_rate);
        if (retry_audio_ptr == nullptr || retry_samples != direct_samples || retry_rate != direct_rate ||
            !std::equal(direct_audio.begin(), direct_audio.end(), retry_audio_ptr)) {
            throw std::runtime_error("retried DECODE differs from uninterrupted execution");
        }

        double sum_squares = 0.0;
        float peak = 0.0F;
        for (const float value : direct_audio) {
            if (!std::isfinite(value)) throw std::runtime_error("DECODE produced non-finite audio");
            peak = std::max(peak, std::abs(value));
            sum_squares += static_cast<double>(value) * value;
        }
        const double rms = std::sqrt(sum_squares / static_cast<double>(direct_audio.size()));
        if (!(peak > 0.0F) || !(rms > 0.0)) {
            throw std::runtime_error("DECODE produced silent audio");
        }

        std::cout << "{\"codes_equal\":true,\"diffuse_equal\":true,"
                  << "\"decode_equal\":true,\"sample_rate\":" << direct_rate
                  << ",\"samples_per_channel\":" << direct_samples
                  << ",\"peak\":" << peak
                  << ",\"rms\":" << rms
                  << ",\"codes_bytes\":" << direct_codes.size
                  << ",\"diffuse_bytes\":" << direct_latents.size
                  << ",\"audio_fnv1a64\":" << audio_fingerprint(direct_audio)
                  << "}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
