#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/flow_transformer.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::assets::TensorStorageType;
using engine::core::BackendConfig;
using engine::core::BackendType;
using engine::models::minimax_music3::MiniMaxMusic3Assets;

constexpr std::size_t kGraphArenaBytes = 32ull * 1024ull * 1024ull;
constexpr std::size_t kWeightContextBytes = 32ull * 1024ull * 1024ull;

struct Options {
    std::filesystem::path model_dir;
    std::filesystem::path input;
    std::filesystem::path condition;
    std::filesystem::path output;
    std::string component;
    std::string backend = "cuda";
    std::int64_t frames = 0;
    float timestep = 0.375F;
    int device = 0;
    int threads = 1;
};

[[noreturn]] void usage(const std::string & message = {}) {
    if (!message.empty()) {
        std::cerr << "error: " << message << "\n\n";
    }
    std::cerr
        << "Usage: minimax-parity --model-dir DIR --component condition|dit|vocoder\n"
        << "                      --input INPUT.f32 [--condition CONDITION.f32]\n"
        << "                      --output OUTPUT.f32 --frames N\n"
        << "                      [--timestep T] [--backend cpu|cuda|hip|vulkan|metal]\n"
        << "                      [--device N] [--threads N]\n";
    throw std::invalid_argument("invalid command line");
}

std::int64_t parse_i64(const std::string & text, const char * name) {
    std::size_t used = 0;
    long long result = 0;
    try {
        result = std::stoll(text, &used, 10);
    } catch (const std::exception &) {
        usage(std::string(name) + " must be an integer");
    }
    if (used != text.size()) usage(std::string(name) + " must be an integer");
    return static_cast<std::int64_t>(result);
}

float parse_float(const std::string & text, const char * name) {
    std::size_t used = 0;
    float result = 0.0F;
    try {
        result = std::stof(text, &used);
    } catch (const std::exception &) {
        usage(std::string(name) + " must be a finite float");
    }
    if (used != text.size() || !std::isfinite(result)) {
        usage(std::string(name) + " must be a finite float");
    }
    return result;
}

Options parse_options(int argc, char ** argv) {
    const std::vector<std::string> flag_names{
        "--model-dir", "--component", "--input", "--condition", "--output",
        "--frames", "--timestep", "--backend", "--device", "--threads"};
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        const std::string flag = argv[index];
        if (flag == "--help" || flag == "-h") usage();
        if (index + 1 >= argc) usage("missing value for " + flag);
        if (std::find(flag_names.begin(), flag_names.end(), flag) == flag_names.end()) {
            usage("unknown option " + flag);
        }
        if (!values.emplace(flag, argv[index + 1]).second) usage("duplicate option " + flag);
    }
    const auto require = [&values](const char * name) -> const std::string & {
        const auto found = values.find(name);
        if (found == values.end() || found->second.empty()) usage(std::string("missing ") + name);
        return found->second;
    };
    Options out;
    out.model_dir = require("--model-dir");
    out.component = require("--component");
    out.input = require("--input");
    out.output = require("--output");
    out.frames = parse_i64(require("--frames"), "--frames");
    if (const auto it = values.find("--condition"); it != values.end()) out.condition = it->second;
    if (const auto it = values.find("--timestep"); it != values.end()) {
        out.timestep = parse_float(it->second, "--timestep");
    }
    if (const auto it = values.find("--backend"); it != values.end()) out.backend = it->second;
    if (const auto it = values.find("--device"); it != values.end()) {
        const auto value = parse_i64(it->second, "--device");
        if (value < 0 || value > std::numeric_limits<int>::max()) usage("--device is out of range");
        out.device = static_cast<int>(value);
    }
    if (const auto it = values.find("--threads"); it != values.end()) {
        const auto value = parse_i64(it->second, "--threads");
        if (value <= 0 || value > std::numeric_limits<int>::max()) usage("--threads is out of range");
        out.threads = static_cast<int>(value);
    }
    if (out.frames <= 0) usage("--frames must be positive");
    if (out.component != "condition" && out.component != "dit" && out.component != "vocoder") {
        usage("--component must be condition, dit, or vocoder");
    }
    if (out.component == "dit" && out.condition.empty()) usage("dit requires --condition");
    if (out.component != "dit" && !out.condition.empty()) usage("--condition is only valid for dit");
    if (out.timestep < 0.0F || out.timestep > 1.0F) usage("--timestep must be in [0, 1]");
    return out;
}

BackendType parse_backend(const std::string & value) {
    if (value == "cpu") return BackendType::Cpu;
    if (value == "cuda") return BackendType::Cuda;
    if (value == "hip") return BackendType::Hip;
    if (value == "vulkan") return BackendType::Vulkan;
    if (value == "metal") return BackendType::Metal;
    usage("unknown --backend " + value);
}

std::vector<float> read_f32(const std::filesystem::path & path, std::size_t expected) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("input is not an existing regular file: " + path.string());
    }
    const auto expected_bytes = expected * sizeof(float);
    if (std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error(
            "input size mismatch for " + path.string() + ": expected " +
            std::to_string(expected_bytes) + " bytes");
    }
    std::vector<float> values(expected);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(expected_bytes));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("failed to read exact input: " + path.string());
    }
    return values;
}

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("failed to open output: " + path.string());
    stream.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("failed to write output: " + path.string());
}

std::shared_ptr<const MiniMaxMusic3Assets> dense_assets(const std::filesystem::path & directory) {
    auto selected = std::make_shared<MiniMaxMusic3Assets>(
        *engine::models::minimax_music3::load_minimax_music3_assets(directory));
    selected->condition_encoder_weights = engine::assets::open_tensor_source(directory / "condition-F32.gguf");
    selected->transformer_weights = engine::assets::open_tensor_source(directory / "dit-F32.gguf");
    selected->vocoder_weights = engine::assets::open_tensor_source(directory / "vae-F32.gguf");
    engine::models::minimax_music3::validate_minimax_music3_anchors(*selected);
    return selected;
}

std::vector<float> run_condition(
    const Options & options,
    const std::shared_ptr<const MiniMaxMusic3Assets> & assets,
    engine::core::ExecutionContext & execution,
    std::vector<std::int64_t> & shape) {
    const auto & config = assets->config.condition;
    const auto input = read_f32(
        options.input,
        static_cast<std::size_t>(options.frames * config.condition_layers * config.condition_hidden_dim));
    engine::models::minimax_music3::MiniMaxMusic3ConditionEncoderRuntime runtime(
        assets, execution, kGraphArenaBytes, kWeightContextBytes, TensorStorageType::Native);
    std::int64_t output_frames = 0;
    auto output = runtime.encode(input, options.frames, output_frames);
    shape = {1, output_frames, config.out_dim};
    return output;
}

std::vector<float> run_dit(
    const Options & options,
    const std::shared_ptr<const MiniMaxMusic3Assets> & assets,
    engine::core::ExecutionContext & execution,
    std::vector<std::int64_t> & shape) {
    const auto & config = assets->config.flow;
    const auto latents = read_f32(
        options.input, static_cast<std::size_t>(config.in_channels * options.frames));
    const auto condition = read_f32(
        options.condition, static_cast<std::size_t>(config.condition_dim * options.frames));
    engine::models::minimax_music3::MiniMaxMusic3FlowTransformerRuntime runtime(
        assets, execution, kGraphArenaBytes, kWeightContextBytes, TensorStorageType::Native);
    runtime.prepare_chunk_condition(condition, options.frames);
    auto output = runtime.predict_velocity_branches(latents, condition, options.frames, options.timestep);
    shape = {2, config.in_channels, options.frames};
    return output;
}

std::vector<float> run_vocoder(
    const Options & options,
    const std::shared_ptr<const MiniMaxMusic3Assets> & assets,
    engine::core::ExecutionContext & execution,
    std::vector<std::int64_t> & shape) {
    const auto & config = assets->config.vocoder;
    const auto latents = read_f32(
        options.input, static_cast<std::size_t>(config.latent_channels * options.frames));
    engine::models::minimax_music3::MiniMaxMusic3VocoderRuntime runtime(
        assets, execution, kGraphArenaBytes, kWeightContextBytes, TensorStorageType::Native);
    auto audio = runtime.decode(latents, options.frames);
    if (audio.channels != 2 || audio.samples.size() % 2 != 0) {
        throw std::runtime_error("vocoder returned invalid stereo output");
    }
    const auto samples = static_cast<std::int64_t>(audio.samples.size() / 2);
    std::vector<float> planar(audio.samples.size());
    for (std::int64_t sample = 0; sample < samples; ++sample) {
        planar[static_cast<std::size_t>(sample)] = audio.samples[static_cast<std::size_t>(2 * sample)];
        planar[static_cast<std::size_t>(samples + sample)] =
            audio.samples[static_cast<std::size_t>(2 * sample + 1)];
    }
    shape = {1, 2, samples};
    return planar;
}

void print_result(const Options & options, const std::vector<std::int64_t> & shape, std::size_t elements) {
    std::cout << "{\"component\":\"" << options.component << "\",\"backend\":\""
              << options.backend << "\",\"elements\":" << elements << ",\"shape\":[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << shape[index];
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const std::uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
            throw std::runtime_error("parity fixture I/O requires a little-endian host");
        }
        engine::core::ExecutionContext execution(
            BackendConfig{parse_backend(options.backend), options.device, options.threads});
        const auto assets = dense_assets(options.model_dir);
        std::vector<std::int64_t> shape;
        std::vector<float> output;
        if (options.component == "condition") {
            output = run_condition(options, assets, execution, shape);
        } else if (options.component == "dit") {
            output = run_dit(options, assets, execution, shape);
        } else {
            output = run_vocoder(options, assets, execution, shape);
        }
        write_f32(options.output, output);
        print_result(options, shape, output.size());
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
