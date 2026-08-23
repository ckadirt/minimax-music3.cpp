#include "minimax.h"
#include "minimax-request.h"

#include "engine/community_models/minimax_music3/session.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax {
namespace {

engine::core::BackendType native_backend(backend_kind backend) {
    switch (backend) {
        case backend_kind::auto_select: return engine::core::BackendType::BestAvailable;
        case backend_kind::cpu: return engine::core::BackendType::Cpu;
        case backend_kind::cuda: return engine::core::BackendType::Cuda;
        case backend_kind::hip: return engine::core::BackendType::Hip;
        case backend_kind::vulkan: return engine::core::BackendType::Vulkan;
        case backend_kind::metal: return engine::core::BackendType::Metal;
    }
    throw std::invalid_argument("unknown MiniMax backend");
}

void resample_stereo(engine::runtime::AudioBuffer & audio, int output_rate) {
    if (audio.sample_rate == output_rate) return;
    if (audio.channels != 2 || audio.samples.size() % 2 != 0) {
        throw std::runtime_error("MiniMax native runtime returned invalid stereo audio");
    }
    std::vector<float> left(audio.samples.size() / 2);
    std::vector<float> right(audio.samples.size() / 2);
    for (std::size_t frame = 0; frame != left.size(); ++frame) {
        left[frame] = audio.samples[2 * frame];
        right[frame] = audio.samples[2 * frame + 1];
    }
    const auto settings = engine::audio::torchaudio_sinc_hann_float32_options();
    left = engine::audio::resample_mono_torchaudio_sinc_hann(left, audio.sample_rate, output_rate, settings);
    right = engine::audio::resample_mono_torchaudio_sinc_hann(right, audio.sample_rate, output_rate, settings);
    const std::size_t frames = std::min(left.size(), right.size());
    audio.samples.resize(frames * 2);
    for (std::size_t frame = 0; frame != frames; ++frame) {
        audio.samples[2 * frame] = left[frame];
        audio.samples[2 * frame + 1] = right[frame];
    }
    audio.sample_rate = output_rate;
}

} // namespace

void generate_wav(const generation_request & request, const generation_options & options) {
    request_io::validate(request);
    if (options.model_directory.empty()) throw std::invalid_argument("model_directory must not be empty");
    if (options.output_wav.empty()) throw std::invalid_argument("output_wav must not be empty");
    if (options.device_index < 0) throw std::invalid_argument("device_index cannot be negative");
    if (options.threads <= 0) throw std::invalid_argument("threads must be positive");

    auto loader = engine::models::minimax_music3::make_minimax_music3_loader();
    engine::runtime::ModelLoadRequest load;
    load.model_path = options.model_directory;
    load.family_hint = "minimax_music3";
    auto model = loader->load(load);

    engine::runtime::SessionOptions session_options;
    session_options.backend = {native_backend(options.backend), options.device_index, options.threads};
    session_options.options["minimax_music3.mem_saver"] = "true";
    if (!options.lm_gguf.empty()) {
        session_options.options["minimax_music3.language_model_gguf"] = options.lm_gguf;
    }
    if (!options.rvq_gguf.empty()) {
        session_options.options["minimax_music3.rvq_depth_decoder_gguf"] = options.rvq_gguf;
    }
    if (!options.condition_gguf.empty()) {
        session_options.options["minimax_music3.condition_encoder_gguf"] = options.condition_gguf;
    }
    if (!options.dit_gguf.empty()) {
        session_options.options["minimax_music3.flow_transformer_gguf"] = options.dit_gguf;
    }
    if (!options.vae_gguf.empty()) {
        session_options.options["minimax_music3.vocoder_gguf"] = options.vae_gguf;
    }
    auto session = model->create_task_session(
        {engine::runtime::VoiceTaskKind::AudioGeneration, engine::runtime::RunMode::Offline},
        session_options);
    session->prepare({});
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) throw std::runtime_error("MiniMax loader did not create an offline session");

    engine::runtime::TaskRequest native_request;
    native_request.text_input = engine::runtime::Transcript{request.description, "auto"};
    native_request.options = {
        {"lyrics", request.lyrics},
        {"duration_sec", std::to_string(request.duration_seconds)},
        {"num_inference_steps", std::to_string(request.euler_steps)},
        {"guidance_scale", std::to_string(request.flow_cfg_scale)},
        {"ar_guidance_scale", std::to_string(request.ar_cfg_scale)},
        {"top_k", std::to_string(request.top_k)},
        {"seed", std::to_string(request.seed)},
    };
    auto result = offline->run(native_request);
    if (!result.audio_output.has_value()) throw std::runtime_error("MiniMax runtime returned no audio");
    auto audio = std::move(*result.audio_output);
    resample_stereo(audio, request.output_sample_rate);
    engine::audio::write_pcm16_wav(options.output_wav, audio.sample_rate, audio.channels, audio.samples);
}

} // namespace minimax
