#include "minimax-chunking.h"

#include "minimax-config.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace minimax {

std::int64_t condition_frame_count(std::int64_t ar_frames) {
    if (ar_frames <= 0) throw std::invalid_argument("AR frame count must be positive");
    const auto & config = default_model_config.condition;
    return static_cast<std::int64_t>(
        static_cast<double>(ar_frames) * static_cast<double>(config.output_sample_rate) /
        static_cast<double>(config.input_sample_rate) * static_cast<double>(config.input_hop) /
        static_cast<double>(config.output_hop));
}

std::vector<acoustic_window> acoustic_windows(std::int64_t ar_frames) {
    if (ar_frames <= 0 || ar_frames > default_model_config.max_audio_frames) {
        throw std::invalid_argument("AR frame count is outside the supported range");
    }
    std::vector<acoustic_window> result;
    if (ar_frames <= default_model_config.window_frames) {
        result.push_back({0, ar_frames, condition_frame_count(ar_frames)});
        return result;
    }
    for (std::int64_t start = 0; start < ar_frames - default_model_config.window_hop;
         start += default_model_config.window_hop) {
        const auto count = std::min(default_model_config.window_frames, ar_frames - start);
        result.push_back({start, count, condition_frame_count(count)});
    }
    return result;
}

std::uint64_t projected_condition_bytes(std::int64_t ar_frames) {
    std::uint64_t occurrences = 0;
    for (const auto & window : acoustic_windows(ar_frames)) {
        occurrences += static_cast<std::uint64_t>(window.frame_count);
    }
    constexpr std::uint64_t bf16_bytes = 2;
    const auto channels = static_cast<std::uint64_t>(default_model_config.condition.output_size);
    if (occurrences > std::numeric_limits<std::uint64_t>::max() / channels / bf16_bytes) {
        throw std::overflow_error("projected condition byte count overflow");
    }
    return occurrences * channels * bf16_bytes;
}

} // namespace minimax
