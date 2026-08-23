#pragma once

#include <cstdint>
#include <vector>

namespace minimax {

struct acoustic_window {
    std::int64_t start_frame = 0;
    std::int64_t frame_count = 0;
    std::int64_t condition_frames = 0;
};

std::int64_t condition_frame_count(std::int64_t ar_frames);
std::vector<acoustic_window> acoustic_windows(std::int64_t ar_frames);
std::uint64_t projected_condition_bytes(std::int64_t ar_frames);

} // namespace minimax
