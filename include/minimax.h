#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax {

enum class backend_kind {
    auto_select,
    cpu,
    cuda,
    gpu,
};

struct backend_info {
    std::string name;
    std::string description;
    backend_kind kind = backend_kind::cpu;
    std::size_t memory_free = 0;
    std::size_t memory_total = 0;
};

struct generation_request {
    std::string lyrics;
    std::string description;
    double duration_seconds = 20.0;
    std::uint64_t seed = 0;
    float ar_cfg_scale = 1.5F;
    std::size_t top_k = 50;
    std::uint64_t flow_seed = 0;
    bool flow_seed_present = false;
    std::size_t euler_steps = 30;
    float flow_cfg_scale = 1.7F;
    int output_sample_rate = 44100;
};

class operation_cancelled : public std::runtime_error {
public:
    operation_cancelled() : std::runtime_error("MiniMax Music 3 operation cancelled") {}
};

const char * version() noexcept;
std::vector<backend_info> available_backends();
std::vector<float> backend_smoke(backend_kind kind = backend_kind::auto_select,
                                 int device_index = 0);

} // namespace minimax
