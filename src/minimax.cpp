#include "minimax.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax {
namespace {

struct backend_deleter {
    void operator()(ggml_backend_t value) const noexcept {
        if (value != nullptr) ggml_backend_free(value);
    }
};

struct context_deleter {
    void operator()(ggml_context * value) const noexcept {
        if (value != nullptr) ggml_free(value);
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t value) const noexcept {
        if (value != nullptr) ggml_backend_buffer_free(value);
    }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

bool gpu_device(ggml_backend_dev_t device) {
    const auto type = ggml_backend_dev_type(device);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

bool cuda_device(ggml_backend_dev_t device) {
    if (!gpu_device(device)) return false;
    auto * reg = ggml_backend_dev_backend_reg(device);
    const std::string name = reg == nullptr ? "" : ggml_backend_reg_name(reg);
    return name == "CUDA" || name == "MUSA";
}

std::string registry_name(ggml_backend_dev_t device) {
    auto * reg = ggml_backend_dev_backend_reg(device);
    return reg == nullptr ? "" : ggml_backend_reg_name(reg);
}

backend_kind classify(ggml_backend_dev_t device) {
    if (cuda_device(device)) return backend_kind::cuda;
    const auto name = registry_name(device);
    if (name == "ROCm") return backend_kind::hip;
    if (name == "Vulkan") return backend_kind::vulkan;
    if (name == "MTL") return backend_kind::metal;
    return backend_kind::cpu;
}

ggml_backend_dev_t select_device(backend_kind requested, int device_index) {
    if (device_index < 0) throw std::invalid_argument("device index cannot be negative");
    ggml_backend_load_all();
    const auto find_nth = [device_index](const auto & predicate) {
        int match = 0;
        for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            auto * device = ggml_backend_dev_get(index);
            if (predicate(device) && match++ == device_index) return device;
        }
        return static_cast<ggml_backend_dev_t>(nullptr);
    };
    if (requested == backend_kind::auto_select) {
        if (auto * device = find_nth(cuda_device)) return device;
        if (auto * device = find_nth(gpu_device)) return device;
        if (auto * device = find_nth([](ggml_backend_dev_t value) {
                return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
            })) return device;
    } else if (requested == backend_kind::cuda) {
        if (auto * device = find_nth(cuda_device)) return device;
    } else if (requested == backend_kind::hip) {
        if (auto * device = find_nth([](ggml_backend_dev_t value) { return registry_name(value) == "ROCm"; })) return device;
    } else if (requested == backend_kind::vulkan) {
        if (auto * device = find_nth([](ggml_backend_dev_t value) { return registry_name(value) == "Vulkan"; })) return device;
    } else if (requested == backend_kind::metal) {
        if (auto * device = find_nth([](ggml_backend_dev_t value) { return registry_name(value) == "MTL"; })) return device;
    } else if (auto * device = find_nth([](ggml_backend_dev_t value) {
            return ggml_backend_dev_type(value) == GGML_BACKEND_DEVICE_TYPE_CPU;
        })) return device;
    throw std::runtime_error("requested GGML backend device is unavailable");
}

} // namespace

const char * version() noexcept { return MINIMAX_VERSION; }

std::vector<backend_info> available_backends() {
    ggml_backend_load_all();
    std::vector<backend_info> result;
    result.reserve(ggml_backend_dev_count());
    for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        auto * device = ggml_backend_dev_get(index);
        backend_info info;
        info.name = ggml_backend_dev_name(device);
        info.description = ggml_backend_dev_description(device);
        info.kind = classify(device);
        ggml_backend_dev_memory(device, &info.memory_free, &info.memory_total);
        result.push_back(std::move(info));
    }
    return result;
}

std::vector<float> backend_smoke(backend_kind kind, int device_index) {
    backend_ptr backend(ggml_backend_dev_init(select_device(kind, device_index), nullptr));
    if (!backend) throw std::runtime_error("failed to initialize requested GGML backend");

    const std::size_t context_size = 3 * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(8, false);
    std::vector<std::byte> storage(context_size);
    context_ptr context(ggml_init({context_size, storage.data(), true}));
    if (!context) throw std::runtime_error("failed to create GGML smoke context");

    auto * lhs = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * rhs = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, 4);
    auto * sum = ggml_add(context.get(), lhs, rhs);
    auto * graph = ggml_new_graph_custom(context.get(), 8, false);
    ggml_build_forward_expand(graph, sum);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend.get()));
    if (!buffer) throw std::runtime_error("failed to allocate GGML smoke tensors");

    const std::array<float, 4> left{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> right{4.0F, 3.0F, 2.0F, 1.0F};
    ggml_backend_tensor_set(lhs, left.data(), 0, sizeof(left));
    ggml_backend_tensor_set(rhs, right.data(), 0, sizeof(right));
    const auto status = ggml_backend_graph_compute(backend.get(), graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("GGML smoke graph failed: ") + ggml_status_to_string(status));
    }
    std::vector<float> output(4);
    ggml_backend_tensor_get(sum, output.data(), 0, output.size() * sizeof(float));
    return output;
}

} // namespace minimax
