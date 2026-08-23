#include "minimax.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const char * kind_name(minimax::backend_kind kind) {
    switch (kind) {
        case minimax::backend_kind::cuda: return "cuda";
        case minimax::backend_kind::gpu: return "gpu";
        case minimax::backend_kind::cpu: return "cpu";
        case minimax::backend_kind::auto_select: return "auto";
    }
    return "unknown";
}

minimax::backend_kind parse_kind(const std::string & value) {
    if (value == "auto") return minimax::backend_kind::auto_select;
    if (value == "cpu") return minimax::backend_kind::cpu;
    if (value == "cuda") return minimax::backend_kind::cuda;
    if (value == "gpu" || value == "vulkan" || value == "metal") return minimax::backend_kind::gpu;
    throw std::invalid_argument("unknown backend kind: " + value);
}

void usage(const char * program) {
    std::cerr << "minimax-music3.cpp " << minimax::version() << "\n\n"
              << "Usage:\n"
              << "  " << program << " --list-backends\n"
              << "  " << program << " --smoke [auto|cpu|cuda|gpu] [device-index]\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 2;
        }
        const std::string command = argv[1];
        if (command == "--list-backends") {
            const auto backends = minimax::available_backends();
            for (std::size_t index = 0; index < backends.size(); ++index) {
                const auto & item = backends[index];
                std::cout << index << '\t' << kind_name(item.kind) << '\t' << item.name << '\t'
                          << item.description << '\t' << item.memory_free << '/' << item.memory_total << '\n';
            }
            return backends.empty() ? 1 : 0;
        }
        if (command == "--smoke") {
            const auto kind = argc >= 3 ? parse_kind(argv[2]) : minimax::backend_kind::auto_select;
            const int device = argc >= 4 ? std::stoi(argv[3]) : 0;
            const auto result = minimax::backend_smoke(kind, device);
            for (std::size_t index = 0; index < result.size(); ++index) {
                if (index) std::cout << ' ';
                std::cout << std::setprecision(8) << result[index];
            }
            std::cout << '\n';
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
