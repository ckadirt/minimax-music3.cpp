#include "minimax.h"
#include "minimax-request.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const char * kind_name(minimax::backend_kind kind) {
    switch (kind) {
        case minimax::backend_kind::cuda: return "cuda";
        case minimax::backend_kind::hip: return "hip";
        case minimax::backend_kind::vulkan: return "vulkan";
        case minimax::backend_kind::metal: return "metal";
        case minimax::backend_kind::cpu: return "cpu";
        case minimax::backend_kind::auto_select: return "auto";
    }
    return "unknown";
}

minimax::backend_kind parse_kind(const std::string & value) {
    if (value == "auto") return minimax::backend_kind::auto_select;
    if (value == "cpu") return minimax::backend_kind::cpu;
    if (value == "cuda") return minimax::backend_kind::cuda;
    if (value == "hip" || value == "rocm") return minimax::backend_kind::hip;
    if (value == "vulkan") return minimax::backend_kind::vulkan;
    if (value == "metal") return minimax::backend_kind::metal;
    throw std::invalid_argument("unknown backend kind: " + value);
}

void usage(const char * program) {
    std::cerr << "minimax-music3.cpp " << minimax::version() << "\n\n"
              << "Usage:\n"
              << "  " << program << " --list-backends\n"
              << "  " << program << " --smoke [auto|cpu|cuda|hip|vulkan|metal] [device-index]\n"
              << "  " << program << " --validate-request REQUEST.json\n"
              << "  " << program << " --generate REQUEST.json --model DIR --output FILE.wav"
                 " [--backend KIND] [--device N] [--threads N] [--lm FILE] [--rvq FILE]"
                 " [--condition FILE] [--dit FILE] [--vae FILE]\n";
}

std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open request: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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
        if (command == "--validate-request" && argc == 3) {
            const auto request = minimax::request_io::parse(read_file(argv[2]));
            std::cout << minimax::request_io::serialize(request) << '\n';
            return 0;
        }
        if (command == "--generate" && argc >= 7) {
            const auto request = minimax::request_io::parse(read_file(argv[2]));
            minimax::generation_options options;
            for (int index = 3; index < argc; index += 2) {
                if (index + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(argv[index]));
                const std::string option = argv[index];
                const std::string value = argv[index + 1];
                if (option == "--model") options.model_directory = value;
                else if (option == "--output") options.output_wav = value;
                else if (option == "--backend") options.backend = parse_kind(value);
                else if (option == "--device") options.device_index = std::stoi(value);
                else if (option == "--threads") options.threads = std::stoi(value);
                else if (option == "--lm") options.lm_gguf = value;
                else if (option == "--rvq") options.rvq_gguf = value;
                else if (option == "--condition") options.condition_gguf = value;
                else if (option == "--dit") options.dit_gguf = value;
                else if (option == "--vae") options.vae_gguf = value;
                else throw std::invalid_argument("unknown generation option: " + option);
            }
            minimax::generate_wav(request, options);
            return 0;
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
