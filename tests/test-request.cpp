#include "minimax-request.h"

#include <cassert>
#include <functional>
#include <stdexcept>
#include <string>

namespace {

void rejects(const std::string & source) {
    bool rejected = false;
    try {
        (void) minimax::request_io::parse(source);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    const std::string json =
        R"({"lyrics":"[Verse]\nHello \uD83C\uDFB5","description":"warm synth pop","duration_seconds":20,"seed":18446744073709551615,"cfg_scale":1.5,"sampling":{"top_k":50},"flow":{"seed":17,"euler_steps":30,"cfg_scale":1.7},"output_sample_rate":44100})";
    const auto request = minimax::request_io::parse(json);
    assert(request.lyrics == u8"[Verse]\nHello 🎵");
    assert(request.seed == 18446744073709551615ULL);
    assert(request.flow_seed_present && request.flow_seed == 17);
    assert(request.top_k == 50 && request.euler_steps == 30);
    const auto round_trip = minimax::request_io::parse(minimax::request_io::serialize(request));
    assert(round_trip.lyrics == request.lyrics);
    assert(round_trip.seed == request.seed);
    assert(round_trip.output_sample_rate == 44100);

    const auto defaults = minimax::request_io::parse(
        R"({"lyrics":"la","description":"solo voice","duration_seconds":0.04})");
    assert(defaults.top_k == 50 && defaults.euler_steps == 30);
    assert(!defaults.flow_seed_present);

    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":10,"typo":1})");
    rejects(R"({"lyrics":"la","description":"voice","duration":10})");
    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":301})");
    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":10,"seed":1.0})");
    rejects(R"({"lyrics":"la","lyrics":"dup","description":"voice","duration_seconds":10})");
    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":10,"sampling":{"top_k":0}})");
    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":10,"flow":{"euler_steps":0}})");
    rejects(R"({"lyrics":"la","description":"voice","duration_seconds":10,"output_sample_rate":48000})");
    rejects(std::string("{\"lyrics\":\"") + static_cast<char>(0xff) +
            "\",\"description\":\"voice\",\"duration_seconds\":10}");
    return 0;
}
