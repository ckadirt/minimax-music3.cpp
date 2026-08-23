#include "minimax-checkpoint.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <exception>
#include <vector>

namespace {

constexpr std::array<char, 8> magic{{'M', 'M', 'X', 'C', 'O', 'D', '0', '1'}};

void expect_rejected(const std::vector<std::uint8_t> & blob) {
    bool rejected = false;
    try {
        (void)minimax::checkpoint::decode(blob, magic);
    } catch (const std::exception &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    using namespace minimax::checkpoint;
    const std::vector<section> sections{
        {section_kind::request_json, {'{', '}', '\n'}},
        {section_kind::ar_codes_i32, {1, 0, 0, 0, 2, 0, 0, 0}},
    };
    const auto blob = encode(magic, 2, sections);
    const auto decoded = decode(blob, magic);
    assert(decoded.version == format_version);
    assert(decoded.stage == 2);
    assert(decoded.sections.size() == sections.size());
    assert(decoded.sections[0].bytes == sections[0].bytes);
    assert(decoded.sections[1].bytes == sections[1].bytes);
    constexpr std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
    assert(hex_digest(sha256(abc.data(), abc.size())) ==
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    auto corrupted = blob;
    corrupted.back() ^= 1U;
    expect_rejected(corrupted);
    corrupted = blob;
    corrupted[0] ^= 1U;
    expect_rejected(corrupted);
    corrupted = blob;
    corrupted.pop_back();
    expect_rejected(corrupted);
    return 0;
}
