#include "minimax-chunking.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

int main() {
    assert(minimax::condition_frame_count(200) == 689);
    auto one = minimax::acoustic_windows(200);
    assert(one.size() == 1 && one.front().start_frame == 0 && one.front().frame_count == 200);

    const auto thirty_seconds = minimax::acoustic_windows(750);
    assert(thirty_seconds.size() == 7);
    assert(thirty_seconds.back().start_frame == 600);
    assert(thirty_seconds.back().frame_count == 150);
    assert(minimax::projected_condition_bytes(750) == 5'529'600ULL);

    const auto maximum = minimax::acoustic_windows(9000);
    assert(maximum.size() == 89);
    assert(maximum.back().start_frame == 8800);
    assert(maximum.back().frame_count == 200);
    assert(minimax::projected_condition_bytes(9000) == 72'908'800ULL);

    bool rejected = false;
    try { (void) minimax::acoustic_windows(9001); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
}
