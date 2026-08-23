#include "minimax.h"

#include <cassert>
#include <cmath>

int main() {
    const auto backends = minimax::available_backends();
    assert(!backends.empty());
    const auto output = minimax::backend_smoke(minimax::backend_kind::cpu);
    assert(output.size() == 4);
    for (const float value : output) assert(std::fabs(value - 5.0F) < 1.0e-6F);
}
