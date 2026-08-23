#include "minimax.h"

#include <stdexcept>

namespace minimax {

void generate_wav(const generation_request &, const generation_options &) {
    throw std::runtime_error(
        "this build excludes the native model runtime; rebuild with MINIMAX_BUILD_NATIVE_RUNTIME=ON");
}

} // namespace minimax
