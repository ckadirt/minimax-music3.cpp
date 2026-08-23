#pragma once

#include "minimax.h"

#include <string>

namespace minimax::request_io {

// Parse the Cantor v1 request. The schema is deliberately closed: misspelled
// or future fields fail instead of silently changing generation semantics.
generation_request parse(const std::string & json);

// Validate a request assembled by an embedding rather than parse().
void validate(const generation_request & request);

// Stable, compact JSON used by durable-state provenance. An absent flow seed
// remains absent so that the runtime can apply the pinned upstream derivation.
std::string serialize(const generation_request & request);

} // namespace minimax::request_io
