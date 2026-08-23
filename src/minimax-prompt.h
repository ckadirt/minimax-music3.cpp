#pragma once

#include "minimax-tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace minimax {

struct prompt_tokens {
    std::vector<std::int64_t> conditional;
    std::vector<std::int64_t> unconditional;
    std::int32_t audio_end_id = 151670;
    std::int32_t audio_cfg_id = 151654;
    std::int32_t audio_code_offset = 151675;
    std::int32_t semantic_vocab_size = 16384;
};

std::string clean_caption(const std::string & caption);
std::string normalize_lyrics(const std::string & lyrics);

prompt_tokens build_prompt(const ByteLevelBPETokenizer & tokenizer,
                           const std::string & description,
                           const std::string & lyrics,
                           std::size_t max_tokens = 5000);

} // namespace minimax
