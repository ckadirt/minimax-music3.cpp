// Adapted for minimax-music3.cpp from audio.cpp at
// 62735eafd96294c52d6c4607f5f38ac55be54f06. Modified file.
// Copyright 2026 ShugoAI LLC. Licensed under Apache-2.0.

#include "minimax-prompt.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <utility>

namespace minimax {
namespace {

constexpr const char * im_start = "<|im_start|>";
constexpr const char * im_end = "<|im_end|>";
constexpr const char * caption_start = "<|caption_start|>";
constexpr const char * caption_end = "<|caption_end|>";
constexpr const char * lyrics_start = "<|lyrics_start|>";
constexpr const char * lyrics_end = "<|lyrics_end|>";
constexpr const char * audio_start = "<|audio_start|>";
constexpr std::int64_t audio_cfg_id = 151654;

std::string strip_markdown_line(std::string line) {
    line = std::regex_replace(line, std::regex(R"(^\s{0,3}#{1,6}\s+)"), "");
    line = std::regex_replace(line, std::regex(R"(^\s*[*+-]\s+)"), "");
    for (;;) {
        auto updated = std::regex_replace(line, std::regex(R"(\*\*([^*]+)\*\*)"), "$1");
        if (updated == line) break;
        line = std::move(updated);
    }
    line = std::regex_replace(line, std::regex(R"(\*([^*\n]+)\*)"), "$1");
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
    return line;
}

} // namespace

std::string clean_caption(const std::string & caption) {
    const std::regex tag(R"(<\|([^|]*)\|>)");
    std::string text;
    std::sregex_iterator iterator(caption.begin(), caption.end(), tag);
    const std::sregex_iterator finish;
    std::size_t cursor = 0;
    for (; iterator != finish; ++iterator) {
        const auto & match = *iterator;
        text.append(caption, cursor, static_cast<std::size_t>(match.position()) - cursor);
        std::string inner = match.str(1);
        while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.front()))) inner.erase(inner.begin());
        while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.back()))) inner.pop_back();
        const auto split = inner.find_first_of(" \t\n\r\f\v");
        if (split == std::string::npos) {
            text += inner;
        } else {
            std::size_t rest = split;
            while (rest < inner.size() && std::isspace(static_cast<unsigned char>(inner[rest]))) ++rest;
            text += inner.substr(0, split) + " is " + inner.substr(rest);
        }
        cursor = static_cast<std::size_t>(match.position() + match.length());
    }
    text.append(caption, cursor, std::string::npos);

    std::string output;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find('\n', start);
        auto line = strip_markdown_line(text.substr(
            start, end == std::string::npos ? std::string::npos : end - start));
        if (!std::regex_match(line, std::regex(R"(^\s*[-*_]{3,}\s*$)"))) {
            if (!output.empty()) output.push_back('\n');
            output += line;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    output = std::regex_replace(output, std::regex(u8"• "), "");
    output = std::regex_replace(output, std::regex("    "), "");
    return std::regex_replace(output, std::regex(R"(\n{2,})"), "\n");
}

std::string normalize_lyrics(const std::string & lyrics) {
    std::string output;
    std::size_t start = 0;
    const std::regex leading_tags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
    while (start <= lyrics.size()) {
        const auto end = lyrics.find('\n', start);
        std::string line = lyrics.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        std::smatch match;
        if (std::regex_search(line, match, leading_tags)) {
            line = match.str(1);
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        }
        if (!output.empty()) output.push_back('\n');
        output += line;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    output = std::regex_replace(output, std::regex(R"(\] )"), "]\n");
    output = std::regex_replace(output, std::regex(R"( \[)"), "\n[");
    output = std::regex_replace(output, std::regex(R"( \^ )"), "\n");

    std::string lowered;
    lowered.reserve(output.size());
    for (std::size_t index = 0; index < output.size();) {
        if (output[index] == '[') {
            const auto close = output.find(']', index + 1);
            if (close != std::string::npos) {
                lowered.push_back('[');
                for (auto cursor = index + 1; cursor < close; ++cursor) {
                    lowered.push_back(static_cast<char>(
                        std::tolower(static_cast<unsigned char>(output[cursor]))));
                }
                lowered.push_back(']');
                index = close + 1;
                continue;
            }
        }
        lowered.push_back(output[index++]);
    }
    return "[start]\n" + lowered;
}

prompt_tokens build_prompt(const ByteLevelBPETokenizer & tokenizer,
                           const std::string & description,
                           const std::string & lyrics,
                           std::size_t max_tokens) {
    if (description.empty()) throw std::invalid_argument("MiniMax Music 3 requires a non-empty description");
    if (lyrics.empty()) throw std::invalid_argument("MiniMax Music 3 requires non-empty lyrics");
    const std::string text = std::string(im_start) + caption_start + clean_caption(description) +
                             caption_end + lyrics_start + normalize_lyrics(lyrics) + lyrics_end +
                             im_end + audio_start;
    prompt_tokens result;
    result.conditional = tokenizer.encode(text);
    if (result.conditional.size() > max_tokens) {
        throw std::invalid_argument("MiniMax Music 3 prompt exceeds the 5,000-token limit");
    }
    result.unconditional = result.conditional;
    if (result.unconditional.size() >= 3) {
        std::fill(result.unconditional.begin() + 1, result.unconditional.end() - 2, audio_cfg_id);
    }
    return result;
}

} // namespace minimax
