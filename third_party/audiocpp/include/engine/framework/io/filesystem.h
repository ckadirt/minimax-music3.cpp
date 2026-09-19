#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::io {

bool is_existing_directory(const std::filesystem::path & path);
bool is_existing_file(const std::filesystem::path & path);

/// True when the file starts with the GGUF magic, whatever it is called.
///
/// A content-addressed store names a blob for its digest and gives it no
/// extension, so a filename cannot decide whether something is a GGUF. Only
/// the first four bytes are read.
bool has_gguf_magic(const std::filesystem::path & path);

std::filesystem::path require_directory(const std::filesystem::path & path, std::string_view role);
std::filesystem::path require_file(const std::filesystem::path & path, std::string_view role);

std::optional<std::filesystem::path> find_first_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<std::filesystem::path> collect_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::string read_text_file(const std::filesystem::path & path);

}  // namespace engine::io
