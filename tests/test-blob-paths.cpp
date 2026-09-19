// A content-addressed store names every blob for its digest, so the files
// Cantor hands the engine have no extension at all. Nothing on the load path
// may decide what a file is by looking at its name.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Not assert(): a release build defines NDEBUG and would leave this file
// asserting nothing at all.
void check(bool condition, std::string_view what) {
    if (!condition) {
        std::fprintf(stderr, "test-blob-paths: %.*s\n", static_cast<int>(what.size()), what.data());
        std::abort();
    }
}

// Named like a real blob: sha256 hex, no extension.
constexpr const char * kDigestName =
    "17e8f57e5712ae47f3b6882fe9756b00fb648d6d8376029fbf1ab02daff5dbe8";

constexpr const char * kSidecarName = "tokenizer/tokenizer_config.json";
constexpr const char * kSidecarBody = "{\"model_max_length\":32768}";

// A GGUF with one tensor and one embedded sidecar, written where asked.
void write_gguf(const std::filesystem::path & path) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 4 + 4096;
    params.no_alloc = false;
    ggml_context * tensors = ggml_init(params);
    check(tensors != nullptr, "could not create a ggml context");
    ggml_tensor * weight = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 4);
    ggml_set_name(weight, "proj.weight");
    for (int index = 0; index != 4; ++index) {
        static_cast<float *>(weight->data)[index] = static_cast<float>(index);
    }

    gguf_context * gguf = gguf_init_empty();
    check(gguf != nullptr, "could not create a gguf context");
    gguf_set_val_str(gguf, "general.architecture", "audiocpp");

    const char * names[]{kSidecarName};
    gguf_set_arr_str(gguf, "audiocpp.embedded_files.names", names, 1);
    const std::string body(kSidecarBody);
    const std::uint64_t offsets[]{0, body.size()};
    gguf_set_arr_data(gguf, "audiocpp.embedded_files.offsets", GGUF_TYPE_UINT64, offsets, 2);
    gguf_set_arr_data(gguf, "audiocpp.embedded_files.data", GGUF_TYPE_UINT8, body.data(), body.size());

    gguf_add_tensor(gguf, weight);
    check(gguf_write_to_file(gguf, path.string().c_str(), false), "could not write the GGUF");

    gguf_free(gguf);
    ggml_free(tensors);
}

void write_text(const std::filesystem::path & path, const char * body) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << body;
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "minimax-blob-paths";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "blobs" / "sha256");

    const auto blob = root / "blobs" / "sha256" / kDigestName;
    write_gguf(blob);

    // Recognised by its first four bytes, with no extension to go on.
    check(engine::io::has_gguf_magic(blob), "an extensionless GGUF was not recognised");

    const auto plain = root / "blobs" / "sha256" / "not-a-model";
    write_text(plain, "{\"config\":true}");
    check(!engine::io::has_gguf_magic(plain), "a JSON file was taken for a GGUF");
    // A name is not evidence in either direction.
    const auto liar = root / "liar.gguf";
    write_text(liar, "{\"config\":true}");
    check(!engine::io::has_gguf_magic(liar), "a .gguf name was taken as evidence");
    check(!engine::io::has_gguf_magic(root / "absent"), "a missing file was taken for a GGUF");

    // The tensors open from the extensionless path.
    const auto source = engine::assets::open_tensor_source(blob);
    check(source->has_tensor("proj.weight"), "the tensor is missing from the opened source");
    check(source->require_f32("proj.weight", {4}).size() == 4, "the tensor did not read back");

    // And its embedded sidecars are materialised, rather than the parent
    // directory being treated as a model root that holds none of them.
    const auto prepared = engine::assets::prepare_model_directory(blob);
    check(prepared.standalone_gguf.has_value(), "the blob was not treated as a standalone GGUF");
    check(engine::io::is_existing_file(prepared.model_root / kSidecarName),
          "the embedded sidecar was not materialised");
    check(engine::io::read_text_file(prepared.model_root / kSidecarName) == kSidecarBody,
          "the materialised sidecar has the wrong contents");

    // A file that is neither named nor shaped like a GGUF is still refused.
    bool refused = false;
    try {
        (void)engine::assets::open_tensor_source(plain);
    } catch (const std::exception &) {
        refused = true;
    }
    check(refused, "a file that is not a GGUF was accepted");

    std::filesystem::remove_all(root);
    return 0;
}
