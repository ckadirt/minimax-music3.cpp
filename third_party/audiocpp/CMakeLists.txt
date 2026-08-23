set(AUDIOCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Framework translation units reachable from the MiniMax session, Cantor ABI,
# CLI, and converter. Unrelated ASR, enhancement, pitch, and codec modules from
# upstream audio.cpp deliberately do not enter this standalone port.
set(AUDIOCPP_FRAMEWORK_SOURCES
    src/framework/assets/resource_bundle.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/weight_norm_folding.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/core/backend.cpp
    src/framework/core/execution_context.cpp
    src/framework/core/module.cpp
    src/framework/debug/profiler.cpp
    src/framework/debug/trace.cpp
    src/framework/io/binary.cpp
    src/framework/io/filesystem.cpp
    src/framework/io/json.cpp
    src/framework/io/safetensors.cpp
    src/framework/io/yaml.cpp
    src/framework/model_spec/options.cpp
    src/framework/model_spec/package.cpp
    src/framework/model_spec/schema.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/attention/scaled_dot_product_attention.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/flow_sampler_runtime.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/transformers/qwen_causal_decode_runtime.cpp
    src/framework/modules/transformers/qwen_causal_decoder.cpp
    src/framework/modules/transformers/qwen_decoder.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/model.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/session_base.cpp
    src/framework/sampling/hf_sampler.cpp
    src/framework/sampling/torch_random.cpp
    src/framework/tokenizers/llama_bpe.cpp)
list(TRANSFORM AUDIOCPP_FRAMEWORK_SOURCES PREPEND "${AUDIOCPP_ROOT}/")
file(GLOB AUDIOCPP_MINIMAX_SOURCES CONFIGURE_DEPENDS
    "${AUDIOCPP_ROOT}/src/community_models/minimax_music3/*.cpp")
file(GLOB AUDIOCPP_LLAMA_TOKENIZER_SOURCES CONFIGURE_DEPENDS
    "${AUDIOCPP_ROOT}/external/llama_tokenizer/*.cpp")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_specs.inc" CONTENT "")
file(READ "${PROJECT_SOURCE_DIR}/model_specs/minimax_music3.json" AUDIOCPP_MINIMAX_SPEC)
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gguf_converter_model_specs.inc"
    CONTENT "    {\"minimax_music3\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MINIMAX_SPEC})AUDIOCPP_SPEC\"},\n")

add_library(minimax-audiocpp-cjson STATIC external/cJSON/cJSON.c)
target_include_directories(minimax-audiocpp-cjson PUBLIC external/cJSON)

add_library(minimax-audiocpp-yaml STATIC
    external/libyaml/src/api.c
    external/libyaml/src/reader.c
    external/libyaml/src/scanner.c
    external/libyaml/src/parser.c
    external/libyaml/src/loader.c)
target_include_directories(minimax-audiocpp-yaml PUBLIC external/libyaml/include)
target_compile_definitions(minimax-audiocpp-yaml PUBLIC
    _POSIX_C_SOURCE=200809L YAML_DECLARE_STATIC=1 YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2 YAML_VERSION_PATCH=5 YAML_VERSION_STRING="0.2.5")

add_library(minimax-native STATIC
    ${AUDIOCPP_FRAMEWORK_SOURCES}
    ${AUDIOCPP_MINIMAX_SOURCES}
    ${AUDIOCPP_LLAMA_TOKENIZER_SOURCES})
add_library(minimax::native ALIAS minimax-native)
target_include_directories(minimax-native
    PUBLIC ${AUDIOCPP_ROOT}/include
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${AUDIOCPP_ROOT}/external/llama_tokenizer
        ${AUDIOCPP_ROOT}/external/cJSON
        ${AUDIOCPP_ROOT}/external/libyaml/include)
target_link_libraries(minimax-native
    PUBLIC ggml::ggml
    PRIVATE minimax-audiocpp-cjson minimax-audiocpp-yaml Threads::Threads)
target_compile_features(minimax-native PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(minimax-native PRIVATE /utf-8)
endif()

add_executable(minimax-gguf-convert app/gguf/main.cpp)
target_include_directories(minimax-gguf-convert PRIVATE
    ${AUDIOCPP_ROOT}/include ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(minimax-gguf-convert PRIVATE minimax::native)
set_target_properties(minimax-gguf-convert PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
