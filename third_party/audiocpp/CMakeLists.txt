set(AUDIOCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE AUDIOCPP_FRAMEWORK_SOURCES CONFIGURE_DEPENDS
    "${AUDIOCPP_ROOT}/src/framework/*.cpp")
list(FILTER AUDIOCPP_FRAMEWORK_SOURCES EXCLUDE REGEX
    "/src/framework/(package_manager/|runtime/registry\\.cpp$|tokenizers/sentencepiece\\.cpp$|modules/vocoders/bigvgan_vocoder\\.cpp$|modules/attention/(common_relative_attention|relative_attention)\\.cpp$)")
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
