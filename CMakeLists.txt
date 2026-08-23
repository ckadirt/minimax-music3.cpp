cmake_minimum_required(VERSION 3.20)

project(minimax-music3
    VERSION 0.1.0
    DESCRIPTION "Portable MiniMax Music 3 inference with GGML"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MINIMAX_BUILD_TESTS "Build weightless unit tests" ON)
option(MINIMAX_BUILD_PARITY_TOOLS "Build real-model parity tools" OFF)
option(MINIMAX_DYNAMIC_BACKENDS "Build GGML backends as runtime-loadable libraries" OFF)
option(MINIMAX_BUILD_NATIVE_RUNTIME "Build the audited MiniMax model graphs and converter" ON)

set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ${MINIMAX_DYNAMIC_BACKENDS} CACHE BOOL "" FORCE)
if(MINIMAX_DYNAMIC_BACKENDS)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
endif()
if(GGML_CUDA AND NOT DEFINED GGML_CUDA_NCCL)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)
find_package(Threads REQUIRED)

add_library(minimax-core STATIC
    src/minimax.cpp
    src/minimax-chunking.cpp
    src/minimax-prompt.cpp
    src/minimax-request.cpp
    src/minimax-tokenizer.cpp
    third_party/llama-unicode/unicode.cpp
    third_party/llama-unicode/unicode-data.cpp
    third_party/utf8proc/utf8proc.c
)
add_library(minimax::core ALIAS minimax-core)

target_include_directories(minimax-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-unicode
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/utf8proc
)
target_link_libraries(minimax-core PUBLIC ggml::ggml Threads::Threads)
target_compile_definitions(minimax-core PUBLIC MINIMAX_VERSION="${PROJECT_VERSION}")
target_compile_features(minimax-core PUBLIC cxx_std_17)

if(MINIMAX_BUILD_NATIVE_RUNTIME)
    add_subdirectory(third_party/audiocpp)
    target_sources(minimax-core PRIVATE src/minimax-native.cpp)
    target_link_libraries(minimax-core PUBLIC minimax::native)
    target_compile_definitions(minimax-core PUBLIC MINIMAX_HAS_NATIVE_RUNTIME=1)
else()
    target_sources(minimax-core PRIVATE src/minimax-native-stub.cpp)
endif()

if(MSVC)
    target_compile_options(minimax-core PRIVATE /W4 /utf-8)
else()
    target_compile_options(minimax-core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(minimax-cli tools/minimax-cli.cpp)
target_link_libraries(minimax-cli PRIVATE minimax::core)
set_target_properties(minimax-core minimax-cli PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(MINIMAX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
