cmake_minimum_required(VERSION 3.20)

project(minimax-music3
    VERSION 0.1.0
    DESCRIPTION "Portable MiniMax Music 3 inference with GGML"
    LANGUAGES C CXX)

include(GNUInstallDirs)

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

# Android's NDK provides the Vulkan loader and C headers but not vulkan.hpp.
# GGML's Vulkan backend uses Vulkan-Hpp, so cross-builds take that header from
# the pinned SDK installed on the host while still linking the NDK loader.
if(ANDROID AND GGML_VULKAN)
    find_path(MINIMAX_VULKAN_HPP_INCLUDE_DIR
        NAMES vulkan/vulkan.hpp
        HINTS "$ENV{VULKAN_SDK}/include"
        NO_DEFAULT_PATH)
    if(NOT MINIMAX_VULKAN_HPP_INCLUDE_DIR)
        message(FATAL_ERROR "Android Vulkan build requires VULKAN_SDK/include/vulkan/vulkan.hpp")
    endif()
    target_include_directories(ggml-vulkan PRIVATE "${MINIMAX_VULKAN_HPP_INCLUDE_DIR}")
endif()

add_library(minimax-core STATIC
    src/minimax-checkpoint.cpp
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
target_compile_definitions(minimax-core PRIVATE UTF8PROC_STATIC)
target_compile_features(minimax-core PUBLIC cxx_std_17)

if(MINIMAX_BUILD_NATIVE_RUNTIME)
    add_subdirectory(third_party/audiocpp)
    target_sources(minimax-core PRIVATE src/minimax-native.cpp)
    target_link_libraries(minimax-core PUBLIC minimax::native)
    target_compile_definitions(minimax-core PUBLIC MINIMAX_HAS_NATIVE_RUNTIME=1)
else()
    target_sources(minimax-core PRIVATE src/minimax-native-stub.cpp)
endif()

if(MINIMAX_BUILD_NATIVE_RUNTIME)
    add_library(cantor-engine SHARED src/cantor-abi.cpp)
    target_link_libraries(cantor-engine PRIVATE minimax::core)
    target_include_directories(cantor-engine
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(cantor-engine PRIVATE CANTOR_ENGINE_BUILD=1)
    target_compile_features(cantor-engine PRIVATE cxx_std_17)
    if(APPLE)
        target_link_options(cantor-engine PRIVATE
            "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cantor_engine.exports")
    elseif(UNIX)
        target_link_options(cantor-engine PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cantor_engine.map")
    endif()
    set_target_properties(cantor-engine PROPERTIES
        OUTPUT_NAME cantor_engine
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    if(APPLE)
        set_target_properties(cantor-engine PROPERTIES INSTALL_RPATH "@loader_path")
    elseif(UNIX)
        set_target_properties(cantor-engine PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
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
if(APPLE)
    set_target_properties(minimax-cli PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_target_properties(minimax-cli PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if(MINIMAX_BUILD_PARITY_TOOLS)
    if(NOT MINIMAX_BUILD_NATIVE_RUNTIME)
        message(FATAL_ERROR "MINIMAX_BUILD_PARITY_TOOLS requires MINIMAX_BUILD_NATIVE_RUNTIME")
    endif()
    add_executable(minimax-parity tools/minimax-parity.cpp)
    target_link_libraries(minimax-parity PRIVATE minimax::native)
    set_target_properties(minimax-parity PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    add_executable(minimax-cantor-smoke tools/minimax-cantor-smoke.cpp)
    target_link_libraries(minimax-cantor-smoke PRIVATE cantor-engine minimax::core)
    target_include_directories(minimax-cantor-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(minimax-cantor-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

install(TARGETS minimax-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MINIMAX_BUILD_NATIVE_RUNTIME)
    install(TARGETS cantor-engine
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(FILES include/cantor_engine.h include/minimax.h include/minimax-request.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE THIRD_PARTY_NOTICES.md README.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/minimax-music3)

if(MINIMAX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
