add_executable(test-backend test-backend.cpp)
target_link_libraries(test-backend PRIVATE minimax::core)
add_test(NAME backend-smoke COMMAND test-backend)

add_executable(test-chunking test-chunking.cpp)
target_link_libraries(test-chunking PRIVATE minimax::core)
target_include_directories(test-chunking PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME chunking-contract COMMAND test-chunking)

add_executable(test-prompt test-prompt.cpp)
target_link_libraries(test-prompt PRIVATE minimax::core)
target_include_directories(test-prompt PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME prompt-normalization COMMAND test-prompt)

add_executable(test-tokenizer test-tokenizer.cpp)
target_link_libraries(test-tokenizer PRIVATE minimax::core)
target_include_directories(test-tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(test-tokenizer PRIVATE
    MINIMAX_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME tokenizer-contract COMMAND test-tokenizer)
