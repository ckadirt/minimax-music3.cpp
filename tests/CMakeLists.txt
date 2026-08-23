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

add_executable(test-request test-request.cpp)
target_link_libraries(test-request PRIVATE minimax::core)
add_test(NAME request-contract COMMAND test-request)

if(MINIMAX_BUILD_NATIVE_RUNTIME)
    add_executable(test-seed test-seed.cpp)
    target_link_libraries(test-seed PRIVATE minimax::native)
    add_test(NAME seed-contract COMMAND test-seed)
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME source-inventory
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_source_inventory.py
                ${PROJECT_SOURCE_DIR})
    add_test(NAME conversion-policy
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/test_conversion_policy.py
                ${PROJECT_SOURCE_DIR})
endif()
