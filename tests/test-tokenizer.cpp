#include "minimax-tokenizer.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#ifndef MINIMAX_TEST_FIXTURES_DIR
#define MINIMAX_TEST_FIXTURES_DIR "tests/fixtures"
#endif

int main() {
    const std::string root = MINIMAX_TEST_FIXTURES_DIR;
    auto tokenizer = minimax::ByteLevelBPETokenizer::load(
        root + "/tiny-vocab.json", root + "/tiny-merges.txt",
        root + "/tiny-tokenizer-config.json");
    tokenizer.add_special_token("[verse]", 20);
    assert(tokenizer.encode("hello world") == std::vector<std::int64_t>({15, 16}));
    assert(tokenizer.encode("a  b!") == std::vector<std::int64_t>({1, 13, 13, 2, 0}));
    assert(tokenizer.encode("<|im_start|>[verse]hello") ==
           std::vector<std::int64_t>({19, 20, 15}));
    assert(tokenizer.decode(tokenizer.encode("hello world")) == "hello world");
}
