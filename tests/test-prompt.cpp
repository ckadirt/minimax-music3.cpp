#include "minimax-prompt.h"

#include <cassert>
#include <string>

int main() {
    assert(minimax::clean_caption("# Global\n- Warm **pop**\n---\n<|bpm 92|>") ==
           "Global\nWarm pop\nbpm is 92");
    assert(minimax::normalize_lyrics("[Verse]\nHello\n[CHORUS]\nWorld") ==
           "[start]\n[verse]\nHello\n[chorus]\nWorld");
    // The official contract retains leading tags and drops text on the same line.
    assert(minimax::normalize_lyrics("[Verse] This is dropped") == "[start]\n[verse]");
    assert(minimax::normalize_lyrics("[Verse] [Chorus]\nLine") ==
           "[start]\n[verse]\n[chorus]\nLine");
}
