#include "cantor_engine.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int main(void) {
    assert(cantor_engine_abi_version() == CANTOR_ENGINE_ABI);
    assert(strcmp(cantor_engine_model(), "minimax-music3") == 0);
    assert(cantor_engine_stages() ==
           ((1U << CANTOR_STAGE_CODES) | (1U << CANTOR_STAGE_DIFFUSE) |
            (1U << CANTOR_STAGE_DECODE)));

    const cantor_component components[] = {
        {"lm", "lm.gguf"},
        {"rvq", "rvq.gguf"},
        {"condition", "condition.gguf"},
        {"dit", "dit.gguf"},
        {"vae", "vae.gguf"},
    };
    cantor_ctx * context = cantor_engine_load(components, 5, 0);
    assert(context != 0);
    const uint8_t request[] = {'{', '}'};
    uint8_t * output = (uint8_t *)(uintptr_t)1;
    size_t output_size = 99;
    assert(cantor_engine_run_stage(
               context, CANTOR_STAGE_CODES, request, sizeof(request),
               &output, &output_size, 0, 0, 0) == CANTOR_ERR);
    assert(output == 0 && output_size == 0);
    assert(cantor_engine_last_error_code() == CANTOR_ERR_OTHER);
    assert(strstr(cantor_engine_last_error(), "required field") != 0);
    assert(cantor_engine_resident_modules(context) == 0);
    cantor_engine_free(context);

    const cantor_component unknown = {"unknown", "x.gguf"};
    assert(cantor_engine_load(&unknown, 1, 0) == 0);
    assert(cantor_engine_last_error_code() == CANTOR_ERR_OTHER);
    return 0;
}
