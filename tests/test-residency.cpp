#include "cantor-residency.h"

#include <iostream>
#include <stdexcept>

struct Runtime {
    int & destroyed;
    int & released;
    Runtime(int & d, int & r) : destroyed(d), released(r) {}
    ~Runtime() { ++destroyed; }
    void release_runtime_graphs() { ++released; }
};
void check(bool value) { if (!value) throw std::runtime_error("residency assertion failed"); }
int main() {
    try {
        int created = 0, destroyed = 0, released = 0;
        std::unique_ptr<Runtime> slot;
        auto factory = [&] { ++created; return std::make_unique<Runtime>(destroyed, released); };
        for (int i = 0; i < 2; ++i) {
            minimax::runtime_lease<Runtime> lease(slot, true, factory);
        }
        check(created == 1 && destroyed == 0 && released == 2);
        // Explicit finish (including a pause) is idempotent and drops graphs.
        {
            minimax::runtime_lease<Runtime> lease(slot, true, factory);
            lease.finish();
            lease.finish();
        }
        check(created == 1 && released == 3);
        // Errors invalidate partial runtime state even when retaining weights.
        try {
            minimax::runtime_lease<Runtime> lease(slot, true, factory);
            throw std::runtime_error("stage failed");
        } catch (const std::runtime_error &) {}
        check(!slot && destroyed == 1 && released == 4);
        for (int i = 0; i < 2; ++i) {
            minimax::runtime_lease<Runtime> lease(slot, false, factory);
        }
        check(!slot && created == 3 && destroyed == 3 && released == 6);
        {
            minimax::runtime_lease<Runtime> lease(slot, true, factory);
        }
        slot.reset();
        check(created == 4 && destroyed == 4);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
