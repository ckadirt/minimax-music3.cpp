#pragma once

#include <exception>
#include <memory>
#include <utility>

namespace minimax {

// One stage borrows a context-owned runtime. Retain weights when requested,
// but always release request graphs/KV state, including pause and error exits.
// An exceptional exit discards the runtime rather than reusing partial state.
template <class Runtime>
class runtime_lease {
public:
    template <class Factory>
    runtime_lease(std::unique_ptr<Runtime> & slot, bool keep, Factory && factory)
        : slot_(slot), keep_(keep), exceptions_(std::uncaught_exceptions()) {
        if (!slot_) slot_ = factory();
    }
    runtime_lease(const runtime_lease &) = delete;
    runtime_lease & operator=(const runtime_lease &) = delete;
    ~runtime_lease() { finish(); }
    Runtime & get() const { return *slot_; }
    void finish() {
        if (finished_) return;
        finished_ = true;
        slot_->release_runtime_graphs();
        if (!keep_ || std::uncaught_exceptions() > exceptions_) slot_.reset();
    }
private:
    std::unique_ptr<Runtime> & slot_;
    bool keep_;
    int exceptions_;
    bool finished_ = false;
};

} // namespace minimax
