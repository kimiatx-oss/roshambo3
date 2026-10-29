#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

namespace r3 {
namespace cpu {

struct ComboWorkerPlacement {
    int requested_cpu = -1;
    bool affinity_attempted = false;
    bool affinity_succeeded = false;
    std::vector<int> actual_cpus;
    std::string qos_requested = "inherit";
    std::string qos_observed;
};

struct ComboExecutorStats {
    std::uint64_t submissions = 0;
    std::uint64_t work_items_completed = 0;
    std::vector<std::uint64_t> work_items_completed_by_worker;
    std::vector<std::uint64_t> claims_by_worker;
    std::vector<std::uint64_t> last_claims_by_worker;
    std::vector<double> last_worker_elapsed_seconds;
};

// Native-only observer for a completely finalized score row. Implementations
// may own row-to-ID maps or borrow validated immutable views, but any borrowed
// owners must remain strongly held through the synchronous score call and the
// sink's finalize/abort decision; the views may not be mutated concurrently.
// Hooks run on native workers and must not call into Python or close/re-enter
// their executor.
class ComboRowSink {
public:
    virtual ~ComboRowSink() = default;
    virtual void on_row(std::size_t worker_id, std::size_t output_row,
                        std::size_t query_row, std::size_t library_row,
                        const float* scores, std::size_t n_score_columns) = 0;
};

// An explicitly owned synchronous executor. Its native threads wait between
// submissions and claim bounded chunks from one cursor; no process-global pool
// or detached work exists. A session serializes concurrent submitters.
class ComboExecutor final {
public:
    ComboExecutor(int workers, std::size_t claim_chunk_rows,
                  bool require_worker_affinity,
                  const std::string& cpu_qos = "inherit");
    ~ComboExecutor();

    ComboExecutor(const ComboExecutor&) = delete;
    ComboExecutor& operator=(const ComboExecutor&) = delete;

    void run(std::size_t work_items, std::size_t rows_per_item,
             const std::function<void(std::size_t, std::size_t)>& work);
    std::unique_lock<std::mutex> acquire_call();
    void close();

    std::size_t worker_count() const noexcept;
    std::size_t claim_chunk_rows() const noexcept;
    bool require_worker_affinity() const noexcept;
    bool is_closed() const noexcept;
    ComboWorkerPlacement worker_placement(std::size_t worker_id) const;
    double worker_elapsed_seconds(std::size_t worker_id) const;
    ComboExecutorStats stats() const;

    // Worker contexts are constructed on first use by their owning native
    // worker and live until close/destruction. They are intended for reusable
    // scratch and per-worker counters, not shared result state.
    template <typename T>
    T& worker_context(std::size_t worker_id) {
        return *static_cast<T*>(get_worker_context(
            worker_id, type_token<T>(), []() -> void* { return new T(); },
            [](void* value) { delete static_cast<T*>(value); }));
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void* get_worker_context(std::size_t worker_id, const void* type,
                             const std::function<void*()>& create,
                             void (*destroy)(void*));

    template <typename T>
    static const void* type_token() noexcept {
        static const unsigned char token = 0;
        return &token;
    }
};

void bind_combo_executor_api(pybind11::module_& module);

}
}  // namespace r3::cpu
