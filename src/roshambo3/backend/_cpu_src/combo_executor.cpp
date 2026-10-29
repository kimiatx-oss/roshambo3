// Persistent, explicitly owned CPU worker pool for combo scoring.
//
// A fixed set of native threads waits between submissions. Each submission
// is one synchronous job over `work_items` indices: workers claim contiguous
// chunks from a single atomic cursor until it runs past the end, so load
// balances dynamically with one queue operation per chunk, never per
// alignment. There is no process-global pool and no detached work; a
// session serializes its callers, and separate executors may run
// concurrently.
//
// Correctness rules that shape the code:
//   - hot mutable per-worker state (counters, scratch headers) and the
//     shared cursor live on separate cache lines; sharing the cursor is
//     deliberate, adjacent writable worker state on one line is not;
//   - the first exception cancels further claims, every worker still
//     finishes its round, and the exception is rethrown to the caller --
//     never a partial result presented as complete;
//   - workers run without the GIL; the Python callback path reacquires it
//     only around the callback.
// OS placement (affinity, macOS QoS) goes through cpu_platform.hpp so no
// OS-specific call reaches this file (doc/APPLE_SILICON.md).
#include "combo_executor.hpp"
#include "cpu_block_prepare.hpp"
#include "r3g_codec_bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include "cpu_platform.hpp"
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace py = pybind11;

namespace r3 {
namespace cpu {
namespace {
// Alignment unit for the false-sharing separation below.
constexpr std::size_t CACHELINE_BYTES = 64;

std::vector<int> allowed_cpus() {
    return platform::allowed_cpus();
}
}  // namespace

struct ComboExecutor::Impl {
    // A type-erased per-worker object (scratch, local top-K, ...) keyed by an
    // address used as a type tag; see `get_worker_context`.
    struct ContextEntry {
        const void* type = nullptr;
        void* value = nullptr;
        void (*destroy)(void*) = nullptr;
    };

    // Everything one worker writes during a job. Cache-line aligned and
    // heap-allocated per worker, so no two workers' counters share a line.
    // `*_total` accumulate over the executor's life; `*_last` describe the
    // most recent submission.
    struct alignas(CACHELINE_BYTES) WorkerSlot {
        std::uint64_t completed_items_total = 0;
        std::uint64_t completed_items_last = 0;
        std::uint64_t claims_total = 0;
        std::uint64_t claims_last = 0;
        double elapsed_seconds = 0.0;
        ComboWorkerPlacement placement;
        std::vector<ContextEntry> contexts;
    };

    // The one intentionally shared hot word: next unclaimed item index.
    struct alignas(CACHELINE_BYTES) ClaimCursor {
        std::atomic<std::size_t> next{0};
    };

    struct alignas(CACHELINE_BYTES) Cancellation {
        std::atomic<bool> requested{false};
    };

    int workers;
    std::size_t claim_rows;
    bool require_affinity;
    std::string cpu_qos;
    std::vector<std::unique_ptr<WorkerSlot>> slots;
    std::vector<std::thread> threads;

    // Job handoff. `generation` increments once per submission; a worker
    // runs a job when it observes a generation it has not seen. `work`,
    // `work_items` and `claim_items` are written only under `state_mutex`
    // while every worker is idle.
    mutable std::mutex state_mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    std::condition_variable startup_done;
    bool stopping = false;
    std::uint64_t generation = 0;
    std::size_t started_workers = 0;
    std::exception_ptr startup_error;
    std::size_t completed_workers = 0;
    std::size_t work_items = 0;
    std::size_t claim_items = 1;
    std::function<void(std::size_t, std::size_t)> work;
    std::exception_ptr worker_error;
    std::mutex error_mutex;

    // Submission is a synchronous public contract: multiple caller threads may
    // use a session, but only one job is active on it at a time.
    mutable std::mutex call_mutex;
    mutable std::mutex submission_mutex;
    std::atomic<bool> closed{false};
    std::uint64_t submissions = 0;
    ClaimCursor cursor;
    Cancellation cancellation;

    Impl(int n_workers, std::size_t n_claim_rows, bool require_worker_affinity,
         const std::string& qos)
        : workers(n_workers), claim_rows(n_claim_rows),
          require_affinity(require_worker_affinity), cpu_qos(qos) {
        // Refuse an unknown or unsupported QoS policy before any thread
        // exists (explicit QoS is macOS-only, doc/APPLE_SILICON.md).
        platform::validate_qos(cpu_qos);
        if (workers < 1)
            throw std::invalid_argument("executor workers must be >= 1");
        // A claim covers roughly 128-512 pair rows (the Python default is
        // 256).
        if (claim_rows < 128 || claim_rows > 512 || claim_rows % 16 != 0)
            throw std::invalid_argument(
                "claim_chunk_rows must be a multiple of 16 in [128, 512]");

        const std::vector<int> cpus = allowed_cpus();
        if (require_affinity && cpus.empty())
            throw std::runtime_error(
                "required combo worker affinity has no allowed CPU mask");
        if (require_affinity && std::size_t(workers) > cpus.size())
            throw std::runtime_error(
                "required combo workers exceed the process allowed CPU mask");

        // Worker w is assigned the w-th allowed CPU, wrapping if there are
        // more workers than CPUs (only possible without strict affinity).
        // Where affinity is unsupported (macOS) the list is empty and no
        // binding is attempted.
        slots.reserve(std::size_t(workers));
        for (int worker = 0; worker < workers; ++worker) {
            auto slot = std::make_unique<WorkerSlot>();
            if (!cpus.empty()) {
                slot->placement.requested_cpu =
                    cpus[std::size_t(worker) % cpus.size()];
                slot->placement.affinity_attempted = true;
            }
            slots.push_back(std::move(slot));
        }

        threads.reserve(std::size_t(workers));
        try {
            for (int worker = 0; worker < workers; ++worker)
                threads.emplace_back(
                    [this, worker] { worker_loop(std::size_t(worker)); });
        } catch (...) {
            stop_threads();
            throw;
        }

        // Wait until every worker has configured itself, then, under strict
        // affinity, require each to read back exactly its requested CPU.
        // Any failure tears the pool down before the constructor throws.
        {
            std::unique_lock<std::mutex> lock(state_mutex);
            startup_done.wait(
                lock, [&] { return started_workers == std::size_t(workers); });
            for (const auto& slot : slots) {
                if (require_affinity &&
                    (!slot->placement.affinity_succeeded ||
                     slot->placement.actual_cpus.size() != 1 ||
                     slot->placement.actual_cpus[0] !=
                         slot->placement.requested_cpu)) {
                    if (!startup_error)
                        startup_error = std::make_exception_ptr(std::runtime_error(
                            "required combo worker CPU affinity could not be verified"));
                    break;
                }
            }
        }
        if (startup_error) {
            stop_threads();
            std::rethrow_exception(startup_error);
        }
    }

    ~Impl() {
        close();
    }

    // Runs once on the worker's own thread: QoS and affinity are per-thread
    // properties, so they must be set from inside it. Requested and observed
    // values are recorded for `stats["worker_placement"]`; per
    // doc/APPLE_SILICON.md they show the scheduling class requested, not the
    // core a worker actually ran on.
    void configure_worker(std::size_t worker) {
        WorkerSlot& slot = *slots[worker];
        platform::set_qos(cpu_qos);
        slot.placement.qos_requested = cpu_qos;
        slot.placement.qos_observed = platform::current_qos();
        if (slot.placement.requested_cpu >= 0)
            slot.placement.affinity_succeeded =
                platform::bind_current_thread(slot.placement.requested_cpu);
        if (slot.placement.affinity_attempted || require_affinity)
            slot.placement.actual_cpus = platform::allowed_cpus();
    }

    // Worker body. `noexcept`: every failure is captured as an
    // exception_ptr and handed to the constructor or to `run`, never allowed
    // to escape a std::thread (which would terminate the process).
    void worker_loop(std::size_t worker) noexcept {
        std::exception_ptr setup_failure;
        try {
            configure_worker(worker);
        } catch (...) {
            setup_failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            if (setup_failure && !startup_error) startup_error = setup_failure;
            ++started_workers;
        }
        startup_done.notify_one();

        std::uint64_t observed_generation = 0;
        for (;;) {
            std::size_t item_count = 0;
            std::size_t item_chunk = 1;
            const std::function<void(std::size_t, std::size_t)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(state_mutex);
                work_ready.wait(lock, [&] {
                    return stopping || generation != observed_generation;
                });
                if (stopping) return;
                observed_generation = generation;
                item_count = work_items;
                item_chunk = claim_items;
                job = &work;
            }

            WorkerSlot& slot = *slots[worker];
            slot.completed_items_last = 0;
            slot.claims_last = 0;
            const auto started = std::chrono::steady_clock::now();
            // Claim loop. Relaxed atomics suffice: the cursor only hands out
            // disjoint index ranges, and the job's inputs were published
            // under `state_mutex` before the generation changed.
            while (!cancellation.requested.load(std::memory_order_relaxed)) {
                const std::size_t first = cursor.next.fetch_add(
                    item_chunk, std::memory_order_relaxed);
                if (first >= item_count) break;
                ++slot.claims_last;
                ++slot.claims_total;
                // The final short claim keeps its valid tail.
                const std::size_t stop = item_count - first < item_chunk
                                             ? item_count
                                             : first + item_chunk;
                for (std::size_t item = first; item < stop; ++item) {
                    if (cancellation.requested.load(std::memory_order_relaxed))
                        break;
                    try {
                        (*job)(worker, item);
                        ++slot.completed_items_last;
                        ++slot.completed_items_total;
                    } catch (...) {
                        // First error wins; cancellation stops other
                        // workers at their next item boundary.
                        {
                            std::lock_guard<std::mutex> error_lock(error_mutex);
                            if (!worker_error)
                                worker_error = std::current_exception();
                        }
                        cancellation.requested.store(true,
                                                     std::memory_order_relaxed);
                        break;
                    }
                }
            }
            slot.elapsed_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              started)
                    .count();

            // Every worker reports completion, including after cancellation,
            // so `run` always wakes.
            bool notify_done = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                ++completed_workers;
                notify_done = completed_workers == std::size_t(workers);
            }
            if (notify_done) work_done.notify_one();
        }
    }

    // One synchronous submission. `submission_mutex` is held for the whole
    // job, which is also what keeps `stats()` (which takes the same mutex)
    // from reading worker counters while workers write them.
    void run(std::size_t item_count, std::size_t rows_per_item,
             const std::function<void(std::size_t, std::size_t)>& callback) {
        if (rows_per_item == 0)
            throw std::invalid_argument("rows_per_item must be >= 1");
        std::unique_lock<std::mutex> submit_lock(submission_mutex);
        if (closed.load(std::memory_order_acquire))
            throw std::runtime_error("combo executor is closed");
        if (!callback)
            throw std::invalid_argument(
                "combo executor work callback is empty");
        ++submissions;
        for (auto& slot : slots) {
            slot->completed_items_last = 0;
            slot->claims_last = 0;
            slot->elapsed_seconds = 0.0;
        }
        if (item_count == 0) return;

        // Items per claim: as many whole items as fit in `claim_rows` rows,
        // and at least one.
        const std::size_t chunk_items =
            std::max<std::size_t>(1, claim_rows / rows_per_item);
        cursor.next.store(0, std::memory_order_relaxed);
        cancellation.requested.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> error_lock(error_mutex);
            worker_error = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            work_items = item_count;
            claim_items = chunk_items;
            work = callback;
            completed_workers = 0;
            ++generation;
        }
        work_ready.notify_all();

        std::exception_ptr failure;
        {
            std::unique_lock<std::mutex> lock(state_mutex);
            work_done.wait(lock, [&] {
                return completed_workers == std::size_t(workers);
            });
            std::lock_guard<std::mutex> error_lock(error_mutex);
            failure = worker_error;
            work = {};
        }
        if (failure) std::rethrow_exception(failure);
    }

    // Idempotent. Joins the workers, then destroys worker contexts; taking
    // `submission_mutex` first means an active job finishes before teardown.
    void close() {
        std::unique_lock<std::mutex> submit_lock(submission_mutex);
        if (closed.exchange(true, std::memory_order_acq_rel)) return;
        stop_threads();
        for (auto& slot : slots) {
            for (auto& context : slot->contexts)
                if (context.value && context.destroy)
                    context.destroy(context.value);
            slot->contexts.clear();
        }
    }

    void stop_threads() noexcept {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (std::thread& thread : threads)
            if (thread.joinable()) thread.join();
        threads.clear();
    }

    // Find or lazily create this worker's context of the given type. Called
    // from the owning worker during a job, so creation (and first touch of
    // large scratch) happens on that worker's thread -- the first-touch
    // rule. Contexts live until close().
    void* get_worker_context(std::size_t worker, const void* type,
                             const std::function<void*()>& create,
                             void (*destroy)(void*)) {
        if (worker >= slots.size())
            throw std::out_of_range("combo worker id is out of range");
        if (closed.load(std::memory_order_acquire))
            throw std::runtime_error("combo executor is closed");
        WorkerSlot& slot = *slots[worker];
        for (ContextEntry& context : slot.contexts)
            if (context.type == type) return context.value;
        ContextEntry context;
        context.type = type;
        context.value = create();
        context.destroy = destroy;
        try {
            slot.contexts.push_back(context);
        } catch (...) {
            destroy(context.value);
            throw;
        }
        return context.value;
    }
};

ComboExecutor::ComboExecutor(int workers, std::size_t claim_chunk_rows,
                             bool require_worker_affinity,
                             const std::string& cpu_qos)
    : impl_(std::make_unique<Impl>(workers, claim_chunk_rows,
                                   require_worker_affinity, cpu_qos)) {}

ComboExecutor::~ComboExecutor() = default;

void ComboExecutor::run(
    std::size_t work_items, std::size_t rows_per_item,
    const std::function<void(std::size_t, std::size_t)>& work) {
    impl_->run(work_items, rows_per_item, work);
}

std::unique_lock<std::mutex> ComboExecutor::acquire_call() {
    return std::unique_lock<std::mutex>(impl_->call_mutex);
}

void ComboExecutor::close() {
    // Lock order is call -> submission everywhere that both are needed.
    // Waiting for a scoring call must not tear down worker contexts mid-job.
    std::unique_lock<std::mutex> call_lock(impl_->call_mutex);
    impl_->close();
}

std::size_t ComboExecutor::worker_count() const noexcept {
    return std::size_t(impl_->workers);
}

std::size_t ComboExecutor::claim_chunk_rows() const noexcept {
    return impl_->claim_rows;
}

bool ComboExecutor::require_worker_affinity() const noexcept {
    return impl_->require_affinity;
}

bool ComboExecutor::is_closed() const noexcept {
    return impl_->closed.load(std::memory_order_acquire);
}

ComboWorkerPlacement
ComboExecutor::worker_placement(std::size_t worker_id) const {
    std::lock_guard<std::mutex> lock(impl_->submission_mutex);
    if (worker_id >= impl_->slots.size())
        throw std::out_of_range("combo worker id is out of range");
    return impl_->slots[worker_id]->placement;
}

double ComboExecutor::worker_elapsed_seconds(std::size_t worker_id) const {
    std::lock_guard<std::mutex> lock(impl_->submission_mutex);
    if (worker_id >= impl_->slots.size())
        throw std::out_of_range("combo worker id is out of range");
    return impl_->slots[worker_id]->elapsed_seconds;
}

ComboExecutorStats ComboExecutor::stats() const {
    std::lock_guard<std::mutex> lock(impl_->submission_mutex);
    ComboExecutorStats result;
    result.submissions = impl_->submissions;
    result.work_items_completed_by_worker.reserve(impl_->slots.size());
    result.claims_by_worker.reserve(impl_->slots.size());
    result.last_claims_by_worker.reserve(impl_->slots.size());
    result.last_worker_elapsed_seconds.reserve(impl_->slots.size());
    for (const auto& slot : impl_->slots) {
        result.work_items_completed += slot->completed_items_total;
        result.work_items_completed_by_worker.push_back(
            slot->completed_items_total);
        result.claims_by_worker.push_back(slot->claims_total);
        result.last_claims_by_worker.push_back(slot->claims_last);
        result.last_worker_elapsed_seconds.push_back(slot->elapsed_seconds);
    }
    return result;
}

void* ComboExecutor::get_worker_context(std::size_t worker_id, const void* type,
                                        const std::function<void*()>& create,
                                        void (*destroy)(void*)) {
    return impl_->get_worker_context(worker_id, type, create, destroy);
}

void bind_combo_executor_api(py::module_& module) {
    bind_cpu_block_prepare_api(module);
    r3g::bind_r3g_codec_api(module);  // format-3 codec
    py::class_<ComboRowSink, std::shared_ptr<ComboRowSink>>(module,
                                                            "_ComboRowSink");

    py::class_<ComboExecutor, std::shared_ptr<ComboExecutor>>(
        module, "_CPUComboExecutor")
        .def(py::init([](int workers, std::size_t claim_chunk_rows,
                         bool require_worker_affinity,
                         const std::string& cpu_qos) {
                 std::shared_ptr<ComboExecutor> executor;
                 {
                     py::gil_scoped_release nogil;
                     executor = std::make_shared<ComboExecutor>(
                         workers, claim_chunk_rows, require_worker_affinity,
                         cpu_qos);
                 }
                 return executor;
             }),
             py::arg("workers"), py::arg("claim_chunk_rows") = 256,
             py::arg("require_worker_affinity") = false,
             py::arg("cpu_qos") = "inherit")
        .def_property_readonly("threads", &ComboExecutor::worker_count)
        .def_property_readonly("claim_chunk_rows",
                               &ComboExecutor::claim_chunk_rows)
        .def_property_readonly("require_worker_affinity",
                               &ComboExecutor::require_worker_affinity)
        .def_property_readonly("closed", &ComboExecutor::is_closed)
        .def_property_readonly(
            "stats",
            [](const ComboExecutor& self) {
                ComboExecutorStats value;
                std::vector<ComboWorkerPlacement> worker_placements;
                {
                    py::gil_scoped_release nogil;
                    value = self.stats();
                    for (std::size_t worker = 0; worker < self.worker_count();
                         ++worker)
                        worker_placements.push_back(
                            self.worker_placement(worker));
                }
                py::dict result;
                result["submissions"] = value.submissions;
                result["work_items_completed"] = value.work_items_completed;
                result["work_items_completed_by_worker"] =
                    value.work_items_completed_by_worker;
                result["claims_by_worker"] = value.claims_by_worker;
                result["last_claims_by_worker"] = value.last_claims_by_worker;
                result["last_worker_elapsed_seconds"] =
                    value.last_worker_elapsed_seconds;
                py::list placements;
                for (const auto& placement : worker_placements) {
                    py::dict item;
                    item["requested_cpu"] = placement.requested_cpu;
                    item["affinity_attempted"] = placement.affinity_attempted;
                    item["affinity_succeeded"] = placement.affinity_succeeded;
                    item["actual_cpus"] = placement.actual_cpus;
                    item["qos_requested"] = placement.qos_requested;
                    item["qos_observed"] = placement.qos_observed;
                    placements.append(item);
                }
                result["worker_placement"] = placements;
                return result;
            })
        // Backs `CPUComboExecutor.run_indexed` in cpu.py: callback(worker_id,
        // item_id) on the persistent workers, e.g. a block decode followed
        // by a native scorer call.
        .def(
            "run_python_items",
            [](ComboExecutor& self, py::function callback,
               std::size_t work_items, std::size_t rows_per_item) {
                if (!callback || !PyCallable_Check(callback.ptr()))
                    throw std::invalid_argument(
                        "combo executor callback must be callable");
                std::unique_lock<std::mutex> call_lock;
                {
                    py::gil_scoped_release nogil;
                    call_lock = self.acquire_call();
                }
                if (self.is_closed())
                    throw std::runtime_error("combo executor is closed");
                // The caller owns callback for the whole synchronous submission.
                // Workers acquire the GIL only around the codec/user callback;
                // its result and any temporary py::object are destroyed before
                // releasing it. Convert Python failures to std::runtime_error
                // while still holding the GIL so a Python exception object is
                // never destroyed on a GIL-free worker.
                const std::function<void(std::size_t, std::size_t)> work =
                    [&callback](std::size_t worker, std::size_t item) {
                        py::gil_scoped_acquire gil;
                        try {
                            {
                                py::object ignored = callback(worker, item);
                            }
                        } catch (py::error_already_set& error) {
                            const std::string message = error.what();
                            throw std::runtime_error(
                                "Python combo work callback failed: " +
                                message);
                        }
                    };
                py::gil_scoped_release nogil;
                self.run(work_items, rows_per_item, work);
            },
            py::arg("callback"), py::arg("work_items"),
            py::arg("rows_per_item") = 1,
            "Synchronously run Python work on persistent native workers. "
            "The waiting caller releases the GIL; each worker reacquires it "
            "only while invoking the callback.")
        .def("close",
             [](ComboExecutor& self) {
                 py::gil_scoped_release nogil;
                 self.close();
             })
        .def("__enter__",
             [](std::shared_ptr<ComboExecutor> self) {
                 if (self->is_closed())
                     throw std::runtime_error("combo executor is closed");
                 return self;
             })
        .def("__exit__",
             [](ComboExecutor& self, py::object, py::object, py::object) {
                 py::gil_scoped_release nogil;
                 self.close();
             });
}

}
}  // namespace r3::cpu
