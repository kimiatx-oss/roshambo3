// MIT License
//
// Copyright (c) 2026 Kimia Therapeutics
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

////////////////////////////////////////////////////////////////////////////////
/// The Python boundary: one scoring entry point, the device arena, the stage
/// timers, pinned host staging, and the device-resident top-K.
///
/// What this file is NOT, and ROSHAMBO2's equivalent was:
///
///  * a method dispatcher. ROSHAMBO2's `optimize_overlap_color_sparse` took 11
///    method codes and ~20 measurement-arm switches, because every experiment
///    added one to the shipping signature and none were ever removed. Two
///    kernels ship here, selected by `pose_search`, and the arms live behind
///    `roshambo3.experimental`.
///
///  * a multi-GPU splitter. ROSHAMBO2 partitioned work inside C++ under OpenMP,
///    in two places that disagreed with each other, and had to forbid the device
///    top-K on more than one GPU because its state is per-device. Here the split
///    is the engine's: one call, one device, one top-K state per device, merged
///    on the host with `numerics.rank_order`. Top-N lists were measured to merge
///    correctly even across heterogeneous hardware.
///
/// OpenMP is therefore gone, and with it `OmpErrorBox` -- exceptions no longer
/// have to be marshalled across a parallel region.
////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <channel_descriptor.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "device_guard.cuh"
#include "kernels.cuh"
#include "molecule_topk.cuh"
#include "topk.cuh"
#include "dense_rows.cuh"

namespace py = pybind11;

// CUDA_CHECK_ERROR comes from kernels.cuh; do not redefine it here.

////////////////////////////////////////////////////////////////////////////////
/// Owning wrapper for a device top-K state, so Python holds a lifetime rather
/// than a raw pointer. See topk.cuh for the algorithm and the determinism proof.
////////////////////////////////////////////////////////////////////////////////
struct TopKHandle {
    DeviceTopKState * p = nullptr;
    int device_id = 0;

    TopKHandle(int n_query, int k, int rank_col, int dv, bool keep_cols, int dev)
        : device_id(dev) {
        p = topk_create(n_query, k, rank_col, dv, keep_cols, dev);
        if (!p) throw std::runtime_error("roshambo3.cuda: top-K allocation failed");
    }
    ~TopKHandle(){ if (p) topk_destroy(p); }
    TopKHandle(const TopKHandle &) = delete;
    TopKHandle & operator=(const TopKHandle &) = delete;
};

////////////////////////////////////////////////////////////////////////////////
/// Owning wrapper for a molecule-level top-K. It owns the conformer-level state
/// it feeds, so `.inner_handle` is for readback only and must not be handed to
/// optimize_pairs(topk=...) -- that would fold the same block in twice, once as
/// conformers and once as molecules. See molecule_topk.cuh.
////////////////////////////////////////////////////////////////////////////////
struct MoleculeTopKHandle {
    MoleculeTopKState * p = nullptr;
    int device_id = 0;

    MoleculeTopKHandle(int n_query, int k, int rank_col, int dv, bool keep_cols,
                       int dev, long max_block_conformers)
        : device_id(dev) {
        p = moltopk_create(n_query, k, rank_col, dv, keep_cols, dev,
                           max_block_conformers);
        if (!p) throw std::runtime_error(
            "roshambo3.cuda: molecule-level top-K allocation failed");
    }
    ~MoleculeTopKHandle(){ if (p) moltopk_destroy(p); }
    MoleculeTopKHandle(const MoleculeTopKHandle &) = delete;
    MoleculeTopKHandle & operator=(const MoleculeTopKHandle &) = delete;
};

////////////////////////////////////////////////////////////////////////////////
/// Stage accounting and the persistent device arena.
///
/// ROSHAMBO2 serialized both with `#pragma omp critical`. OpenMP has no other
/// use here now, so they take ordinary mutexes -- still needed, because the
/// engine calls into this module from a reduction thread while producer threads
/// allocate pinned buffers.
////////////////////////////////////////////////////////////////////////////////
namespace runtime {

struct StageTimers {
    double alloc_s      = 0.0;
    double h2d_s        = 0.0;
    double kernel_s     = 0.0;   // the optimizer launchers
    double precompute_s = 0.0;   // self-overlaps, the query prescale, the field build
    double d2h_s        = 0.0;
    double free_s       = 0.0;
    double index_s      = 0.0;   // host-side work-item construction
    double topk_s       = 0.0;   // device-side output reduction
    long long h2d_bytes = 0;
    long long d2h_bytes = 0;
    long long n_calls   = 0;
    // Self-overlap work counters, so "no library self-overlap was
    // evaluated" is provable on the GPU rather than inferred from precompute_s.
    // An evaluation is one conformer's (shape, color) self-overlap pair, as
    // `compute_self_overlaps_gpu` writes them. `library_self_supplied_calls`
    // counts calls whose library self-overlaps came from the caller instead;
    // no path supplies them yet, so it stays 0 (the field is the consumer's to
    // set). The bound scan's rows and flags are summed over calls that asked for
    // it (`check_self_overlap_bound`); `self_bound_s` is its time.
    long long query_self_evaluations      = 0;
    long long library_self_evaluations    = 0;
    long long library_self_supplied_calls = 0;
    long long self_bound_rows_checked     = 0;
    long long self_bound_rows_flagged     = 0;
    double    self_bound_s                = 0.0;
};

/// The last pair-kernel launch's occupancy and shared request, plus the
/// envelope over every launch since the last reset. A screen covers many
/// buckets and therefore many shared demands, so "the" blocks per SM is a
/// range; the min is the one that binds and the record carries both.
struct LaunchFacts {
    long long blocks_per_sm = -1;
    long long shared_bytes  = -1;
    long long carveout      = -1;
    long long nthreads      = -1;
    long long blocks_min    = -1;
    long long blocks_max    = -1;
    long long shared_min    = -1;
    long long shared_max    = -1;
    long long n_launches    = 0;
};

static StageTimers g_stages;
static LaunchFacts g_launch;
static bool g_enabled = false;
static std::mutex g_stage_mu;
static std::mutex g_arena_mu;

static inline double now_s(){
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-9;
}

/// Per-thread stack of the slots a Section is currently timing.
///
/// The arena's own `dfree()` and `arena_trim()` fallbacks call `cudaFree`
/// directly, and every *caller* of `dfree` already wraps it in a `free_s`
/// Section -- so a naive wrap would count the same `cudaFree` twice into the
/// same bucket. Section is therefore idempotent per slot: if one for the same
/// field is already live on this thread, the inner one measures nothing. Two
/// *different* fields still nest legitimately (the field build times its H2D
/// inside `precompute_s`), and only same-slot nesting collapses.
static const int SECTION_MAX_DEPTH = 8;
static thread_local const double * g_active[SECTION_MAX_DEPTH] = {nullptr};
static thread_local int g_active_n = 0;

struct Section {
    double * slot; double t0; bool on;
    explicit Section(double * s) : slot(s), t0(0.0), on(g_enabled) {
        if (on){
            for (int i = 0; i < g_active_n; ++i){
                if (g_active[i] == s){ on = false; return; }
            }
            if (g_active_n < SECTION_MAX_DEPTH) g_active[g_active_n++] = s;
            t0 = now_s();
        }
    }
    ~Section(){
        if (!on) return;
        double dt = now_s() - t0;
        if (g_active_n > 0 && g_active[g_active_n-1] == slot) --g_active_n;
        std::lock_guard<std::mutex> _lk(g_stage_mu);
        *slot += dt;
    }
};

static inline void add_bytes(long long h2d, long long d2h){
    if (!g_enabled) return;
    std::lock_guard<std::mutex> _lk(g_stage_mu);
    g_stages.h2d_bytes += h2d; g_stages.d2h_bytes += d2h;
}

static inline void count_call(){
    if (!g_enabled) return;
    std::lock_guard<std::mutex> _lk(g_stage_mu);
    g_stages.n_calls += 1;
}

/// One call's self-overlap work and bound-scan result, folded into the running
/// totals. Gated on `g_enabled` like every other stage counter; the per-call
/// values reach the caller through `optimize_pairs(call_stats=...)` whether or
/// not stage accounting is on.
static inline void add_self_work(long long query_self, long long library_self,
                                 bool library_supplied, long long bound_checked,
                                 long long bound_flagged){
    if (!g_enabled) return;
    std::lock_guard<std::mutex> _lk(g_stage_mu);
    g_stages.query_self_evaluations += query_self;
    g_stages.library_self_evaluations += library_self;
    g_stages.library_self_supplied_calls += library_supplied ? 1 : 0;
    if (bound_checked >= 0) g_stages.self_bound_rows_checked += bound_checked;
    if (bound_flagged >= 0) g_stages.self_bound_rows_flagged += bound_flagged;
}

////////////////////////////////////////////////////////////////////////////////
/// The persistent device arena.
///
/// Every launch used to `cudaMalloc` its fifteen buffers and `cudaFree` them
/// again. A screen is thousands of launches over the same shapes, so those are
/// thousands of allocations of a handful of distinct sizes -- and `cudaFree`
/// synchronizes the device, which on a streamed pipeline is not merely slow but
/// a correctness-shaped hazard: it serializes everything the streams were
/// arranged to overlap.
///
/// A caching allocator replaces both. Sizes round up to a power of two above
/// 1 MB so a block whose conformer count wobbles between batches still reuses
/// the same allocation -- exact-size bins miss on the short final block of
/// every group and quietly allocate a fresh buffer each time.
///
/// It caches rather than bump-allocates on purpose: a bump arena needs the peak
/// footprint known before the first allocation, which this entry point cannot
/// know until it has counted its work items.
////////////////////////////////////////////////////////////////////////////////

#define ROSHAMBO3_MAX_DEVICES 16

struct DeviceCache {
    std::vector<std::pair<size_t, void*>> free_blocks;
    /// Every live pointer handed out, and the size class it came from.
    /// `cudaFree` needs no size, so the call sites do not carry one.
    std::unordered_map<void*, size_t> live;
    size_t held_bytes = 0;
    long long hits = 0, misses = 0;
};

static DeviceCache g_cache[ROSHAMBO3_MAX_DEVICES];
static bool g_arena_enabled = true;

static inline size_t size_class(size_t n){
    const size_t SMALL = 1u << 20;
    if (n <= SMALL) return n;
    size_t c = SMALL;
    while (c < n) c <<= 1;
    return c;
}

static inline cudaError_t dmalloc(void ** p, size_t n){
    if (!g_arena_enabled || n == 0) return cudaMalloc(p, n);
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= ROSHAMBO3_MAX_DEVICES) return cudaMalloc(p, n);
    const size_t want = size_class(n);
    {
        std::lock_guard<std::mutex> _lk(g_arena_mu);
        DeviceCache & c = g_cache[dev];
        *p = nullptr;
        for (size_t i = 0; i < c.free_blocks.size(); ++i){
            if (c.free_blocks[i].first == want){
                *p = c.free_blocks[i].second;
                c.free_blocks[i] = c.free_blocks.back();
                c.free_blocks.pop_back();
                c.hits += 1;
                break;
            }
        }
        if (*p){ c.live[*p] = want; }
    }
    if (*p) return cudaSuccess;
    cudaError_t e = cudaMalloc(p, want);
    if (e == cudaSuccess){
        std::lock_guard<std::mutex> _lk(g_arena_mu);
        g_cache[dev].misses += 1;
        g_cache[dev].held_bytes += want;
        g_cache[dev].live[*p] = want;
    }
    return e;
}

static inline cudaError_t dfree(void * p){
    if (!g_arena_enabled || p == nullptr) { Section _s(&g_stages.free_s);
      return cudaFree(p); }
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= ROSHAMBO3_MAX_DEVICES) { Section _s(&g_stages.free_s);
      return cudaFree(p); }
    bool known = false;
    size_t want = 0;
    {
        std::lock_guard<std::mutex> _lk(g_arena_mu);
        DeviceCache & c = g_cache[dev];
        auto it = c.live.find(p);
        if (it != c.live.end()){
            want = it->second; c.live.erase(it); known = true;
            c.free_blocks.push_back(std::make_pair(want, p));
        }
    }
    // A pointer this allocator never handed out (allocated before the arena was
    // switched on, say) goes back to the driver, as it must.
    if (known) return cudaSuccess;
    { Section _s(&g_stages.free_s);
      return cudaFree(p); }
}

/// Hand everything back to the driver. Called from Python between phases so a
/// long screen does not sit on device memory a sibling process wants.
static void arena_trim(){
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess) return;
    int cur = 0; cudaGetDevice(&cur);
    for (int d = 0; d < n_dev && d < ROSHAMBO3_MAX_DEVICES; ++d){
        cudaSetDevice(d);
        for (auto & b : g_cache[d].free_blocks) { Section _s(&g_stages.free_s);
          cudaFree(b.second); }
        g_cache[d].free_blocks.clear();
        g_cache[d].held_bytes = 0;
    }
    cudaSetDevice(cur);
}

}  // namespace runtime

////////////////////////////////////////////////////////////////////////////////
/// The one scoring entry point.
///
/// Computes overlap for an explicit list of (query row, library row) pairs on one
/// device. `pose_search` selects the kernel:
///
///   0  FIELD     the production path. A precomputed query field drives the pose
///                search; the converged pose is rescored with the exact analytic
///                pair sum in the same launch. Shape only, for now -- the field
///                is the query's SHAPE field and there is no color field.
///   1  ANALYTIC  the exact analytic optimizer. The Tier-3 specification, the
///                path that carries color, and how ground truth is generated.
///
/// Both read the transposed 32-row library tile (roshambo3.layout.tile32), so a
/// caller switches between them without re-laying out a byte of the corpus.
///
/// With `topk` non-zero the dense `[n_rows, 20]` score block is never copied
/// back and `V` is never written: it is folded into the running device-resident
/// top-K instead. On a 1e8-conformer tier that copy was 63.98 GB and the host
/// reduction that consumed it was 36.8% of the critical path.
////////////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------
// THE QUERY FIELD CACHE
//
// `build_query_field` (kernels.cu) is a pure function of the query panel and
// the lattice parameters, and the block that calls it says exactly that -- "a
// per-query cost amortized over every library". It was not amortized. The
// whole build ran on every `optimize_pairs` call, and the engine issues one
// call per 4,096-conformer block, so a screen rebuilt every query field
// thousands of times over.
//
// Measured on one corpus group at 8 queries: 0.0247 s of a
// 60-launch pass against 0.0014 s for the same work in one launch -- 26% of a
// shipped launch, and 42% of everything launch granularity costs.
//
// One entry per device, keyed on an EXACT COPY of every input the fields are
// built from and compared element-wise. A hash would be smaller and would risk
// handing one query's field to another query -- a silently wrong score, not a
// crash. The key is a few KB against fields of megabytes.
// ---------------------------------------------------------------------------
namespace {

struct QueryFieldKey {
    int   n_q = 0, NmolA = 0, D = 0, N_features = 0;
    int   color_min_sources = 0, color_placement = 0;
    int   optim_color = 0, build_color = 0;
    float spacing = 0.f, pad = 0.f, color_spacing = 0.f;
    std::vector<float> molAs;
    std::vector<int>   molA_num;
    std::vector<int>   molA_type;
    std::vector<float> rmat;

    bool operator==(const QueryFieldKey & o) const {
        return n_q == o.n_q && NmolA == o.NmolA && D == o.D
            && N_features == o.N_features
            && color_min_sources == o.color_min_sources
            && color_placement == o.color_placement
            && optim_color == o.optim_color && build_color == o.build_color
            && spacing == o.spacing && pad == o.pad
            && color_spacing == o.color_spacing
            && molAs == o.molAs && molA_num == o.molA_num
            && molA_type == o.molA_type && rmat == o.rmat;
    }
};

struct QueryFieldEntry {
    bool valid = false;
    int  device_id = -1;
    QueryFieldKey key;
    std::vector<cudaArray_t>         arrays;
    std::vector<cudaTextureObject_t> texs;
    std::vector<cudaArray_t>         carrays;
    std::vector<cudaTextureObject_t> ctexs;
    QueryField * d_fields = nullptr;
    QueryField * d_color_fields = nullptr;
};

std::mutex g_field_cache_mu;
std::unordered_map<int, QueryFieldEntry> g_field_cache;

// The caller must hold g_field_cache_mu. Sets the device first: a texture and
// its array belong to the context they were made in, and freeing them from
// another one is not a no-op.
//
// It *restores* the device too. Setting and not restoring left the
// calling thread on the freed entry's device, so a Python-side
// `clear_field_cache()` silently moved the current device out from under
// whatever ran next -- the same defect as the one fixed in the top-K
// sinks, pointing the other way. No-throw, because this runs on cleanup paths.
void field_cache_release_locked(QueryFieldEntry & e){
    ScopedDevice on_device(e.valid ? e.device_id : -1, std::nothrow);
    for (size_t i = 0; i < e.texs.size(); ++i)
        if (e.texs[i]) cudaDestroyTextureObject(e.texs[i]);
    for (size_t i = 0; i < e.arrays.size(); ++i)
        if (e.arrays[i]) cudaFreeArray(e.arrays[i]);
    for (size_t i = 0; i < e.ctexs.size(); ++i)
        if (e.ctexs[i]) cudaDestroyTextureObject(e.ctexs[i]);
    for (size_t i = 0; i < e.carrays.size(); ++i)
        if (e.carrays[i]) cudaFreeArray(e.carrays[i]);
    if (e.d_fields)       runtime::dfree(e.d_fields);
    if (e.d_color_fields) runtime::dfree(e.d_color_fields);
    e = QueryFieldEntry();
}

long long field_cache_clear_all(){
    std::lock_guard<std::mutex> g(g_field_cache_mu);
    long long n = 0;
    for (auto & kv : g_field_cache){
        if (kv.second.valid) ++n;
        field_cache_release_locked(kv.second);
    }
    g_field_cache.clear();
    return n;
}

long long g_field_cache_hits = 0, g_field_cache_misses = 0;

}  // namespace

// `BCNT` is the library's per-conformer COUNT VECTOR, uint8
// [n_t, N_features] with column 0 the real atom count and columns 1.. the
// per-type feature counts. It replaces BOTH the per-slot type plane (`BT`) and
// the separate atom-count array (`BN`) -- counts[:, 0] IS n_real, so there is
// one number now rather than two that had to agree. `library_color_offset` is
// where a library conformer's color run begins, i.e. the group's padded shape
// width; every launch is single-bucket, so it is a scalar.
// `B`, the LIBRARY coordinate record, is untyped here because its element
// type is now a choice. int16 [n_t_rows, L, 3] is the wire record -- what the
// corpus holds and what the engine sends, 6 B/atom instead of 16. float32
// [n_t_rows, L, 4] is the four-component record, which a caller whose
// coordinates are not on the corpus's wire-step grid must keep: quantizing those
// would be a numerics change rather than an encoding, and the committed golden
// fixture, the ROSHAMBO2 port gate and the hand-built fixtures are all off-grid.
// The dtype selects the instantiation; it is never converted here.
/// `optimize_pairs(call_stats=...)`: one call's self-overlap work and bound
/// scan, as int64. Mirrored by `_CALL_STATS_FIELDS` in backend/cuda.py.
enum : int {
    CALL_STATS_QUERY_SELF = 0,            // query self-overlaps evaluated
    CALL_STATS_LIBRARY_SELF = 1,          // library self-overlaps evaluated
    CALL_STATS_LIBRARY_SELF_SUPPLIED = 2, // 1 if the library's were supplied
    CALL_STATS_BOUND_CHECKED = 3,         // rows the bound scan read; -1 off
    CALL_STATS_BOUND_FLAGGED = 4,         // rows past the bound; -1 off
    CALL_STATS_BOUND_FIRST_ROW = 5,       // lowest flagged row; -1 none / off
    CALL_STATS_LEN = 6
};

void optimize_pairs(py::array_t<float> A, py::array_t<int> AT, py::array_t<int> AN,
                    py::array B, py::array_t<unsigned char> BCNT,
                    py::array_t<float> RMAT, py::array_t<float> PMAT,
                    // The pair list, or None and None for a DENSE launch
                    // (row q*n + t is query q against library row t).
                    py::object OUT_Q, py::object OUT_T,
                    py::array_t<float> V,
                    bool optim_color, float mixing_param,
                    float lr_q, float lr_t, int nsteps,
                    int start_mode, int pose_search, int device_id,
                    bool legacy_combo_gradient,
                    // the analytic kernel's staging switches
                    bool query_vector_load, int stage_width, bool stage_int16,
                    bool hoist_cross_product, float distance_cutoff_sq,
                    bool omit_shape_weights, bool stage_xyz_only, int update_rule,
                    // The color loop's structure, and where the color term
                    // is evaluated. COLOR_LOOP_* / COLOR_PLACEMENT_* in
                    // kernels.cuh; both are runtime, so mode 3 stays reachable.
                    int color_loop, int color_placement,
                    // The per-plane color field. `color_field_min_sources`
                    // is the query-side count a plane must clear before a field
                    // is built for it -- 0 is the all-analytic arm, and the two
                    // are one binary so the comparison is a causal ablation on
                    // the production kernel rather than between two builds.
                    int color_field_min_sources, float color_field_spacing_a,
                    // The lane layout, the register budget, the
                    // color channel's load hint and the interleave arm.
                    int lanes_per_group, int color_residency, int color_stage,
                    int query_color_residency, int query_shape_width,
                    int library_shape_width, int library_color_offset,
                    int color_ballot, int interleave, int stream_min_blocks_per_sm,
                    // the field kernel's knobs
                    float field_spacing_a, float field_pad_a, int update_lanes,
                    int smem_carveout, int coord_residency, int min_blocks_per_sm,
                    int start_groups_per_row,
                    // the device top-K
                    std::uintptr_t topk, long long gid_base, long max_rows_per_query,
                    // the molecule-level top-K, which owns its own conformer-level
                    // state and reduces this block before folding it in
                    std::uintptr_t topk_molecule,
                    // Which tile the LIBRARY array is in. > 0 the
                    // interleaved tile [tile][atom][lane][component], < 0 the
                    // planar one [tile][atom][component][lane]. It is the sign
                    // of the kernels' `TILEB`, and it is the caller's to state
                    // rather than this file's to assume: whoever packed the
                    // buffer is the only party that knows how it is laid out.
                    // See roshambo3.layout.tile32.
                    int library_tile_mode,
                    // Count the rows past the self-overlap
                    // bound, on the device, over the whole block before any
                    // reduction. Off by default; off launches nothing.
                    bool check_self_overlap_bound,
                    double self_overlap_bound_margin,
                    // None, or a writable int64 array of CALL_STATS_LEN that
                    // receives this call's counters (see CALL_STATS_* below).
                    py::object call_stats,
                    py::object winning_starts, py::object start_objectives)
{
    const int D  = 4;
    const int DV = 20;
    const bool field = (pose_search == 0);

    // This call's counters, written to `call_stats` if the caller passed one.
    // Layout (int64): see the CALL_STATS_* constants and cuda.py's
    // `_CALL_STATS_FIELDS`, which must agree.
    long long * stats_out = nullptr;
    if (!call_stats.is_none()){
        // The caller's own array, not a converted copy: a cast to array_t would
        // silently allocate a new one for any other dtype, and the counters
        // would be written somewhere nobody reads. `call_stats` keeps it alive
        // for the whole call.
        if (!py::isinstance<py::array>(call_stats))
            throw std::runtime_error("roshambo3.cuda: call_stats must be a numpy array");
        py::array cs = py::reinterpret_borrow<py::array>(call_stats);
        if (!cs.dtype().is(py::dtype::of<long long>()) || cs.ndim() != 1
            || cs.size() < CALL_STATS_LEN || !cs.writeable()
            || !(cs.flags() & py::array::c_style))
            throw std::runtime_error(
                "roshambo3.cuda: call_stats must be a writable contiguous int64 "
                "array of at least " + std::to_string(CALL_STATS_LEN) + " elements");
        stats_out = static_cast<long long *>(cs.mutable_data());
        stats_out[CALL_STATS_QUERY_SELF] = 0;
        stats_out[CALL_STATS_LIBRARY_SELF] = 0;
        stats_out[CALL_STATS_LIBRARY_SELF_SUPPLIED] = 0;
        // -1 is "the scan did not run", never "it ran and found nothing".
        stats_out[CALL_STATS_BOUND_CHECKED] = -1;
        stats_out[CALL_STATS_BOUND_FLAGGED] = -1;
        stats_out[CALL_STATS_BOUND_FIRST_ROW] = -1;
    }
    // Finite and above -1, so (1 + margin) is a positive factor. A negative
    // margin is not a production setting (scores.SELF_OVERLAP_MARGIN_POLICY
    // holds those), but it is how a test makes ordinary rows count.
    if (check_self_overlap_bound
        && !(std::isfinite(self_overlap_bound_margin)
             && self_overlap_bound_margin > -1.0))
        throw std::runtime_error(
            "roshambo3.cuda: self_overlap_bound_margin must be finite and > -1");

    if (pose_search != 0 && pose_search != 1)
        throw std::runtime_error("roshambo3.cuda: pose_search must be 0 (FIELD) or 1 (ANALYTIC)");
    if (update_rule < 0 || update_rule >= UPDATE_RULE_COUNT)
        throw std::runtime_error("roshambo3.cuda: update_rule must be 0.."
                                 + std::to_string(UPDATE_RULE_COUNT - 1)
                                 + ", got " + std::to_string(update_rule));
    // COLOR_LOOP_MASKED_PADDED (0) is retired. It was the only structure with
    // a padding-reachable loop, BOUNDED reproduces it
    // bitwise, and it is refused here rather than renumbered away, so a stale 0
    // from an archived script fails loudly instead of selecting a different
    // structure.
    if (color_loop <= COLOR_LOOP_MASKED_PADDED || color_loop >= COLOR_LOOP_COUNT)
        throw std::runtime_error("roshambo3.cuda: color_loop must be "
                                 + std::to_string(COLOR_LOOP_BOUNDED) + ".."
                                 + std::to_string(COLOR_LOOP_COUNT - 1)
                                 + ", got " + std::to_string(color_loop)
                                 + " (0, MASKED_PADDED, is retired)");
    if (color_placement < 0 || color_placement >= COLOR_PLACEMENT_COUNT)
        throw std::runtime_error("roshambo3.cuda: color_placement must be 0 "
                                 "(IN_GRADIENT) or 1 (AT_POSE), got "
                                 + std::to_string(color_placement));

    int deviceCount = 0;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));
    if (device_id < 0 || device_id >= deviceCount)
        throw std::runtime_error("roshambo3.cuda: device_id " + std::to_string(device_id)
                                 + " out of range (" + std::to_string(deviceCount) + " devices)");

    auto molAs          = A.unchecked<3>();
    auto molA_type      = AT.unchecked<2>();
    auto molA_num_atoms = AN.unchecked<1>();
    auto molB_counts    = BCNT.unchecked<2>();
    // The library record's element type, taken from the ARRAY rather than
    // from a flag, so a caller cannot describe one record and pass another.
    const bool lib_i16 = B.dtype().is(py::dtype::of<short>());
    if (!lib_i16 && !B.dtype().is(py::dtype::of<float>()))
        throw std::runtime_error(
            "roshambo3.cuda: library_coords must be int16 (the [n_rows, L, 3] "
            "wire record) or float32 (the [n_rows, L, 4] one)");
    if (B.ndim() != 3)
        throw std::runtime_error(
            "roshambo3.cuda: library_coords must be 3-dimensional, got " +
            std::to_string(B.ndim()));
    if (!(B.flags() & py::array::c_style))
        throw std::runtime_error(
            "roshambo3.cuda: library_coords must be C-contiguous");
    const int DB_HOST = lib_i16 ? ROSHAMBO3_D16 : D_HOST;
    if (B.shape(2) != DB_HOST)
        throw std::runtime_error(
            "roshambo3.cuda: an " + std::string(lib_i16 ? "int16" : "float32") +
            " library record is " + std::to_string(DB_HOST) +
            " components wide, got " + std::to_string(B.shape(2)));
    auto rmat           = RMAT.unchecked<2>();
    auto pmat           = PMAT.unchecked<2>();
    // No pair list means a dense launch. The pair list is the sparse
    // form, kept for callers that score an explicit subset; a dense launch
    // derives (query, library row) from the launch shape instead, so nothing
    // is built, scanned or uploaded for it.
    if (OUT_Q.is_none() != OUT_T.is_none())
        throw std::invalid_argument(
            "roshambo3.cuda: pass both halves of the pair list, or neither for "
            "a dense launch");
    const bool dense = OUT_Q.is_none();
    py::array_t<int> out_q_arr, out_t_arr;
    if (!dense){
        out_q_arr = py::array_t<int>::ensure(OUT_Q);
        out_t_arr = py::array_t<int>::ensure(OUT_T);
        if (!out_q_arr || !out_t_arr || out_q_arr.ndim() != 1 || out_t_arr.ndim() != 1)
            throw std::invalid_argument(
                "roshambo3.cuda: the pair list must be two int32 vectors");
        if (out_q_arr.shape(0) != out_t_arr.shape(0))
            throw std::invalid_argument(
                "roshambo3.cuda: the two halves of the pair list differ in length");
    }

    int  N_features = rmat.shape(0);
    long NmolA  = molAs.shape(1);
    long NmolB  = B.shape(1);
    long n_q    = molAs.shape(0);
    // The library array is padded up to a whole number of 32-row tiles, so its
    // row count is NOT the library count. BCNT is never tiled -- it is one
    // 8-byte record per conformer -- so it is the authority on how many targets
    // there really are.
    long n_t_rows = B.shape(0);
    long n_t      = molB_counts.shape(0);
    long n_rows   = dense ? n_q * n_t : (long)out_q_arr.shape(0);
    // Output rows are int work-item offsets on the device, in both forms.
    if (n_rows > (long)std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "roshambo3.cuda: " + std::to_string(n_rows) + " pairs in one launch "
            "exceeds the int32 row index; launch fewer library rows");
    // Optional dense search diagnostics. Keep the caller's exact arrays:
    // py::array_t conversion would silently allocate on a dtype mismatch.
    int * winners_out = nullptr;
    float * objectives_out = nullptr;
    const int diagnostic_starts = start_mode == 0 ? 1 : start_mode == 1 ? 4 : start_mode == 2 ? 10 : 0;
    if (!winning_starts.is_none() || !start_objectives.is_none()) {
        if (topk || topk_molecule)
            throw std::invalid_argument("CUDA start metadata requires a dense call");
        if (!diagnostic_starts)
            throw std::invalid_argument("invalid start_mode for CUDA start metadata");
        const int diagnostic_color_mode = !optim_color ? 0
            : ((color_placement == COLOR_PLACEMENT_AT_POSE) ? 1 : 2);
        if (field && n_rows > 0 && !field_start_metadata_supported(
                library_tile_mode, lanes_per_group, coord_residency, update_lanes,
                update_rule, query_color_residency, color_ballot, interleave,
                color_residency, color_stage, start_groups_per_row,
                stream_min_blocks_per_sm, diagnostic_color_mode))
            throw std::invalid_argument(
                "CUDA FIELD start metadata requires a supported default dispatch cell");
        if (!winning_starts.is_none()) {
            if (!py::isinstance<py::array>(winning_starts))
                throw std::invalid_argument("winning_starts must be a numpy array");
            auto array = py::reinterpret_borrow<py::array>(winning_starts);
            if (!array.dtype().is(py::dtype::of<int>()) || array.ndim() != 1 ||
                array.shape(0) != n_rows || !array.writeable() ||
                !(array.flags() & py::array::c_style))
                throw std::invalid_argument("winning_starts must be writable contiguous int32 [n_pairs]");
            winners_out = static_cast<int *>(array.mutable_data());
        }
        if (!start_objectives.is_none()) {
            if (!py::isinstance<py::array>(start_objectives))
                throw std::invalid_argument("start_objectives must be a numpy array");
            auto array = py::reinterpret_borrow<py::array>(start_objectives);
            if (!array.dtype().is(py::dtype::of<float>()) || array.ndim() != 2 ||
                array.shape(0) != n_rows || array.shape(1) != diagnostic_starts ||
                !array.writeable() || !(array.flags() & py::array::c_style))
                throw std::invalid_argument("start_objectives must be writable contiguous float32 [n_pairs,n_starts]");
            objectives_out = static_cast<float *>(array.mutable_data());
        }
    }


    const auto output_overlaps = [](const void* a, size_t an, const void* b, size_t bn) {
        if (!an || !bn) return false;
        const auto ap = reinterpret_cast<std::uintptr_t>(a);
        const auto bp = reinterpret_cast<std::uintptr_t>(b);
        return ap < bp + bn && bp < ap + an;
    };
    const size_t winner_bytes = winners_out ? size_t(n_rows)*sizeof(int) : 0;
    const size_t objective_bytes = objectives_out ? size_t(n_rows)*diagnostic_starts*sizeof(float) : 0;
    if (output_overlaps(winners_out, winner_bytes, objectives_out, objective_bytes) ||
        output_overlaps(winners_out, winner_bytes, V.data(), size_t(V.size())*sizeof(float)) ||
        output_overlaps(objectives_out, objective_bytes, V.data(), size_t(V.size())*sizeof(float)))
        throw std::invalid_argument("CUDA start metadata output buffers must not overlap");

    // The count vector's width has to be the interaction matrix's, because
    // both are indexed by type id and the kernel uses one to index the other.
    if (molB_counts.shape(1) != N_features)
        throw std::runtime_error(
            "roshambo3.cuda: the library count vector is " +
            std::to_string(molB_counts.shape(1)) + " wide and the interaction "
            "matrix is " + std::to_string(N_features) + "x" +
            std::to_string(N_features) + ". Both are indexed by feature type "
            "id, so they must agree.");
    // Byte 7 of the packed word is the redundant color total, so seven type
    // bytes is the ceiling. Refused here rather than silently aliasing the sum
    // over the last type's count.
    if (N_features > ROSHAMBO3_MAX_FEATURES)
        throw std::runtime_error(
            "roshambo3.cuda: " + std::to_string(N_features) + " feature types, "
            "but the per-conformer count vector packs into one 64-bit word "
            "whose byte 7 carries the redundant color total -- at most " +
            std::to_string(ROSHAMBO3_MAX_FEATURES) + " types fit.");
    // Where a library conformer's color run begins. >= 0 is an absolute slot,
    // uniform over the launch, which is what a corpus block wants: every launch
    // is single-bucket and `combine_shape_color` puts a conformer's features at
    // the group's PADDED shape width. < 0 means "immediately after each
    // conformer's own real atoms", which is how `engine/query.py` lays a query
    // out and how the hand-built fixtures lay a library out.
    if (optim_color && library_color_offset > NmolB)
        throw std::runtime_error(
            "roshambo3.cuda: library_color_offset " +
            std::to_string(library_color_offset) + " is past the library's "
            "padded width " + std::to_string(NmolB));

    // Both kernels read a tiled library layout, and they read either
    // of the two. The value travels unchanged to `compute_self_overlaps_gpu`
    // and to both pair launchers, so one array cannot be read at two layouts
    // inside one call -- which is the failure this used to be pinned to avoid,
    // and which the plumbing now prevents by construction instead.
    if (library_tile_mode == 0)
        throw std::runtime_error(
            "roshambo3.cuda: library_tile_mode must be > 0 (the interleaved "
            "tile) or < 0 (the planar tile); the untiled per-molecule layout is "
            "not read by either pair kernel. See roshambo3.layout.tile32.");
    const int tile_mode = library_tile_mode;
    if (n_t_rows < ((n_t + 31)/32)*32)
        throw std::runtime_error("roshambo3.cuda: the library array must be padded to whole "
                                 "32-row tiles (see roshambo3.layout.tile32)");

    // Guards for things the field path does NOT do, named rather than silently
    // ignored.
    if (field){
        // The field path carries color. The field is still the query's
        // SHAPE field and there is still no color field -- what changed is that
        // the color GRADIENT does not need one: it is an analytic
        // pair sum over a small same-type set, evaluated in the same loop that
        // samples the shape texture.
        //
        // It implements COLOR_LOOP_SAME_TYPE_NOW and only that structure, so the
        // per-type constants can be hoisted into a per-block table. That is
        // exact for the canonical diagonal interaction matrix and wrong for any
        // other, which backend.cuda.score_pairs already refuses; naming a
        // different structure here is refused rather than silently substituted.
        if (optim_color && color_loop != COLOR_LOOP_SAME_TYPE_NOW)
            throw std::runtime_error(
                "roshambo3.cuda: the field path implements color_loop=3 "
                "(SAME_TYPE_NOW) only -- it hoists the per-type interaction "
                "constants into a per-block table, which is exact for the "
                "canonical diagonal matrix and for no other. Use "
                "pose_search=1 (ANALYTIC) for the other structures.");
        // The field kernel carries an OPT template parameter, so the
        // two SHIPPING rules both reach it. The other three do not: Adam, QN2
        // and BFGS each need per-pose auxiliary state (8, 12 and 33 floats) on
        // top of `cache`, against a kernel pinned at 64 registers per thread,
        // and they are analytic-path measurement arms rather than shipping
        // rules. Refused by name, not ignored.
        if (update_rule != UPDATE_ADAGRAD_QADD && update_rule != UPDATE_ADAGRAD_EXP)
            throw std::runtime_error("roshambo3.cuda: the field path carries update_rule 0 "
                                     "(ADAGRAD_QADD) and 1 (ADAGRAD_EXP) only; Adam, QN2 and "
                                     "BFGS are instantiated on the analytic path");
        if (!(field_spacing_a > 0.0f))
            throw std::runtime_error("roshambo3.cuda: field_spacing_a must be > 0");
        if (update_lanes != 0 && update_lanes != 2 && update_lanes != 3)
            throw std::runtime_error("roshambo3.cuda: update_lanes must be 0, 2 or 3");
        // 4 is GLOBAL_LDCS. This check is duplicated in
        // optimize_pairs_field and was the one that still said 3 -- a second
        // copy of a range is a second place to forget, and it refused six cells
        // of the shape matrix after the launcher had already accepted them.
        if (coord_residency < 0 || coord_residency > 4)
            throw std::runtime_error("roshambo3.cuda: coord_residency must be 0 (SHARED), "
                                     "1 (GLOBAL), 2 (GLOBAL_LDCG), 3 (REGISTER) or "
                                     "4 (GLOBAL_LDCS)");
        if (min_blocks_per_sm < 1 || min_blocks_per_sm > 4)
            throw std::runtime_error("roshambo3.cuda: min_blocks_per_sm must be 1..4");
        // The kernel-side range check lives in optimize_pairs_field, which
        // also knows the start count; this one exists because `chunk` below is
        // computed from it BEFORE that launcher runs, and a bad value there
        // silently mis-sizes every work item rather than throwing.
        if (start_groups_per_row < 1 || start_groups_per_row > 8
            || (start_groups_per_row & (start_groups_per_row - 1)) != 0)
            throw std::runtime_error(
                "roshambo3.cuda: start_groups_per_row must be a power of two in 1..8");
        // Checked HERE as well as in the launcher, because `chunk`
        // is computed from it before the launcher ever runs.
        if (lanes_per_group < 1 || lanes_per_group > 32
            || (lanes_per_group & (lanes_per_group - 1)) != 0)
            throw std::runtime_error(
                "roshambo3.cuda: lanes_per_group must be a power of two in 1..32");
        if ((long)lanes_per_group*(long)start_groups_per_row > 32)
            throw std::runtime_error(
                "roshambo3.cuda: lanes_per_group * start_groups_per_row must be "
                "<= 32 so the per-row best stays an in-warp reduction");
        if ((long)COMBO_NTHREADS %
                ((long)lanes_per_group*(long)start_groups_per_row))
            throw std::runtime_error(
                "roshambo3.cuda: lanes_per_group * start_groups_per_row must "
                "divide the block width");
    }
    if (!field && start_groups_per_row != 1)
        throw std::runtime_error(
            "roshambo3.cuda: start_groups_per_row is a field-kernel lane assignment; "
            "the analytic kernel gives one thread a whole pair and has no start split");
    const int stage = (!field && stage_width > 0) ? stage_width : 0;
    if (!field && optim_color && stage)
        throw std::runtime_error("roshambo3.cuda: library staging is shape-only; the color loop "
                                 "reads its coordinates at the global stride");

    // A staging *format* is meaningless without staging, and this is the one
    // place where that contradiction is expressible: `stage_int16` and
    // `stage_xyz_only` default to the shipped staged record, so a caller who
    // says "no staging" and names no format -- which is every caller of
    // backend.score_pairs, and every color caller, since ScreenConfig forces
    // stage_width to 0 for color -- used to send the format flags on down and
    // compose a dispatch key for a staged kernel with stage width 0. That key
    // is not instantiated and the launch threw. Clear the flags here instead,
    // so `stage_width == 0` means no staging and nothing else.
    const bool stage_i16_eff = stage ? stage_int16    : false;
    const bool stage_p3_eff  = stage ? stage_xyz_only : false;

    assert(V.shape(1) == DV);
    DeviceTopKState * topk_state = nullptr;
    MoleculeTopKState * mol_state = nullptr;
    if (topk && topk_molecule)
        throw std::runtime_error(
            "roshambo3.cuda: pass either a conformer-level or a molecule-level "
            "top-K, not both -- the molecule-level state owns its own");
    if (topk){
        topk_state = reinterpret_cast<DeviceTopKState *>(topk);
        if (topk_dv(topk_state) != DV)
            throw std::runtime_error("roshambo3.cuda: the top-K state was built for a "
                                     "different number of score columns");
    }
    if (topk_molecule){
        mol_state = reinterpret_cast<MoleculeTopKState *>(topk_molecule);
        if (topk_dv(moltopk_inner(mol_state)) != DV)
            throw std::runtime_error("roshambo3.cuda: the top-K state was built for a "
                                     "different number of score columns");
    }
    runtime::count_call();
    if (n_rows == 0){
        if (stats_out && check_self_overlap_bound){
            stats_out[CALL_STATS_BOUND_CHECKED] = 0;
            stats_out[CALL_STATS_BOUND_FLAGGED] = 0;
        }
        return;
    }

    const float * ptr_molAs     = molAs.data(0,0,0);
    const int   * ptr_molA_type = molA_type.data(0,0);
    const int   * ptr_molA_num  = molA_num_atoms.data(0);
    const void  * ptr_molBs     = B.data();
    const float * ptr_rmat      = rmat.data(0,0);
    const float * ptr_pmat      = pmat.data(0,0);
    const int   * ptr_out_q     = dense ? nullptr : out_q_arr.data();
    const int   * ptr_out_t     = dense ? nullptr : out_t_arr.data();
    float       * ptr_V         = V.mutable_data(0, 0);

    // ---- pack the count vectors, while the GIL is still held ----------------
    //
    // One 64-bit word per conformer: byte 0 the real atom count, byte t the
    // count of type t, byte 7 the REDUNDANT color total. The redundancy is the
    // point -- it makes the color bound one mask read on the device instead of
    // a six-way sum, and it is checkable, which a count vector needs because a
    // wrong one is a plausible wrong number rather than a loud failure. A wrong
    // TYPE PLANE read back as "every color term exactly zero". An
    // off-by-one in a run boundary does not.
    std::vector<r3_counts_t> h_molB_cnt((size_t)n_t);
    for (long r = 0; r < n_t; ++r){
        r3_counts_t w = 0;
        int nc = 0;
        for (int k = 0; k < N_features; ++k){
            const unsigned int c = (unsigned int)molB_counts(r, k);
            w |= (r3_counts_t)c << (8*k);
            if (k) nc += (int)c;
        }
        if (nc > 255)
            throw std::runtime_error(
                "roshambo3.cuda: library row " + std::to_string(r) + " carries " +
                std::to_string(nc) + " color features, more than the 255 the "
                "redundant total byte can hold");
        w |= (r3_counts_t)(unsigned int)nc << 56;
        h_molB_cnt[(size_t)r] = w;
        // Read the word back OUT OF THE ARRAY, not out of the local: this
        // assertion is what turns a packing bug into a refusal instead of into
        // a color run that is one slot short for every conformer in the launch,
        // and checking the local checks the wrong thing. The first draft of
        // this loop forgot the store above and the assertion passed anyway --
        // every count reached the device as zero, so every library molecule had
        // zero atoms and every score was 0.0. Loud, as it happens, but only
        // because n_real is in the same word; a lost COLOR byte would not have
        // been.
        const r3_counts_t got = h_molB_cnt[(size_t)r];
        if (r3_n_color(got) != nc || r3_n_real(got) != (int)molB_counts(r, 0))
            throw std::runtime_error(
                "roshambo3.cuda: the packed count word does not read back as it "
                "was packed at library row " + std::to_string(r));
    }

    // The query's own count vector, for the self-overlap precompute -- which
    // runs `volume_single`, the same function the pair loop runs, so V_AA and
    // V_AB are produced by one code path. The query plane
    // puts its features immediately after its real atoms (engine/query.py), so
    // its color offset is its own n_real, which `compute_self_overlaps_gpu`
    // takes as a negative `color_offset`.
    //
    // **Contiguity is checked here, not assumed.** The count vector is a
    // lossless description of a plane whose types form one ascending run per
    // type and of no other; a query whose features were interleaved would have
    // its self-overlap computed over different pairs than its cross-overlap.
    std::vector<r3_counts_t> h_molA_cnt((size_t)n_q);
    for (long q = 0; q < n_q; ++q){
        const int nr = ptr_molA_num[q];
        r3_counts_t w = (r3_counts_t)(unsigned int)nr;
        int nc = 0, last = 0;
        for (long i = nr; i < NmolA; ++i){
            const int t = ptr_molA_type[(size_t)q*NmolA + i];
            if (t == 0) continue;
            if (t < 0 || t >= N_features)
                throw std::runtime_error(
                    "roshambo3.cuda: query " + std::to_string(q) + " slot " +
                    std::to_string(i) + " carries feature type " +
                    std::to_string(t) + ", outside the " +
                    std::to_string(N_features) + "-type interaction matrix");
            if (t < last)
                throw std::runtime_error(
                    "roshambo3.cuda: query " + std::to_string(q) + "'s color "
                    "types are not one ascending run per type (type " +
                    std::to_string(t) + " follows type " + std::to_string(last) +
                    "). The per-conformer count vector cannot represent "
                    "interleaving -- see pharmacophore.generate_color_atoms.");
            last = t;
            w += (r3_counts_t)1 << (8*t);
            ++nc;
        }
        w |= (r3_counts_t)(unsigned int)nc << 56;
        h_molA_cnt[(size_t)q] = w;
        if (r3_n_color(h_molA_cnt[(size_t)q]) != nc
            || r3_n_real(h_molA_cnt[(size_t)q]) != nr)
            throw std::runtime_error(
                "roshambo3.cuda: the packed count word does not read back as it "
                "was packed at query row " + std::to_string(q));
    }

    // Every pointer taken above is taken while the GIL is still held. The work
    // below runs under gil_scoped_release so a caller can decode the next block
    // on other threads while this one is on the GPU -- pybind11 holds the GIL
    // across a call by default, which made the device call a stop-the-world for
    // the whole host pipeline. Nothing below may touch a Python object.
    py::gil_scoped_release _nogil;

    CUDA_CHECK_ERROR(cudaSetDevice(device_id));

    // --- the full molecule arrays: any output row may reference any row -------
    float * d_molAs;     long molAs_sz     = n_q*NmolA*D*sizeof(float);
    int   * d_molA_type; long molA_type_sz = n_q*NmolA*sizeof(int);
    int   * d_molA_num;  long molA_num_sz  = n_q*sizeof(int);
    // The library record is int16 [.., L, 3] or float32 [.., L, 4] -- 6
    // bytes per atom on the wire instead of 16. This is the whole of the H2D
    // reduction; `stage_timings()` counts it.
    void  * d_molBs;     long molBs_sz     = n_t_rows*NmolB*DB_HOST
                                             * (lib_i16 ? (long)sizeof(short)
                                                        : (long)sizeof(float));
    // `n_t_rows`, not `n_t`, so a warp holding the last, partly padded
    // tile reads a defined word for lanes no output row references. 8 B per
    // conformer against the type plane's 4*NmolB.
    r3_counts_t * d_molB_cnt; long molB_cnt_sz = n_t_rows*sizeof(r3_counts_t);
    r3_counts_t * d_molA_cnt; long molA_cnt_sz = n_q*sizeof(r3_counts_t);
    float * d_rmat;      long rmat_sz      = (long)N_features*N_features*sizeof(float);
    float * d_pmat;      long pmat_sz      = rmat_sz;

    #define R3_ALLOC(p, sz)  { runtime::Section _s(&runtime::g_stages.alloc_s); \
        CUDA_CHECK_ERROR(runtime::dmalloc((void **)&(p), (sz))); }
    #define R3_H2D(dst, src, sz) { runtime::Section _s(&runtime::g_stages.h2d_s); \
        CUDA_CHECK_ERROR(cudaMemcpy((dst), (src), (sz), cudaMemcpyHostToDevice)); \
        runtime::add_bytes((long long)(sz), 0); }
    #define R3_FREE(p) { runtime::Section _s(&runtime::g_stages.free_s); runtime::dfree(p); }

    R3_ALLOC(d_molAs, molAs_sz);         R3_H2D(d_molAs, ptr_molAs, molAs_sz);
    R3_ALLOC(d_molA_type, molA_type_sz); R3_H2D(d_molA_type, ptr_molA_type, molA_type_sz);
    R3_ALLOC(d_molA_num, molA_num_sz);   R3_H2D(d_molA_num, ptr_molA_num, molA_num_sz);
    R3_ALLOC(d_molBs, molBs_sz);         R3_H2D(d_molBs, ptr_molBs, molBs_sz);
    R3_ALLOC(d_molB_cnt, molB_cnt_sz);
    // Zero first, then overwrite the real rows: the tail lanes of the last tile
    // hold no molecule and no output row references them, but an uninitialized
    // count would still be read by a warp that spans the boundary.
    { runtime::Section _s(&runtime::g_stages.h2d_s);
      CUDA_CHECK_ERROR(cudaMemset(d_molB_cnt, 0, molB_cnt_sz)); }
    R3_H2D(d_molB_cnt, h_molB_cnt.data(), n_t*(long)sizeof(r3_counts_t));
    R3_ALLOC(d_molA_cnt, molA_cnt_sz);   R3_H2D(d_molA_cnt, h_molA_cnt.data(), molA_cnt_sz);
    R3_ALLOC(d_rmat, rmat_sz);           R3_H2D(d_rmat, ptr_rmat, rmat_sz);
    R3_ALLOC(d_pmat, pmat_sz);           R3_H2D(d_pmat, ptr_pmat, pmat_sz);

    // --- precomputed self-overlaps -------------------------------------------
    // Every query's and every library's own reduced self-overlap, computed once
    // here rather than once per (pair, start mode) inside the kernel. n_q and
    // n_t are far smaller than the output-row count, so this is a small
    // one-time cost.
    float * d_va; long va_sz = n_q*2*sizeof(float);
    float * d_vb; long vb_sz = n_t*2*sizeof(float);
    R3_ALLOC(d_va, va_sz);
    R3_ALLOC(d_vb, vb_sz);
    // Library: raw coordinates with cscale = COORD_SCALE, because molB is never
    // rotated. Matches the old per-pair V_BB exactly.
    { runtime::Section _s(&runtime::g_stages.precompute_s);
      if (lib_i16)
          compute_self_overlaps_gpu((const short *)d_molBs, d_molB_cnt, (int)NmolB, n_t,
                                    library_color_offset,
                                    d_rmat, d_pmat, N_features, optim_color,
                                    ROSHAMBO3_COORD_SCALE, d_vb, device_id, tile_mode,
                                    color_loop);
      else
          compute_self_overlaps_gpu((const float *)d_molBs, d_molB_cnt, (int)NmolB, n_t,
                                    library_color_offset,
                                    d_rmat, d_pmat, N_features, optim_color,
                                    ROSHAMBO3_COORD_SCALE, d_vb, device_id, tile_mode,
                                    color_loop); }
    // Query: prescale first, then cscale = 1.0f -- bit-identical to what the
    // first start mode already puts in shared memory.
    float * d_molAs_scaled;
    R3_ALLOC(d_molAs_scaled, molAs_sz);
    { runtime::Section _s(&runtime::g_stages.precompute_s);
      prescale_coords_gpu(d_molAs, d_molAs_scaled, (int)NmolA, n_q, device_id, !field); }
    { runtime::Section _s(&runtime::g_stages.precompute_s);
      compute_self_overlaps_gpu(d_molAs_scaled, d_molA_cnt, (int)NmolA, n_q,
                                /*color_offset=*/-1,
                                d_rmat, d_pmat, N_features, optim_color,
                                1.0f, d_va, device_id, /*tile_mode=*/0,
                                color_loop); }
    R3_FREE(d_molAs_scaled);
    // Both self-overlap sets were just evaluated on the device, every call:
    // n_t library conformers and n_q queries. A consumer that supplies stored
    // library self-overlaps would skip the library launch above and report
    // library_self = 0, supplied = 1 instead.
    const long long query_self_evals = (long long)n_q;
    const long long library_self_evals = (long long)n_t;

    // --- one query field per query, on the device ----------------------------
    // The lattice is planned on the host from the raw coordinates, filled by a
    // device kernel, copied into a 3D CUDA array and wrapped in a texture with
    // cudaAddressModeBorder + cudaFilterModeLinear. Border is not a detail: it
    // returns exactly 0 outside the box, which is the field's true asymptote,
    // so no software bounds test is needed. Timed under precompute_s because
    // that is what it is -- a per-query cost amortized over every library.
    std::vector<cudaArray_t>         field_arrays;
    std::vector<cudaTextureObject_t> field_texs;
    QueryField * d_fields = nullptr;
    std::vector<cudaArray_t>         cfield_arrays;
    std::vector<cudaTextureObject_t> cfield_texs;
    QueryField * d_color_fields = nullptr;
    const bool build_color_fields =
        field && optim_color && color_placement == COLOR_PLACEMENT_IN_GRADIENT
        && color_field_min_sources > 0;

    // --- the cache lookup ----------------------------------------------------
    // Keyed on everything the fields are a function of. Note what is NOT in the
    // key: anything about the library. That is the whole point -- the field
    // does not depend on the block being scored, which is why rebuilding it per
    // block was pure repetition.
    bool field_from_cache = false;
    QueryFieldKey fkey;
    if (field){
        fkey.n_q = (int)n_q; fkey.NmolA = (int)NmolA; fkey.D = (int)D;
        fkey.N_features = (int)N_features;
        fkey.color_min_sources = (int)color_field_min_sources;
        fkey.color_placement = (int)color_placement;
        fkey.optim_color = optim_color ? 1 : 0;
        fkey.build_color = build_color_fields ? 1 : 0;
        fkey.spacing = field_spacing_a; fkey.pad = field_pad_a;
        fkey.color_spacing = color_field_spacing_a;
        fkey.molAs.assign(ptr_molAs, ptr_molAs + (size_t)n_q*NmolA*D);
        fkey.molA_num.assign(ptr_molA_num, ptr_molA_num + (size_t)n_q);
        fkey.molA_type.assign(ptr_molA_type, ptr_molA_type + (size_t)n_q*NmolA);
        fkey.rmat.assign(ptr_rmat, ptr_rmat + (size_t)N_features*N_features);

        std::lock_guard<std::mutex> g(g_field_cache_mu);
        auto it = g_field_cache.find((int)device_id);
        if (it != g_field_cache.end() && it->second.valid && it->second.key == fkey){
            const QueryFieldEntry & e = it->second;
            field_arrays = e.arrays; field_texs = e.texs;
            cfield_arrays = e.carrays; cfield_texs = e.ctexs;
            d_fields = e.d_fields; d_color_fields = e.d_color_fields;
            field_from_cache = true;
            ++g_field_cache_hits;
        } else {
            if (it != g_field_cache.end()) field_cache_release_locked(it->second);
            ++g_field_cache_misses;
        }
    }

    if (field && !field_from_cache){
        runtime::Section _s(&runtime::g_stages.precompute_s);
        std::vector<QueryField> h_fields((size_t)n_q);
        field_arrays.assign((size_t)n_q, (cudaArray_t)nullptr);
        field_texs.assign((size_t)n_q, (cudaTextureObject_t)0);
        size_t max_nodes = 0;
        std::vector<FieldGeometry> geoms((size_t)n_q);
        for (long q = 0; q < n_q; ++q){
            geoms[q] = plan_field_lattice(ptr_molAs + (size_t)q*NmolA*D,
                                          ptr_molA_num[q], (int)NmolA,
                                          field_spacing_a, field_pad_a);
            size_t nn = (size_t)(geoms[q].nx+1)*(geoms[q].ny+1)*(geoms[q].nz+1);
            if (nn > max_nodes) max_nodes = nn;
        }
        void * d_nodes = nullptr;
        CUDA_CHECK_ERROR(runtime::dmalloc(&d_nodes, max_nodes*8));   // half4
        cudaChannelFormatDesc ch = cudaCreateChannelDescHalf4();
        for (long q = 0; q < n_q; ++q){
            const FieldGeometry G = geoms[q];
            const int NX = G.nx+1, NY = G.ny+1, NZ = G.nz+1;
            build_query_field(d_molAs + (size_t)q*NmolA*D, ptr_molA_num[q],
                              (int)NmolA, G, d_nodes, device_id);
            cudaArray_t arr = nullptr;
            cudaExtent ext = make_cudaExtent(NX, NY, NZ);
            CUDA_CHECK_ERROR(cudaMalloc3DArray(&arr, &ch, ext, 0));
            cudaMemcpy3DParms p; memset(&p, 0, sizeof(p));
            p.srcPtr   = make_cudaPitchedPtr(d_nodes, (size_t)NX*8, NX, NY);
            p.dstArray = arr;
            p.extent   = ext;
            p.kind     = cudaMemcpyDeviceToDevice;
            CUDA_CHECK_ERROR(cudaMemcpy3D(&p));
            cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
            rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
            cudaTextureDesc td; memset(&td, 0, sizeof(td));
            td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeBorder;
            td.filterMode       = cudaFilterModeLinear;
            td.readMode         = cudaReadModeElementType;
            td.normalizedCoords = 0;
            cudaTextureObject_t tex = 0;
            CUDA_CHECK_ERROR(cudaCreateTextureObject(&tex, &rd, &td, NULL));
            field_arrays[q] = arr; field_texs[q] = tex;
            QueryField gd;
            gd.tex = (unsigned long long)tex;
            gd.ox = G.ox; gd.oy = G.oy; gd.oz = G.oz;
            gd.inv_hs = 1.0f/G.hs;
            gd.nx = G.nx; gd.ny = G.ny; gd.nz = G.nz;
            h_fields[q] = gd;
        }
        runtime::dfree(d_nodes);
        CUDA_CHECK_ERROR(runtime::dmalloc((void **)&d_fields, (size_t)n_q*sizeof(QueryField)));
        // Already inside the enclosing precompute_s Section, so this copy must
        // NOT be wrapped again. It is 32 B per query and belongs to the field
        // prep, not to the H2D of the corpus.
        CUDA_CHECK_ERROR(cudaMemcpy(d_fields, h_fields.data(),
                                    (size_t)n_q*sizeof(QueryField),
                                    cudaMemcpyHostToDevice));
    }

    // --- one color field per (query, served plane) ---------------------------
    // A field replaces a plane's O(n_query x n_library) pair sum with an
    // O(n_library) pass of lookups, so the saving factor is the QUERY count in
    // that plane. Measured: Hydrophobe mean 11.47 and 88.2% of same-type pair work;
    // the other five total ~7.4 atoms and cannot pay for a grid. The threshold
    // is what expresses that, per query, on the host, where the choice is
    // uniform over the block that will read it.
    //
    // The width is fixed and that is the enabler: `rmat` is all ones, so every
    // same-type pair of type t shares fac = rmat[t][t] * INV_TWO_ALPHA, and the
    // height p^2 a^-1.5 is a scalar the kernel applies after the fetch.
    if (build_color_fields && !field_from_cache){
        runtime::Section _s(&runtime::g_stages.precompute_s);
        const size_t ncell = (size_t)n_q*N_features;
        std::vector<QueryField> h_cf(ncell);
        memset(h_cf.data(), 0, ncell*sizeof(QueryField));
        cfield_arrays.assign(ncell, (cudaArray_t)nullptr);
        cfield_texs.assign(ncell, (cudaTextureObject_t)0);

        const float spacing = (color_field_spacing_a > 0.0f)
            ? color_field_spacing_a : field_spacing_a;

        // Per (query, type): the contiguous run of query slots carrying that
        // type, and its geometry. Two passes so one node buffer serves all.
        std::vector<int> lo(ncell, 0), hi(ncell, 0);
        std::vector<FieldGeometry> cgeom(ncell);
        size_t max_nodes = 0;
        for (long q = 0; q < n_q; ++q){
            const int * ty = ptr_molA_type + (size_t)q*NmolA;
            for (long i = ptr_molA_num[q]; i < NmolA; ++i){
                const int ft = ty[i];
                if (ft <= 0 || ft >= N_features) continue;
                const size_t c = (size_t)q*N_features + ft;
                if (hi[c] == 0) lo[c] = (int)i;
                hi[c] = (int)i + 1;
            }
            for (int ft = 1; ft < N_features; ++ft){
                const size_t c = (size_t)q*N_features + ft;
                if (hi[c] - lo[c] < color_field_min_sources) { hi[c] = lo[c]; continue; }
                if (ptr_rmat[ft*N_features + ft] == 0.0f) { hi[c] = lo[c]; continue; }
                cgeom[c] = plan_field_lattice_range(
                    ptr_molAs + (size_t)q*NmolA*D, lo[c], hi[c], (int)NmolA,
                    spacing, field_pad_a);
                const size_t nn = (size_t)(cgeom[c].nx+1)*(cgeom[c].ny+1)*(cgeom[c].nz+1);
                if (nn > max_nodes) max_nodes = nn;
            }
        }
        if (max_nodes > 0){
            void * d_nodes = nullptr;
            CUDA_CHECK_ERROR(runtime::dmalloc(&d_nodes, max_nodes*8));   // half4
            cudaChannelFormatDesc ch = cudaCreateChannelDescHalf4();
            for (long q = 0; q < n_q; ++q){
                for (int ft = 1; ft < N_features; ++ft){
                    const size_t c = (size_t)q*N_features + ft;
                    if (hi[c] <= lo[c]) continue;
                    const FieldGeometry G = cgeom[c];
                    const int NX = G.nx+1, NY = G.ny+1, NZ = G.nz+1;
                    const float fac = ptr_rmat[ft*N_features + ft]
                                    * ROSHAMBO3_INV_TWO_ALPHA;
                    build_query_color_field(d_molAs + (size_t)q*NmolA*D,
                                            lo[c], hi[c], (int)NmolA, G, fac,
                                            d_nodes, device_id);
                    cudaArray_t arr = nullptr;
                    cudaExtent ext = make_cudaExtent(NX, NY, NZ);
                    CUDA_CHECK_ERROR(cudaMalloc3DArray(&arr, &ch, ext, 0));
                    cudaMemcpy3DParms p; memset(&p, 0, sizeof(p));
                    p.srcPtr   = make_cudaPitchedPtr(d_nodes, (size_t)NX*8, NX, NY);
                    p.dstArray = arr;
                    p.extent   = ext;
                    p.kind     = cudaMemcpyDeviceToDevice;
                    CUDA_CHECK_ERROR(cudaMemcpy3D(&p));
                    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd));
                    rd.resType = cudaResourceTypeArray; rd.res.array.array = arr;
                    cudaTextureDesc td; memset(&td, 0, sizeof(td));
                    td.addressMode[0] = td.addressMode[1] = td.addressMode[2]
                        = cudaAddressModeBorder;
                    td.filterMode       = cudaFilterModeLinear;
                    td.readMode         = cudaReadModeElementType;
                    td.normalizedCoords = 0;
                    cudaTextureObject_t tex = 0;
                    CUDA_CHECK_ERROR(cudaCreateTextureObject(&tex, &rd, &td, NULL));
                    cfield_arrays[c] = arr; cfield_texs[c] = tex;
                    QueryField gd;
                    gd.tex = (unsigned long long)tex;
                    gd.ox = G.ox; gd.oy = G.oy; gd.oz = G.oz;
                    gd.inv_hs = 1.0f/G.hs;
                    gd.nx = G.nx; gd.ny = G.ny; gd.nz = G.nz;
                    h_cf[c] = gd;
                }
            }
            runtime::dfree(d_nodes);
        }
        CUDA_CHECK_ERROR(runtime::dmalloc((void **)&d_color_fields,
                                          ncell*sizeof(QueryField)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_color_fields, h_cf.data(),
                                    ncell*sizeof(QueryField),
                                    cudaMemcpyHostToDevice));
    }

    // --- publish the freshly built fields ------------------------------------
    // Stored AFTER both builds, so an entry is never visible holding the shape
    // field without the color fields the same key promises.
    if (field && !field_from_cache){
        std::lock_guard<std::mutex> g(g_field_cache_mu);
        QueryFieldEntry & e = g_field_cache[(int)device_id];
        if (e.valid) field_cache_release_locked(e);
        e.valid = true; e.device_id = (int)device_id; e.key = fkey;
        e.arrays = field_arrays; e.texs = field_texs;
        e.carrays = cfield_arrays; e.ctexs = cfield_texs;
        e.d_fields = d_fields; e.d_color_fields = d_color_fields;
        field_from_cache = true;      // the cache owns them now; do not destroy
    }

    // --- the pair list ----------------------------------------------------------
    // Sparse: uploaded as given. Dense: nothing is built on the host or
    // uploaded. The pair kernels' library-row plane is written on the device
    // (row q*n + t gets t), so those kernels -- and their code generation --
    // are exactly what they were; the query side needs no plane at all (work
    // items carry the query, the top-K takes it from its grid). Deriving the
    // row inside the pair kernels instead was tried and measured: the extra
    // argument re-allocated registers in the 64-register start-metadata
    // rescore cell and moved 6-44 of its score words by 1-2 ULP.
    int * d_out_q = nullptr; int * d_out_t = nullptr;
    if (!dense){
        R3_ALLOC(d_out_q, n_rows*sizeof(int)); R3_H2D(d_out_q, ptr_out_q, n_rows*sizeof(int));
        R3_ALLOC(d_out_t, n_rows*sizeof(int)); R3_H2D(d_out_t, ptr_out_t, n_rows*sizeof(int));
    } else {
        R3_ALLOC(d_out_t, n_rows*sizeof(int));
        runtime::Section _s(&runtime::g_stages.index_s);
        dense_library_rows(d_out_t, (int)n_q, (long)n_t, device_id);
    }

    float * d_scores;
    R3_ALLOC(d_scores, n_rows*DV*sizeof(float));
    int * d_winning_starts = nullptr;
    float * d_start_objectives = nullptr;
    if (winners_out) R3_ALLOC(d_winning_starts, n_rows*sizeof(int));
    if (objectives_out) R3_ALLOC(d_start_objectives, n_rows*diagnostic_starts*sizeof(float));


    // Work items: chunk runs of equal query row. The field kernel owns one
    // output row per GROUP of cooperating lanes, so a field block covers
    // COMBO_NTHREADS/(WA*S) rows. A staged analytic arm narrows the block, because
    // the staged library molecules are a per-thread shared allocation and the work items
    // must be chunked to the same width or the tail threads index past it.
    //
    // Under start-per-lane, `start_groups_per_row` groups SHARE a row and
    // split its starts; the field path uses the independently measured combo
    // block size while analytic kernels remain at NTHREADS.
    // This divisor and the kernel's `rgrp = (threadIdx.x/WA)/S` are the same
    // number written twice; if they disagree the kernel silently drops library
    // rows off the end of each work item (divisor too large) or writes rows
    // belonging to the next item twice (too small). Neither faults. Row coverage
    // is asserted directly in tests/test_start_groups_field_cuda.py.
    // WA is a runtime choice, so this must follow it. The constraint above
    // now has a second way to fail: a `chunk` that disagrees with
    // the kernel's rows-per-block drops or duplicates library rows with no
    // error at all.
    const long chunk = field ? (long)COMBO_NTHREADS/
                                   ((long)lanes_per_group*(long)start_groups_per_row)
                             : ((stage > 0) ? (long)stage : (long)NTHREADS);
    std::vector<int> wi_qrow, wi_tstart, wi_tcount;
    if (dense){
      // Each query's rows are the contiguous range [q*n, (q+1)*n), so
      // its work items are written down directly -- the same items the scan
      // below finds in the equivalent pair list, in the same order.
      runtime::Section _s(&runtime::g_stages.index_s);
      const long per_query = n_t ? (n_t + chunk - 1) / chunk : 0;
      wi_qrow.reserve((size_t)(n_q * per_query));
      wi_tstart.reserve((size_t)(n_q * per_query));
      wi_tcount.reserve((size_t)(n_q * per_query));
      for (long q = 0; q < n_q; ++q){
          for (long c = 0; c < n_t; c += chunk){
              wi_qrow.push_back((int)q);
              wi_tstart.push_back((int)(q*n_t + c));
              wi_tcount.push_back((int)std::min(chunk, n_t - c));
          }
      }
    } else { runtime::Section _s(&runtime::g_stages.index_s);
      long i = 0;
      while (i < n_rows){
          int q = ptr_out_q[i];
          long j = i;
          while (j < n_rows && ptr_out_q[j] == q) ++j;
          long run = j - i;
          for (long c = 0; c < run; c += chunk){
              wi_qrow.push_back(q);
              wi_tstart.push_back((int)(i + c));
              wi_tcount.push_back((int)std::min(chunk, run - c));
          }
          i = j;
      }
    }
    int n_wi = (int)wi_qrow.size();

    int *d_wi_q, *d_wi_s, *d_wi_c;
    R3_ALLOC(d_wi_q, n_wi*sizeof(int)); R3_H2D(d_wi_q, wi_qrow.data(),   n_wi*sizeof(int));
    R3_ALLOC(d_wi_s, n_wi*sizeof(int)); R3_H2D(d_wi_s, wi_tstart.data(), n_wi*sizeof(int));
    R3_ALLOC(d_wi_c, n_wi*sizeof(int)); R3_H2D(d_wi_c, wi_tcount.data(), n_wi*sizeof(int));

    // --- the launch ----------------------------------------------------------
    if (field){
        runtime::Section _s(&runtime::g_stages.kernel_s);
        // The three color modes, as one integer. 0 is a shape-only launch
        // and every line of the kernel it reaches is the one that shipped;
        // 1 is AT_POSE (mode 2), 2 is IN_GRADIENT (mode 3).
        const int color_mode = !optim_color ? 0
            : ((color_placement == COLOR_PLACEMENT_AT_POSE) ? 1 : 2);
        // One argument list, two element types. The dtype of the array
        // the caller passed decides which instantiation runs.
        #define R3_LAUNCH_FIELD(PTR) optimize_pairs_field(                     \
            d_molAs, d_molA_num, (int)NmolA,                                   \
            (PTR), d_molB_cnt, (int)NmolB, library_color_offset,               \
            d_out_t, d_wi_q, d_wi_s, d_wi_c, n_wi,                             \
            d_fields,                                                          \
            d_scores, lr_q, lr_t, nsteps, start_mode,                          \
            d_va, d_vb, update_lanes, smem_carveout, device_id,                \
            coord_residency, min_blocks_per_sm, update_rule,                   \
            start_groups_per_row,                                              \
            d_molA_type, d_rmat, d_pmat, N_features,                           \
            mixing_param, legacy_combo_gradient, color_mode, d_color_fields,   \
            lanes_per_group, color_residency, color_stage, interleave,         \
            stream_min_blocks_per_sm, query_color_residency,                   \
            query_shape_width ? query_shape_width : (int)NmolA,                \
            color_ballot, library_shape_width, tile_mode,                   \
            d_winning_starts, d_start_objectives)
        if (lib_i16) R3_LAUNCH_FIELD((const short *)d_molBs);
        else         R3_LAUNCH_FIELD((const float *)d_molBs);
        #undef R3_LAUNCH_FIELD
    } else {
        runtime::Section _s(&runtime::g_stages.kernel_s);
        #define R3_LAUNCH_ANALYTIC(PTR) optimize_pairs_analytic(               \
            d_molAs, d_molA_type, d_molA_num, (int)NmolA,                       \
            (PTR), d_molB_cnt, (int)NmolB, library_color_offset,                \
            d_out_t, d_wi_q, d_wi_s, d_wi_c, n_wi,                              \
            d_rmat, d_pmat, N_features,                                         \
            d_scores, optim_color, lr_q, lr_t, nsteps, mixing_param,            \
            start_mode, device_id, legacy_combo_gradient, d_va, d_vb,           \
            tile_mode, query_vector_load, stage, stage_i16_eff,                 \
            hoist_cross_product, distance_cutoff_sq, omit_shape_weights,        \
            stage_p3_eff, update_rule, color_loop, color_placement,                \
            d_winning_starts, d_start_objectives)
        if (lib_i16) R3_LAUNCH_ANALYTIC((const short *)d_molBs);
        else         R3_LAUNCH_ANALYTIC((const float *)d_molBs);
        #undef R3_LAUNCH_ANALYTIC
    }

    R3_FREE(d_wi_q); R3_FREE(d_wi_s); R3_FREE(d_wi_c);

    // --- the self-overlap bound scan ------------------------------------------
    // Over every row of `d_scores`, before the top-K reductions or the copy-out
    // below, so a row the denominator clamp reports as 0 -- which would rank
    // last and never be returned -- is counted too. It reads the scores and
    // writes only its own two counters; when not asked for it is not launched.
    long long bound_checked = -1, bound_flagged = -1, bound_first = -1;
    if (check_self_overlap_bound){
        unsigned long long * d_bound;
        R3_ALLOC(d_bound, 2*sizeof(unsigned long long));
        unsigned long long flagged = 0;
        { runtime::Section _s(&runtime::g_stages.self_bound_s);
          self_overlap_bound_scan(d_scores, (long)n_rows, DV,
                                  self_overlap_bound_margin, d_bound,
                                  &flagged, &bound_first, device_id); }
        R3_FREE(d_bound);
        bound_checked = (long long)n_rows;
        bound_flagged = (long long)flagged;
    }
    runtime::add_self_work(query_self_evals, library_self_evals,
                           /*library_supplied=*/false,
                           bound_checked, bound_flagged);
    if (stats_out){
        stats_out[CALL_STATS_QUERY_SELF] = query_self_evals;
        stats_out[CALL_STATS_LIBRARY_SELF] = library_self_evals;
        stats_out[CALL_STATS_LIBRARY_SELF_SUPPLIED] = 0;
        stats_out[CALL_STATS_BOUND_CHECKED] = bound_checked;
        stats_out[CALL_STATS_BOUND_FLAGGED] = bound_flagged;
        stats_out[CALL_STATS_BOUND_FIRST_ROW] = bound_first;
    }

    // --- the output ----------------------------------------------------------
    if (winners_out) {
        CUDA_CHECK_ERROR(cudaMemcpy(winners_out, d_winning_starts, n_rows*sizeof(int), cudaMemcpyDeviceToHost));
        runtime::add_bytes(0, (long long)(n_rows*sizeof(int)));
        R3_FREE(d_winning_starts);
    }
    if (objectives_out) {
        CUDA_CHECK_ERROR(cudaMemcpy(objectives_out, d_start_objectives,
            n_rows*diagnostic_starts*sizeof(float), cudaMemcpyDeviceToHost));
        runtime::add_bytes(0, (long long)(n_rows*diagnostic_starts*sizeof(float)));
        R3_FREE(d_start_objectives);
    }

    if (topk_state){
        runtime::Section _s(&runtime::g_stages.topk_s);
        // Dense: d_out_q is null and the top-K takes (query, row) from the
        // launch shape (topk_compact_dense_kernel).
        topk_add_block(topk_state, d_scores, (long)n_rows, d_out_q, d_out_t,
                       (long)n_t, gid_base, max_rows_per_query, device_id);
    } else if (mol_state){
        runtime::Section _s(&runtime::g_stages.topk_s);
        moltopk_add_block(mol_state, d_scores, (long)n_rows, ptr_out_q, ptr_out_t,
                          max_rows_per_query, device_id);
    } else {
        runtime::Section _s(&runtime::g_stages.d2h_s);
        CUDA_CHECK_ERROR(cudaMemcpy(ptr_V, d_scores, n_rows*DV*sizeof(float),
                                    cudaMemcpyDeviceToHost));
        runtime::add_bytes(0, (long long)(n_rows*DV*sizeof(float)));
    }

    R3_FREE(d_molAs); R3_FREE(d_molA_type); R3_FREE(d_molA_num);
    R3_FREE(d_molBs); R3_FREE(d_molB_cnt); R3_FREE(d_molA_cnt);
    R3_FREE(d_rmat);  R3_FREE(d_pmat);
    R3_FREE(d_va);    R3_FREE(d_vb);
    R3_FREE(d_out_q); R3_FREE(d_out_t); R3_FREE(d_scores);
    // The fields are NOT freed here: the cache owns them, and the next call
    // with the same query panel reuses them. They are released when a different
    // panel evicts the entry, or by `field_cache_clear()`. `field_from_cache`
    // is true on both paths by this point -- set on a hit, and set again after
    // publishing a fresh build -- so this is a guard against a future edit that
    // adds a path leaving fields unowned, not dead code.
    if (field && !field_from_cache){
        runtime::Section _s(&runtime::g_stages.free_s);
        for (size_t q = 0; q < field_texs.size(); ++q){
            if (field_texs[q])   cudaDestroyTextureObject(field_texs[q]);
            if (field_arrays[q]) cudaFreeArray(field_arrays[q]);
        }
        if (d_fields) runtime::dfree(d_fields);
        for (size_t c = 0; c < cfield_texs.size(); ++c){
            if (cfield_texs[c])   cudaDestroyTextureObject(cfield_texs[c]);
            if (cfield_arrays[c]) cudaFreeArray(cfield_arrays[c]);
        }
        if (d_color_fields) runtime::dfree(d_color_fields);
    }

    #undef R3_ALLOC
    #undef R3_H2D
    #undef R3_FREE
}


////////////////////////////////////////////////////////////////////////////////
/// wrapper functions for testing framework
////////////////////////////////////////////////////////////////////////////////

/// @brief shape overlap volume of every (A, B) pair; mirrors the reference implementation.test_overlap
void test_overlap(py::array_t<float> A, py::array_t<float> B, py::array_t<float> V){
    auto molAs = A.unchecked<3>();
    auto molBs = B.unchecked<3>();
    auto molV  = V.mutable_unchecked<2>();

    { runtime::Section _s(&runtime::g_stages.kernel_s);
      test_overlap_gpu(molAs.data(0,0,0), molAs.shape(1), molAs.shape(0),
                     molBs.data(0,0,0), molBs.shape(1), molBs.shape(0),
                     molV.mutable_data(0,0)); }
}

/// @brief shape overlap gradient of one pair; mirrors the reference implementation.test_gradient
std::array<float,7> test_gradient(py::array_t<float> A, py::array_t<float> B,
                                  py::array_t<float> q, py::array_t<float> t){
    auto molA = A.unchecked<2>();
    auto molB = B.unchecked<2>();

    std::array<float,7> gradient = {0,0,0,0,0,0,0};
    { runtime::Section _s(&runtime::g_stages.kernel_s);
      test_gradient_gpu(molA.data(0,0), molA.shape(0),
                      molB.data(0,0), molB.shape(0),
                      q.unchecked<1>().data(0), t.unchecked<1>().data(0),
                      gradient.data()); }
    return gradient;
}




////////////////////////////////////////////////////////////////////////////////
/// The build stamp.
///
/// ROSHAMBO2 baked this into its always-built C++/OpenMP extension. That
/// extension is gone, so it lives here -- which means a CUDA-less install has no
/// stamp at all, and `roshambo3._build_info.get_build_info()` reports
/// `with_cuda=False` rather than raising. The benchmark harness must require
/// BOTH `ArithmeticCheck.ok` and `checked_a_gpu_build`: "no forbidden flag was
/// found" is vacuously true when there is nothing to inspect.
////////////////////////////////////////////////////////////////////////////////

// Which SOURCE was compiled, as opposed to which commit was checked out.
// Written by scripts/source_stamp.py into the build tree on every build, and
// picked up here with __has_include so a compile driven by something other than
// this project's CMake still builds -- and reports "unknown" rather than
// silently reporting clean.
#if defined(__has_include)
#  if __has_include("roshambo3_source_stamp.h")
#    include "roshambo3_source_stamp.h"
#  endif
#endif

#ifndef ROSHAMBO3_BUILD_TYPE
#define ROSHAMBO3_BUILD_TYPE "unknown"
#endif
#ifndef ROSHAMBO3_GIT_SHA
#define ROSHAMBO3_GIT_SHA "unknown"
#endif
#ifndef ROSHAMBO3_CUDA_SRC_SHA256
#define ROSHAMBO3_CUDA_SRC_SHA256 "unknown"
#endif
#ifndef ROSHAMBO3_CUDA_SRC_TREE
#define ROSHAMBO3_CUDA_SRC_TREE "unknown"
#endif
// Tri-state, and it has to be: 1 dirty, 0 clean, -1 nobody looked. A boolean
// here would make an unstamped build indistinguishable from a clean one, which
// is the same defect that was fixed for `checked_a_gpu_build` and the
// architecture probe.
#ifndef ROSHAMBO3_CUDA_SRC_DIRTY
#define ROSHAMBO3_CUDA_SRC_DIRTY (-1)
#endif
#ifndef ROSHAMBO3_CUDA_SRC_DIRTY_FILES
#define ROSHAMBO3_CUDA_SRC_DIRTY_FILES ""
#endif
#ifndef ROSHAMBO3_SOURCE_STAMP_PHASE
#define ROSHAMBO3_SOURCE_STAMP_PHASE "none"
#endif
#ifndef ROSHAMBO3_CUDA_ARCHITECTURES
#define ROSHAMBO3_CUDA_ARCHITECTURES "unknown"
#endif
#ifndef ROSHAMBO3_NVCC_VERSION
#define ROSHAMBO3_NVCC_VERSION "unknown"
#endif
#ifndef ROSHAMBO3_CXX_FLAGS
#define ROSHAMBO3_CXX_FLAGS "unknown"
#endif
#ifndef ROSHAMBO3_CUDA_FLAGS
#define ROSHAMBO3_CUDA_FLAGS "unknown"
#endif
#ifndef ROSHAMBO3_NTHREADS
#define ROSHAMBO3_NTHREADS 256
#endif
#ifndef ROSHAMBO3_FINAL_EXACT_SHAPE
#define ROSHAMBO3_FINAL_EXACT_SHAPE 1
#endif
#ifndef ROSHAMBO3_FINAL_EXACT_COLOR
#define ROSHAMBO3_FINAL_EXACT_COLOR 1
#endif



// kernels.cu's launch macros call this with what the driver granted.
// Declared in kernels.cuh, defined here because the stats block lives here.
// Guarded by the same mutex the stage timers use -- a screen runs the reduction
// on one thread and the producers on others.
void r3_record_launch(int blocks_per_sm, unsigned long long shared_bytes,
                      int carveout, int nthreads)
{
    std::lock_guard<std::mutex> held(runtime::g_stage_mu);
    runtime::LaunchFacts & f = runtime::g_launch;
    f.blocks_per_sm = blocks_per_sm;
    f.shared_bytes  = (long long)shared_bytes;
    f.carveout      = carveout;
    f.nthreads      = nthreads;
    if (f.n_launches == 0){
        f.blocks_min = f.blocks_max = blocks_per_sm;
        f.shared_min = f.shared_max = (long long)shared_bytes;
    } else {
        if (blocks_per_sm < f.blocks_min) f.blocks_min = blocks_per_sm;
        if (blocks_per_sm > f.blocks_max) f.blocks_max = blocks_per_sm;
        if ((long long)shared_bytes < f.shared_min) f.shared_min = (long long)shared_bytes;
        if ((long long)shared_bytes > f.shared_max) f.shared_max = (long long)shared_bytes;
    }
    f.n_launches += 1;
}

PYBIND11_MODULE(_roshambo3_cuda, m) {
    m.def("compiled_wire_step", []() { return ROSHAMBO3_STAGE_INV_Q_XYZ; });
    m.def("compiled_wire_inverse_step", []() { return ROSHAMBO3_STAGE_Q_XYZ; });
    m.doc() = "ROSHAMBO3 CUDA backend. See doc/NUMERICS_POLICY.md before "
              "changing anything that produces a score.";

    m.def("optimize_pairs", &optimize_pairs,
          "Score an explicit list of (query row, library row) pairs on one device.",
          py::arg("A"), py::arg("AT"), py::arg("AN"),
          py::arg("B"), py::arg("BCNT"),
          py::arg("RMAT"), py::arg("PMAT"),
          py::arg("OUT_Q"), py::arg("OUT_T"), py::arg("V"),
          py::arg("optim_color"), py::arg("mixing_param"),
          py::arg("lr_q"), py::arg("lr_t"), py::arg("nsteps"),
          py::arg("start_mode"),
          // 0 = FIELD (production), 1 = ANALYTIC (the Tier-3 specification).
          py::arg("pose_search") = 0,
          py::arg("device_id") = 0,
          py::arg("legacy_combo_gradient") = false,
          // The analytic kernel's staging switches. Defaults are the shipped
          // configuration; anything else is a measurement arm and belongs
          // behind roshambo3.experimental.
          py::arg("query_vector_load") = true,
          py::arg("stage_width") = 256,
          py::arg("stage_int16") = true,
          py::arg("hoist_cross_product") = true,
          py::arg("distance_cutoff_sq") = 0.0f,
          py::arg("omit_shape_weights") = true,
          py::arg("stage_xyz_only") = true,
          py::arg("update_rule") = 0,
          // `color_loop` names the color pair loop's structure and
          // defaults to the measured winner; `color_placement`
          // is 0 for the combo-optimal mode 3 (the earlier default) and
          // 1 for its mode 2, the pose on shape with the color term evaluated
          // once at the converged pose. Both are runtime.
          py::arg("color_loop") = COLOR_LOOP_DEFAULT,
          py::arg("color_placement") = COLOR_PLACEMENT_IN_GRADIENT,
          // 0 is the all-analytic color arm; a positive value builds a
          // field for every (query, plane) whose query-side count clears it.
          py::arg("color_field_min_sources") = 0,
          py::arg("color_field_spacing_a") = 0.0f,
          // The lane layout and register budget.
          py::arg("lanes_per_group") = 4,
          py::arg("color_residency") = 1,
          py::arg("color_stage") = 0,
          py::arg("query_color_residency") = 0,
          py::arg("query_shape_width") = 0,
          py::arg("library_shape_width") = 0,
          py::arg("library_color_offset") = 0,
          py::arg("color_ballot") = 0,
          py::arg("interleave") = 0,
          py::arg("stream_min_blocks_per_sm") = 0,
          // The field kernel. field_spacing_a is the lattice spacing in
          // ANGSTROM (the reduced-frame spacing is COORD_SCALE * h).
          // smem_carveout is the shared/L1 split and is ALWAYS set explicitly:
          // cudaFuncSetAttribute is sticky, so "unset" is not a control.
          py::arg("field_spacing_a") = 0.8f,
          py::arg("field_pad_a") = 8.0f,
          py::arg("update_lanes") = 2,
          py::arg("smem_carveout") = 50,
          py::arg("coord_residency") = 0,
          py::arg("min_blocks_per_sm") = 4,
          // Direct extension callers retain the historical general default.
          // The Python API chooses the selected mode-3 operating point.
          py::arg("start_groups_per_row") = 1,
          // Pass DeviceTopK.handle and V is never written.
          py::arg("topk") = (std::uintptr_t)0,
          py::arg("gid_base") = (long long)0,
          py::arg("max_rows_per_query") = (long)0,
          // Pass DeviceMoleculeTopK.handle instead, having staged this block's
          // mol_id sidecar slice first.
          py::arg("topk_molecule") = (std::uintptr_t)0,
          py::arg("library_tile_mode") = 1,
          // Off by default, and off launches nothing. The margin
          // has no value here: it is defined and chosen only in Python
          // (roshambo3.scores.SELF_OVERLAP_MARGIN_POLICY), and
          // backend.cuda.score_pairs always passes it. The NaN default makes a
          // direct caller that turns the check on without a margin fail the
          // finiteness check above instead of using a stale copy.
          py::arg("check_self_overlap_bound") = false,
          py::arg("self_overlap_bound_margin") =
              std::numeric_limits<double>::quiet_NaN(),
          py::arg("call_stats") = py::none(),
          py::arg("winning_starts") = py::none(),
          py::arg("start_objectives") = py::none());

    // ---- the device-resident top-K -----------------------------------------
    // One object per (screen, device). Hand `.handle` to
    // optimize_pairs(topk=...) and the dense score block never crosses the bus.
    // Multi-GPU is one state per device, merged on the host with
    // numerics.rank_order -- the state has its own atomics and cannot be shared.
    py::class_<TopKHandle>(m, "DeviceTopK")
        .def(py::init<int,int,int,int,bool,int>(),
             py::arg("n_query"), py::arg("k"), py::arg("rank_col") = 0,
             py::arg("dv") = 20, py::arg("keep_cols") = true, py::arg("device_id") = 0)
        .def_property_readonly("handle", [](TopKHandle & s){
             return (std::uintptr_t)s.p; })
        .def_property_readonly("n_query", [](TopKHandle & s){ return topk_n_query(s.p); })
        .def_property_readonly("k", [](TopKHandle & s){ return topk_k(s.p); })
        .def_property_readonly("capacity_bytes", [](TopKHandle & s){
             return topk_capacity_bytes(s.p); })
        .def("flush", [](TopKHandle & s, int device_id){
             py::gil_scoped_release _n; topk_flush(s.p, device_id); },
             py::arg("device_id") = 0)
        .def("thresholds", [](TopKHandle & s){
             const int nq = topk_n_query(s.p);
             py::array_t<float> out((py::ssize_t)nq);
             topk_thresholds(s.p, out.mutable_data());
             return out;
         }, "the running per-query K-th score (-inf until K rows are held)")
        .def("counters", [](TopKHandle & s){
             long long o=0,a=0,n=0,b=0;
             topk_counters(s.p, &o, &a, &n, &b);
             py::dict d;
             d["rows_offered"] = o; d["rows_admitted"] = a;
             d["n_select"] = n; d["bytes_d2h"] = b;
             return d;
         })
        .def("result", [](TopKHandle & s, int device_id){
             const int nq = topk_n_query(s.p), k = topk_k(s.p), dv = topk_dv(s.p);
             py::array_t<float> scores({(py::ssize_t)nq, (py::ssize_t)k, (py::ssize_t)dv});
             py::array_t<long long> ids({(py::ssize_t)nq, (py::ssize_t)k});
             std::vector<int> counts((size_t)nq, 0);
             { py::gil_scoped_release _n;
               topk_result(s.p, scores.mutable_data(), ids.mutable_data(),
                           counts.data(), device_id); }
             py::list out;
             for (int q = 0; q < nq; ++q){
                 py::ssize_t n = (py::ssize_t)counts[(size_t)q];
                 py::array_t<float> sq({n, (py::ssize_t)dv});
                 py::array_t<long long> iq(n);
                 if (n){
                     std::memcpy(sq.mutable_data(), scores.data(q,0,0),
                                 (size_t)n*dv*sizeof(float));
                     std::memcpy(iq.mutable_data(), ids.data(q,0),
                                 (size_t)n*sizeof(long long));
                 }
                 out.append(py::make_tuple(sq, iq));
             }
             return out;
         }, py::arg("device_id") = 0,
         "flush, then return [(scores[n,dv], ids[n])] per query in canonical "
         "order (descending score, ties by ascending conformer id)");

    // ---- the molecule-level top-K ------------------------------------------
    // A molecule scores as the best of its conformers. Stage each block's
    // mol_id sidecar slice, then hand `.handle` to
    // optimize_pairs(topk_molecule=...). See molecule_topk.cuh for the
    // segmented argmax, the boundary carry and the tie-break.
    py::class_<MoleculeTopKHandle>(m, "DeviceMoleculeTopK")
        .def(py::init<int,int,int,int,bool,int,long>(),
             py::arg("n_query"), py::arg("k"), py::arg("rank_col") = 0,
             py::arg("dv") = 20, py::arg("keep_cols") = true,
             py::arg("device_id") = 0, py::arg("max_block_conformers") = 4096)
        .def_property_readonly("handle", [](MoleculeTopKHandle & s){
             return (std::uintptr_t)s.p; })
        .def_property_readonly("n_query", [](MoleculeTopKHandle & s){
             return topk_n_query(moltopk_inner(s.p)); })
        .def_property_readonly("k", [](MoleculeTopKHandle & s){
             return topk_k(moltopk_inner(s.p)); })
        .def_property_readonly("capacity_bytes", [](MoleculeTopKHandle & s){
             return topk_capacity_bytes(moltopk_inner(s.p)); })
        .def("stage_block", [](MoleculeTopKHandle & s,
                               py::array_t<long long, py::array::c_style
                                           | py::array::forcecast> mol_id,
                               long long gid_base, bool tail_open,
                               long long base_gid, int device_id){
             if (mol_id.ndim() != 1)
                 throw std::runtime_error("roshambo3.cuda: mol_id must be 1-D");
             const long n = (long)mol_id.shape(0);
             const long long * ptr = mol_id.data(0);
             py::gil_scoped_release _n;
             moltopk_stage_block(s.p, ptr, n, gid_base, tail_open ? 1 : 0,
                                 base_gid, device_id);
         }, py::arg("mol_id"), py::arg("gid_base"), py::arg("tail_open"),
            py::arg("base_gid"), py::arg("device_id") = 0,
            "stage one block's mol_id sidecar slice, before the launch that "
            "scores it")
        .def("last_block", [](MoleculeTopKHandle & s){
             int n_seg = 0, tail = 0;
             moltopk_last_block(s.p, &n_seg, &tail);
             py::dict d;
             d["n_segments"] = n_seg;
             d["tail_segment_start"] = tail;
             return d;
         }, "the device's own reading of the last staged block's segmentation, "
            "so the caller can check it against the sidecar")
        .def("finish", [](MoleculeTopKHandle & s, int device_id){
             py::gil_scoped_release _n; moltopk_finish(s.p, device_id); },
             py::arg("device_id") = 0,
             "refuse to finish with a molecule still open across a boundary")
        .def("flush", [](MoleculeTopKHandle & s, int device_id){
             py::gil_scoped_release _n; topk_flush(moltopk_inner(s.p), device_id); },
             py::arg("device_id") = 0)
        .def("thresholds", [](MoleculeTopKHandle & s){
             DeviceTopKState * in = moltopk_inner(s.p);
             const int nq = topk_n_query(in);
             py::array_t<float> out((py::ssize_t)nq);
             topk_thresholds(in, out.mutable_data());
             return out;
         })
        .def("counters", [](MoleculeTopKHandle & s){
             long long o=0,a=0,n=0,b=0, nc=0, nm=0, ncar=0, nblk=0;
             topk_counters(moltopk_inner(s.p), &o, &a, &n, &b);
             moltopk_counters(s.p, &nc, &nm, &ncar, &nblk);
             py::dict d;
             d["rows_offered"] = o; d["rows_admitted"] = a;
             d["n_select"] = n; d["bytes_d2h"] = b;
             d["conformers_reduced"] = nc; d["molecule_rows_emitted"] = nm;
             d["blocks_ending_mid_molecule"] = ncar; d["n_blocks"] = nblk;
             return d;
         })
        .def("result", [](MoleculeTopKHandle & s, int device_id){
             DeviceTopKState * in = moltopk_inner(s.p);
             const int nq = topk_n_query(in), k = topk_k(in), dv = topk_dv(in);
             py::array_t<float> scores({(py::ssize_t)nq, (py::ssize_t)k,
                                        (py::ssize_t)dv});
             py::array_t<long long> ids({(py::ssize_t)nq, (py::ssize_t)k});
             std::vector<int> counts((size_t)nq, 0);
             { py::gil_scoped_release _n;
               moltopk_finish(s.p, device_id);
               topk_result(in, scores.mutable_data(), ids.mutable_data(),
                           counts.data(), device_id); }
             py::list out;
             for (int q = 0; q < nq; ++q){
                 py::ssize_t n = (py::ssize_t)counts[(size_t)q];
                 py::array_t<float> sq({n, (py::ssize_t)dv});
                 py::array_t<long long> iq(n);
                 if (n){
                     std::memcpy(sq.mutable_data(), scores.data(q,0,0),
                                 (size_t)n*dv*sizeof(float));
                     std::memcpy(iq.mutable_data(), ids.data(q,0),
                                 (size_t)n*sizeof(long long));
                 }
                 out.append(py::make_tuple(sq, iq));
             }
             return out;
         }, py::arg("device_id") = 0,
         "flush, then return [(scores[n,dv], winning_gids[n])] per query in "
         "canonical order -- one row per molecule, the gid naming the conformer "
         "that represents it");

    // The bound scan on a caller-supplied score block, so its
    // predicate can be tested on planted rows (an inflated Tanimoto, a clamped
    // denominator, the margin's edge) that consistent inputs never produce.
    m.def("self_overlap_bound_scan", [](py::array_t<float, py::array::c_style> V,
                                        double margin, int device_id){
        if (V.ndim() != 2)
            throw std::runtime_error("roshambo3.cuda: expected [n_rows, dv] float32");
        const long n = (long)V.shape(0);
        const int dv = (int)V.shape(1);
        const float * h = V.data();
        unsigned long long flagged = 0; long long first = -1;
        {
            py::gil_scoped_release _n;
            ScopedDevice on_device(device_id);
            float * d_v = nullptr; unsigned long long * d_c = nullptr;
            const size_t bytes = (size_t)std::max<long>(n, 1) * dv * sizeof(float);
            CUDA_CHECK_ERROR(cudaMalloc((void **)&d_v, bytes));
            CUDA_CHECK_ERROR(cudaMalloc((void **)&d_c, 2*sizeof(unsigned long long)));
            if (n) CUDA_CHECK_ERROR(cudaMemcpy(d_v, h, (size_t)n*dv*sizeof(float),
                                               cudaMemcpyHostToDevice));
            try {
                self_overlap_bound_scan(d_v, n, dv, margin, d_c, &flagged,
                                        &first, device_id);
            } catch (...) { cudaFree(d_v); cudaFree(d_c); throw; }
            cudaFree(d_v); cudaFree(d_c);
        }
        return py::make_tuple((long long)flagged, first);
    }, py::arg("V"), py::arg("margin"), py::arg("device_id") = 0,
       "count rows of a score block past the self-overlap bound on the "
       "device; returns (n_flagged, first_flagged_row or -1)");

    // Diagnostic: observe the actual production prefilter builder. No
    // alternate Gaussian formula or texture arithmetic is introduced here.
    m.def("numerics_build_field", [](py::array_t<float, py::array::c_style> query,
            std::array<float,3> origin, float spacing, std::array<int,3> cells,
            bool shape, float fac) {
        if (query.ndim()!=2 || query.shape(1)!=D_HOST || query.shape(0)<1 ||
            query.shape(0)>std::numeric_limits<int>::max())
            throw std::invalid_argument("query must be nonempty [n,4]");
        if (!std::isfinite(spacing) || spacing<=0 || !std::isfinite(fac) || fac<=0)
            throw std::invalid_argument("spacing and Gaussian factor must be positive finite");
        size_t count=1;
        for (int k=0;k<3;++k) {
            if (!std::isfinite(origin[k]) || cells[k]<1 || cells[k]>128)
                throw std::invalid_argument("finite origin and 1..128 cells per axis required");
            count*=size_t(cells[k]+1);
            if (!std::isfinite(origin[k]+float(cells[k])*spacing))
                throw std::invalid_argument("field extent must be finite");
        }
        for (py::ssize_t i=0;i<query.size();++i)
            if (!std::isfinite(query.data()[i]))
                throw std::invalid_argument("query must be finite");
        struct Resources {
            float* query=nullptr; void* nodes=nullptr;
            ~Resources() { if(query) cudaFree(query); if(nodes) cudaFree(nodes); }
        } resources;
        py::array_t<uint16_t> output({py::ssize_t(cells[2]+1),py::ssize_t(cells[1]+1),
                                     py::ssize_t(cells[0]+1),py::ssize_t(4)});
        CUDA_CHECK_ERROR(cudaMalloc((void**)&resources.query,size_t(query.size())*sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&resources.nodes,count*8));
        CUDA_CHECK_ERROR(cudaMemcpy(resources.query,query.data(),size_t(query.size())*sizeof(float),cudaMemcpyHostToDevice));
        FieldGeometry geometry{origin[0],origin[1],origin[2],spacing,cells[0],cells[1],cells[2]};
        int device=0;
        CUDA_CHECK_ERROR(cudaGetDevice(&device));
        if (shape) build_query_field(resources.query,int(query.shape(0)),int(query.shape(0)),geometry,resources.nodes,device);
        else build_query_color_field(resources.query,0,int(query.shape(0)),int(query.shape(0)),geometry,fac,resources.nodes,device);
        CUDA_CHECK_ERROR(cudaMemcpy(output.mutable_data(),resources.nodes,count*8,cudaMemcpyDeviceToHost));
        return output;
    }, py::arg("query"),py::arg("origin_reduced"),py::arg("spacing_reduced"),
       py::arg("cells"),py::arg("shape")=true,py::arg("factor")=1.f);

    m.def("numerics_sample_field", [](py::array_t<uint16_t, py::array::c_style> nodes,
                                          py::array_t<float, py::array::c_style> points,
                                          std::array<float,3> origin, float spacing) {
        if (!std::isfinite(spacing) || spacing<=0 || !std::isfinite(1.f/spacing))
            throw std::invalid_argument("positive finite reciprocal spacing required");
        for (float v:origin) if (!std::isfinite(v))
            throw std::invalid_argument("finite sampler origin required");
        if (nodes.ndim()!=4 || nodes.shape(3)!=4 || nodes.shape(0)<2 ||
            nodes.shape(1)<2 || nodes.shape(2)<2 || points.ndim()!=2 || points.shape(1)!=3)
            throw std::invalid_argument("nodes must be [nz,ny,nx,4], points [n,3]");
        for (int axis=0;axis<3;++axis)
            if (nodes.shape(axis)>std::numeric_limits<int>::max())
                throw std::invalid_argument("sampler grid dimension exceeds int capacity");
        if (points.shape(0)>std::numeric_limits<int>::max() ||
            size_t(points.shape(0))>std::numeric_limits<size_t>::max()/(4*sizeof(float)))
            throw std::invalid_argument("sampler point count exceeds allocation capacity");
        for (py::ssize_t i=0;i<points.size();++i)
            if (!std::isfinite(points.data()[i]))
                throw std::invalid_argument("sampler points must be finite");
        const int nx=nodes.shape(2), ny=nodes.shape(1), nz=nodes.shape(0), n=points.shape(0);
        py::array_t<float> output({py::ssize_t(n),py::ssize_t(4)});
        if (!n) return output;
        struct Resources {
            cudaArray_t arr=nullptr;
            cudaTextureObject_t tex=0;
            float *dp=nullptr,*ds=nullptr;
            ~Resources() {
                if (dp) cudaFree(dp);
                if (ds) cudaFree(ds);
                if (tex) cudaDestroyTextureObject(tex);
                if (arr) cudaFreeArray(arr);
            }
        } resources;
        auto& arr=resources.arr;
        auto& tex=resources.tex;
        auto& dp=resources.dp;
        auto& ds=resources.ds;
        const auto ch=cudaCreateChannelDescHalf4();
        CUDA_CHECK_ERROR(cudaMalloc3DArray(&arr,&ch,make_cudaExtent(nx,ny,nz)));
        cudaMemcpy3DParms cp{};
        cp.srcPtr=make_cudaPitchedPtr(nodes.mutable_data(),size_t(nx)*8,nx,ny);
        cp.dstArray=arr; cp.extent=make_cudaExtent(nx,ny,nz); cp.kind=cudaMemcpyHostToDevice;
        CUDA_CHECK_ERROR(cudaMemcpy3D(&cp));
        cudaResourceDesc rd{}; rd.resType=cudaResourceTypeArray; rd.res.array.array=arr;
        cudaTextureDesc td{};
        td.addressMode[0]=td.addressMode[1]=td.addressMode[2]=cudaAddressModeBorder;
        td.filterMode=cudaFilterModeLinear; td.readMode=cudaReadModeElementType;
        CUDA_CHECK_ERROR(cudaCreateTextureObject(&tex,&rd,&td,nullptr));
        CUDA_CHECK_ERROR(cudaMalloc(&dp,size_t(n)*3*sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&ds,size_t(n)*4*sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemcpy(dp,points.data(),size_t(n)*3*sizeof(float),cudaMemcpyHostToDevice));
        QueryField field{}; field.tex=tex; field.inv_hs=1.f/spacing;
        field.ox=origin[0]; field.oy=origin[1]; field.oz=origin[2];
        field.nx=nx-1; field.ny=ny-1; field.nz=nz-1;
        numerics_sample_field_gpu(field,dp,n,ds);
        CUDA_CHECK_ERROR(cudaMemcpy(output.mutable_data(),ds,size_t(n)*4*sizeof(float),cudaMemcpyDeviceToHost));
        return output;
    }, py::arg("half4_node_bits"), py::arg("lattice_points"),
       py::arg("origin_reduced")=std::array<float,3>{0.f,0.f,0.f},
       py::arg("spacing_reduced")=1.f);
    m.def("test_overlap", &test_overlap,
          "shape overlap volume of every (A, B) pair; for checking against the "
          "float64 reference in roshambo3.reference");
    m.def("test_gradient", &test_gradient,
          "quaternion gradient of a single overlap, for the same purpose");

    // ---- the device half of the stage breakdown ----------------------------
    m.def("stage_timing_enable", [](bool on){ runtime::g_enabled = on; },
          "enable/disable device-side stage accounting", py::arg("on"));
    m.def("stage_timing_reset", [](){ runtime::g_stages = runtime::StageTimers();
                                     runtime::g_launch = runtime::LaunchFacts(); },
          "zero the device-side stage counters");
    m.def("field_cache_clear", [](){
        return field_cache_clear_all();
    }, "release every cached query field; returns how many entries were held");
    m.def("field_cache_stats", [](){
        std::lock_guard<std::mutex> g(g_field_cache_mu);
        py::dict d;
        d["hits"] = g_field_cache_hits;
        d["misses"] = g_field_cache_misses;
        long long held = 0;
        for (auto & kv : g_field_cache) if (kv.second.valid) ++held;
        d["entries"] = held;
        return d;
    }, "query-field cache hits, misses and entries currently held");

    m.def("launch_facts", [](){
        py::dict d;
        d["blocks_per_sm"] = runtime::g_launch.blocks_per_sm;
        d["shared_bytes"]  = runtime::g_launch.shared_bytes;
        d["carveout"]      = runtime::g_launch.carveout;
        d["nthreads"]      = runtime::g_launch.nthreads;
        d["blocks_min"]    = runtime::g_launch.blocks_min;
        d["blocks_max"]    = runtime::g_launch.blocks_max;
        d["shared_min"]    = runtime::g_launch.shared_min;
        d["shared_max"]    = runtime::g_launch.shared_max;
        d["n_launches"]    = runtime::g_launch.n_launches;
        return d;
    }, "cudaOccupancyMaxActiveBlocksPerMultiprocessor, the shared request "
       "and the carveout of the last pair-kernel launch, and the envelope over "
       "every launch since the last stage_timing_reset(). carveout is -1 where "
       "the launcher does not set one -- which the ANALYTIC launcher does not, "
       "so the driver infers the minimum partition that fits one block and the "
       "staged analytic path is at one block per SM whatever its demand.");

    m.def("stage_timings", [](){
        py::dict d;
        d["alloc_s"]      = runtime::g_stages.alloc_s;
        d["h2d_s"]        = runtime::g_stages.h2d_s;
        d["kernel_s"]     = runtime::g_stages.kernel_s;
        d["precompute_s"] = runtime::g_stages.precompute_s;
        d["d2h_s"]        = runtime::g_stages.d2h_s;
        d["free_s"]       = runtime::g_stages.free_s;
        d["index_s"]      = runtime::g_stages.index_s;
        d["topk_s"]       = runtime::g_stages.topk_s;
        d["h2d_bytes"]    = runtime::g_stages.h2d_bytes;
        d["d2h_bytes"]    = runtime::g_stages.d2h_bytes;
        d["n_calls"]      = runtime::g_stages.n_calls;
        d["query_self_evaluations"]      = runtime::g_stages.query_self_evaluations;
        d["library_self_evaluations"]    = runtime::g_stages.library_self_evaluations;
        d["library_self_supplied_calls"] = runtime::g_stages.library_self_supplied_calls;
        d["self_bound_rows_checked"]     = runtime::g_stages.self_bound_rows_checked;
        d["self_bound_rows_flagged"]     = runtime::g_stages.self_bound_rows_flagged;
        d["self_bound_s"]                = runtime::g_stages.self_bound_s;
        d["enabled"]      = runtime::g_enabled;
        return d;
    }, "accumulated device-side stage times since the last reset");

    // ---- the persistent device arena ---------------------------------------
    m.def("arena_enable", [](bool on){ runtime::g_arena_enabled = on; },
          "route device allocations through the caching allocator (default on). "
          "Off restores per-call cudaMalloc/cudaFree.", py::arg("on"));
    m.def("arena_trim", [](){ runtime::arena_trim(); },
          "return every cached device block to the driver");
    m.def("arena_stats", [](){
        py::list out;
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        for (int d = 0; d < n_dev && d < ROSHAMBO3_MAX_DEVICES; ++d){
            py::dict e;
            e["device"]      = d;
            e["hits"]        = runtime::g_cache[d].hits;
            e["misses"]      = runtime::g_cache[d].misses;
            e["held_bytes"]  = (long long)runtime::g_cache[d].held_bytes;
            e["free_blocks"] = (long long)runtime::g_cache[d].free_blocks.size();
            e["live_blocks"] = (long long)runtime::g_cache[d].live.size();
            out.append(e);
        }
        return out;
    }, "per-device caching-allocator hit/miss counts and bytes held");

    // ---- pinned host staging -----------------------------------------------
    // A numpy array whose storage is page-locked. The point is not only
    // bandwidth: a copy out of pageable memory cannot be asynchronous, so for
    // as long as the producer writes into ordinary numpy arrays there is no
    // version of this pipeline in which H2D overlaps anything at all. Handing
    // the producer a pinned buffer to write into costs nothing extra -- it has
    // to write somewhere.
    m.def("pinned_empty_f32", [](std::vector<py::ssize_t> shape){
        size_t n = 1;
        for (auto s : shape) n *= (size_t)s;
        void * p = nullptr;
        CUDA_CHECK_ERROR(cudaHostAlloc(&p, n * sizeof(float), cudaHostAllocPortable));
        py::capsule owner(p, [](void * q){ cudaFreeHost(q); });
        return py::array_t<float>(shape, (float *)p, owner);
    }, "page-locked float32 array", py::arg("shape"));
    // The library wire record's element type. The pool stages int16 now,
    // and a pinned float32 buffer viewed as int16 would work but would hide the
    // element count in a reinterpretation -- the sizes downstream are in
    // ELEMENTS, so the allocator should be too.
    m.def("pinned_empty_i16", [](std::vector<py::ssize_t> shape){
        size_t n = 1;
        for (auto s : shape) n *= (size_t)s;
        void * p = nullptr;
        CUDA_CHECK_ERROR(cudaHostAlloc(&p, n * sizeof(short), cudaHostAllocPortable));
        py::capsule owner(p, [](void * q){ cudaFreeHost(q); });
        return py::array_t<short>(shape, (short *)p, owner);
    }, "page-locked int16 array", py::arg("shape"));
    m.def("pinned_empty_i32", [](std::vector<py::ssize_t> shape){
        size_t n = 1;
        for (auto s : shape) n *= (size_t)s;
        void * p = nullptr;
        CUDA_CHECK_ERROR(cudaHostAlloc(&p, n * sizeof(int), cudaHostAllocPortable));
        py::capsule owner(p, [](void * q){ cudaFreeHost(q); });
        return py::array_t<int>(shape, (int *)p, owner);
    }, "page-locked int32 array", py::arg("shape"));
    m.def("is_pinned", [](py::array a){
        cudaPointerAttributes at{};
        if (cudaPointerGetAttributes(&at, a.data()) != cudaSuccess){
            cudaGetLastError();
            return false;
        }
        return at.type == cudaMemoryTypeHost;
    }, "whether an array's storage is page-locked", py::arg("a"));

    // ---- device capabilities ----------------------------------------------
    // The shared-memory limit is the one that has to be asked rather than
    // assumed. It is 64 KB on Turing, 99 KB on sm_86/89/120, 163 KB on sm_80 and
    // 227 KB on sm_90 -- so a per-bucket staging policy with a hardcoded budget
    // is correct on exactly one architecture and over-requests on Turing, where
    // the launcher then (correctly, but avoidably) refuses the launch.
    m.def("device_info", [](int device_id){
        int n_dev = 0;
        CUDA_CHECK_ERROR(cudaGetDeviceCount(&n_dev));
        if (device_id < 0 || device_id >= n_dev)
            throw std::runtime_error("roshambo3.cuda: device_id out of range");
        cudaDeviceProp prop{};
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device_id));
        int optin = 0;
        CUDA_CHECK_ERROR(cudaDeviceGetAttribute(
            &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
        py::dict d;
        d["device"] = device_id;
        d["name"] = std::string(prop.name);
        d["compute_capability"] = py::make_tuple(prop.major, prop.minor);
        d["multiprocessor_count"] = prop.multiProcessorCount;
        d["shared_memory_per_block_optin"] = optin;
        d["shared_memory_per_block_default"] = (int)prop.sharedMemPerBlock;
        d["shared_memory_per_multiprocessor"] = (int)prop.sharedMemPerMultiprocessor;
        d["total_global_memory"] = (long long)prop.totalGlobalMem;
        d["warp_size"] = prop.warpSize;
        return d;
    }, "device capabilities that a launch policy has to respect",
       py::arg("device_id") = 0);

    // ---- the grid path's signed-permutation precondition -------------------
    // Exported so a test exercises the launch guard's own body rather than a
    // reimplementation of it. `optimize_pairs_field` calls
    // require_signed_permutation_starts on every launch; see the block comment
    // over start_quaternion in kernels.cu.
    m.def("check_start_rotations", [](int start_mode){
        return signed_permutation_start_count(start_mode);
    }, "raise unless every start rotation of start_mode is a signed permutation; "
       "returns the number of starts checked",
       py::arg("start_mode"));

    m.def("rotation_is_signed_permutation", [](float w, float x, float y, float z){
        const float q[4] = {w, x, y, z};
        return rotation_is_signed_permutation(q);
    }, "whether the rotation this unit quaternion represents is a signed permutation",
       py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"));

    m.def("device_count", [](){
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess){ cudaGetLastError(); return 0; }
        return n;
    }, "number of visible CUDA devices, or 0");

    // ---- the build stamp ---------------------------------------------------
    m.attr("__build_type__")          = ROSHAMBO3_BUILD_TYPE;
    m.attr("__git_sha__")             = ROSHAMBO3_GIT_SHA;
    // BESIDE the commit, never instead of it -- archived run records key on
    // __git_sha__. These four say which SOURCE this binary was compiled from:
    // a sha256 over the bytes of every compiled file (the one that moves for an
    // uncommitted edit), git's tree hash of the compiled subtree (the one that
    // does not, which is why the marker below is not optional), the porcelain
    // dirty marker, and when it was taken.
    m.attr("__cuda_src_sha256__")     = ROSHAMBO3_CUDA_SRC_SHA256;
    m.attr("__cuda_src_tree__")       = ROSHAMBO3_CUDA_SRC_TREE;
    m.attr("__cuda_src_dirty__")      = (int)(ROSHAMBO3_CUDA_SRC_DIRTY);
    m.attr("__cuda_src_dirty_files__") = ROSHAMBO3_CUDA_SRC_DIRTY_FILES;
    m.attr("__source_stamp_phase__")  = ROSHAMBO3_SOURCE_STAMP_PHASE;
    m.attr("__cuda_architectures__")  = ROSHAMBO3_CUDA_ARCHITECTURES;
    m.attr("__nvcc_version__")        = ROSHAMBO3_NVCC_VERSION;
    m.attr("__cxx_flags__")           = ROSHAMBO3_CXX_FLAGS;
    m.attr("__cuda_flags__")          = ROSHAMBO3_CUDA_FLAGS;
    // Empty unless a compiler cache was configured. A cache is a thing that can
    // hand back the wrong object, so a run record has to be able to say whether
    // one was in play rather than leave it implicit.
    m.attr("__compiler_launcher__")   = ROSHAMBO3_COMPILER_LAUNCHER_USED;
    m.attr("__nthreads__")           = ROSHAMBO3_NTHREADS;
    m.attr("__final_exact_shape__") = (bool)ROSHAMBO3_FINAL_EXACT_SHAPE;
    m.attr("__final_exact_color__") = (bool)ROSHAMBO3_FINAL_EXACT_COLOR;
}
