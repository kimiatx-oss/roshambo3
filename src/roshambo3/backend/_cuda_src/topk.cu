/// Device-resident top-K. See topk.cuh for the design and the
/// determinism argument.

#include "topk.cuh"
#include "device_guard.cuh"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define TOPK_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess){                                                \
            throw std::runtime_error(std::string("roshambo3.cuda: ")     \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"                 \
                + std::to_string(__LINE__));                                   \
        }                                                                      \
    } while (0)


////////////////////////////////////////////////////////////////////////////////
// Order-preserving u32 encoding of a float32.
//
// For IEEE-754, flipping the sign bit of a non-negative float and flipping
// every bit of a negative one gives an unsigned integer whose natural order is
// the float's order. Scores here are Tanimotos in [0, 1], but the encoding is
// written for the general case so a caller ranking on a signed column does not
// get a silently wrong sort.
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ unsigned int score_to_sortable_u32(float f){
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}


struct DeviceTopKState {
    int n_query = 0;
    int k = 0;
    int cap = 0;
    int rank_col = 0;
    int dv = 20;
    bool keep_cols = true;
    int device_id = 0;

    // candidate buffers, [n_query * cap] (cols: [n_query * cap * dv])
    float *     d_score = nullptr;
    long long * d_gid   = nullptr;
    float *     d_cols  = nullptr;
    // compaction destination for select(), same shapes
    float *     d_score2 = nullptr;
    long long * d_gid2   = nullptr;
    float *     d_cols2  = nullptr;

    unsigned int * d_count = nullptr;    // [n_query]
    float *        d_thr   = nullptr;    // [n_query]
    unsigned int * d_over  = nullptr;    // [1] overflow flag

    // radix-sort scratch, [cap]
    unsigned long long * d_k1 = nullptr;
    unsigned long long * d_k1o = nullptr;
    unsigned int * d_v1 = nullptr;
    unsigned int * d_v1o = nullptr;
    unsigned int * d_k2 = nullptr;
    unsigned int * d_k2o = nullptr;
    unsigned int * d_v2 = nullptr;
    unsigned int * d_v2o = nullptr;
    void *  d_tmp = nullptr;
    size_t  tmp_bytes = 0;

    std::vector<unsigned int> h_count;
    std::vector<float>        h_thr;

    long long n_offered  = 0;
    long long n_admitted = 0;
    long long n_select   = 0;
    long long bytes_d2h  = 0;
};


////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

/// Admit one row of a launch's score block into its query's candidate buffer.
/// Shared by the pair-list and the dense compaction kernels below, so the two
/// admit by one rule.
__device__ __forceinline__ void topk_admit_row(const float * __restrict__ scores,
                              long r, int q, long long gid, int dv, int rank_col,
                              float * __restrict__ cand_score,
                              long long * __restrict__ cand_gid,
                              float * __restrict__ cand_cols,
                              unsigned int * __restrict__ count,
                              const float * __restrict__ thr,
                              int cap, int keep_cols,
                              unsigned int * __restrict__ overflow){
    float s = scores[(size_t)r * dv + rank_col];
    // NaN fails `>=` and is dropped rather than ranked.
    if (!(s >= thr[q])) return;
    unsigned int slot = atomicAdd(&count[q], 1u);
    if (slot >= (unsigned int)cap){
        // Cannot happen when the caller respects max_rows_per_query, but a
        // silent drop here would be a wrong answer, not a slow one.
        atomicExch(overflow, 1u);
        return;
    }
    size_t dst = (size_t)q * cap + slot;
    cand_score[dst] = s;
    cand_gid[dst]   = gid;
    if (keep_cols){
        const float * src = scores + (size_t)r * dv;
        float * d = cand_cols + dst * dv;
        for (int j = 0; j < dv; ++j) d[j] = src[j];
    }
}

/// One launch's score block, addressed through a pair list -> the per-query
/// candidate buffers.
///
/// Grid-stride so the launch config is independent of `n_rows` -- which is also
/// what makes "identical across launch configurations" testable: the admitted
/// set is `{score >= thr}` whatever the grid looks like.
__global__ void topk_compact_kernel(const float * __restrict__ scores, long n_rows, int dv,
                              int rank_col,
                              const int * __restrict__ out_q,
                              const int * __restrict__ out_t,
                              long long gid_base,
                              float * __restrict__ cand_score,
                              long long * __restrict__ cand_gid,
                              float * __restrict__ cand_cols,
                              unsigned int * __restrict__ count,
                              const float * __restrict__ thr,
                              int cap, int keep_cols,
                              unsigned int * __restrict__ overflow){
    long stride = (long)blockDim.x * gridDim.x;
    for (long r = (long)blockIdx.x * blockDim.x + threadIdx.x; r < n_rows; r += stride){
        topk_admit_row(scores, r, out_q[r], gid_base + (long long)out_t[r], dv,
                       rank_col, cand_score, cand_gid, cand_cols, count, thr,
                       cap, keep_cols, overflow);
    }
}

/// The same, for a DENSE block: row `q * rows_per_query + t` is query `q`
/// against library row `t`. The query comes from the grid's y axis and the
/// library row from the x axis, so no pair array is read and no index is
/// divided. The candidate buffers are put into canonical order by gid before
/// anything reads them (pass 1 below), so which slot a row wins -- and hence
/// the grid shape -- does not reach the result.
__global__ void topk_compact_dense_kernel(const float * __restrict__ scores,
                              int n_query, long rows_per_query, int dv,
                              int rank_col, long long gid_base,
                              float * __restrict__ cand_score,
                              long long * __restrict__ cand_gid,
                              float * __restrict__ cand_cols,
                              unsigned int * __restrict__ count,
                              const float * __restrict__ thr,
                              int cap, int keep_cols,
                              unsigned int * __restrict__ overflow){
    long stride = (long)blockDim.x * gridDim.x;
    for (int q = blockIdx.y; q < n_query; q += gridDim.y){
        const long base = (long)q * rows_per_query;
        for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
             t < rows_per_query; t += stride){
            topk_admit_row(scores, base + t, q, gid_base + (long long)t, dv,
                           rank_col, cand_score, cand_gid, cand_cols, count,
                           thr, cap, keep_cols, overflow);
        }
    }
}

/// Pass-1 keys: the conformer id itself. Ids are unique and non-negative, so
/// sorting by them puts the buffer into a canonical sequence that no longer
/// depends on which atomicAdd slot each row happened to win.
__global__ void topk_keys_by_id_kernel(const long long * __restrict__ gid, int n,
                               unsigned long long * __restrict__ key,
                               unsigned int * __restrict__ val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key[i] = (unsigned long long)gid[i];
    val[i] = (unsigned int)i;
}

/// Pass-2 keys: the *complement* of the score encoding, so an ascending sort
/// orders by descending score. Ascending CUB radix sort is stable, so the
/// ascending-id order established by pass 1 survives inside each tie group --
/// which is `rank_order`'s tie-break, done on the device. (Sorting descending
/// instead would rely on `SortPairsDescending` being stable; this does not.)
__global__ void topk_keys_by_score_kernel(const float * __restrict__ score,
                                 const unsigned int * __restrict__ perm, int n,
                                 unsigned int * __restrict__ key,
                                 unsigned int * __restrict__ val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int p = perm[i];
    key[i] = ~score_to_sortable_u32(score[p]);
    val[i] = p;
}

__global__ void topk_gather_kernel(const float * __restrict__ score,
                             const long long * __restrict__ gid,
                             const float * __restrict__ cols,
                             const unsigned int * __restrict__ perm, int n, int dv,
                             int keep_cols,
                             float * __restrict__ oscore,
                             long long * __restrict__ ogid,
                             float * __restrict__ ocols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int p = perm[i];
    oscore[i] = score[p];
    ogid[i]   = gid[p];
    if (keep_cols){
        const float * s = cols + (size_t)p * dv;
        float * d = ocols + (size_t)i * dv;
        for (int j = 0; j < dv; ++j) d[j] = s[j];
    }
}

/// count[q] = n; thr[q] = score of the k-th held row, or -inf if fewer than k.
__global__ void topk_set_state_kernel(unsigned int * count, float * thr, int q, int n, int k,
                                const float * __restrict__ score){
    if (threadIdx.x || blockIdx.x) return;
    count[q] = (unsigned int)n;
    thr[q] = (n >= k) ? score[k - 1] : -INFINITY;
}


////////////////////////////////////////////////////////////////////////////////
// Host side
////////////////////////////////////////////////////////////////////////////////

static void topk_alloc(DeviceTopKState * st){
    // The guard sits on the allocator itself, not on its caller: the caller is
    // where it was missing for the molecule-level sink.
    ScopedDevice on_device(st->device_id);
    const size_t nq = (size_t)st->n_query, cap = (size_t)st->cap, dv = (size_t)st->dv;
    TOPK_CHECK(cudaMalloc((void **)&st->d_score,  nq * cap * sizeof(float)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_gid,    nq * cap * sizeof(long long)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_score2, cap * sizeof(float)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_gid2,   cap * sizeof(long long)));
    if (st->keep_cols){
        TOPK_CHECK(cudaMalloc((void **)&st->d_cols,  nq * cap * dv * sizeof(float)));
        TOPK_CHECK(cudaMalloc((void **)&st->d_cols2, cap * dv * sizeof(float)));
    }
    TOPK_CHECK(cudaMalloc((void **)&st->d_count, nq * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_thr,   nq * sizeof(float)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_over,  sizeof(unsigned int)));
    TOPK_CHECK(cudaMemset(st->d_count, 0, nq * sizeof(unsigned int)));
    TOPK_CHECK(cudaMemset(st->d_over,  0, sizeof(unsigned int)));

    std::vector<float> ninf((size_t)st->n_query, -std::numeric_limits<float>::infinity());
    TOPK_CHECK(cudaMemcpy(st->d_thr, ninf.data(), nq * sizeof(float), cudaMemcpyHostToDevice));

    TOPK_CHECK(cudaMalloc((void **)&st->d_k1,  cap * sizeof(unsigned long long)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_k1o, cap * sizeof(unsigned long long)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_v1,  cap * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_v1o, cap * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_k2,  cap * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_k2o, cap * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_v2,  cap * sizeof(unsigned int)));
    TOPK_CHECK(cudaMalloc((void **)&st->d_v2o, cap * sizeof(unsigned int)));

    size_t b1 = 0, b2 = 0;
    TOPK_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, b1, st->d_k1, st->d_k1o,
                                              st->d_v1, st->d_v1o, st->cap));
    TOPK_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, b2, st->d_k2, st->d_k2o,
                                              st->d_v2, st->d_v2o, st->cap));
    st->tmp_bytes = std::max(b1, b2);
    TOPK_CHECK(cudaMalloc(&st->d_tmp, st->tmp_bytes));

    st->h_count.assign((size_t)st->n_query, 0u);
    st->h_thr.assign((size_t)st->n_query, -std::numeric_limits<float>::infinity());
}

static void topk_free(DeviceTopKState * st){
    void * ps[] = {st->d_score, st->d_gid, st->d_cols, st->d_score2, st->d_gid2,
                   st->d_cols2, st->d_count, st->d_thr, st->d_over,
                   st->d_k1, st->d_k1o, st->d_v1, st->d_v1o,
                   st->d_k2, st->d_k2o, st->d_v2, st->d_v2o, st->d_tmp};
    for (void * p : ps) if (p) cudaFree(p);
}

DeviceTopKState * topk_create(int n_query, int k, int rank_col, int dv,
                          bool keep_cols, int device_id){
    if (n_query <= 0) throw std::runtime_error("roshambo3.cuda: n_query must be > 0");
    if (k <= 0)       throw std::runtime_error("roshambo3.cuda: k must be > 0");
    if (rank_col < 0 || rank_col >= dv)
        throw std::runtime_error("roshambo3.cuda: rank_col out of range");
    // Nothing is allocated here, so there is nothing to put on the wrong
    // device: capacity is fixed on the first block, inside `topk_add_block`,
    // which sets the device. That is why the conformer-level sink survived the
    // defect found in the molecule-level one -- an accident of laziness,
    // not a property, so `topk_alloc` carries the guard.
    DeviceTopKState * st = new DeviceTopKState();
    st->n_query = n_query; st->k = k; st->rank_col = rank_col; st->dv = dv;
    st->keep_cols = keep_cols; st->device_id = device_id;
    // Capacity is set on the first block, when max_rows_per_query is known.
    st->cap = 0;
    return st;
}

void topk_destroy(DeviceTopKState * st){
    if (!st) return;
    // No-throw: this runs from ~TopKHandle, where an exception would terminate.
    ScopedDevice on_device(st->device_id, std::nothrow);
    topk_free(st);
    delete st;
}

/// Reduce one query's buffer to its best `min(count, k)` rows, in canonical
/// order, and raise its threshold.
static void topk_select(DeviceTopKState * st, int q){
    ScopedDevice on_device(st->device_id);
    unsigned int n = st->h_count[(size_t)q];
    if (n == 0){
        st->h_thr[(size_t)q] = -std::numeric_limits<float>::infinity();
        return;
    }
    const int nn = (int)n;
    const size_t base = (size_t)q * (size_t)st->cap;
    float *     score = st->d_score + base;
    long long * gid   = st->d_gid   + base;
    float *     cols  = st->keep_cols ? (st->d_cols + base * st->dv) : nullptr;

    const int TPB = 256;
    const int nb = (nn + TPB - 1) / TPB;

    // pass 1: ascending by conformer id -> a canonical sequence
    topk_keys_by_id_kernel<<<nb, TPB>>>(gid, nn, st->d_k1, st->d_v1);
    TOPK_CHECK(cudaGetLastError());
    size_t tb = st->tmp_bytes;
    TOPK_CHECK(cub::DeviceRadixSort::SortPairs(st->d_tmp, tb, st->d_k1, st->d_k1o,
                                              st->d_v1, st->d_v1o, nn));
    // pass 2: ascending by ~monokey(score) -> descending score, ties still by id
    topk_keys_by_score_kernel<<<nb, TPB>>>(score, st->d_v1o, nn, st->d_k2, st->d_v2);
    TOPK_CHECK(cudaGetLastError());
    tb = st->tmp_bytes;
    TOPK_CHECK(cub::DeviceRadixSort::SortPairs(st->d_tmp, tb, st->d_k2, st->d_k2o,
                                              st->d_v2, st->d_v2o, nn));

    const int keep = std::min(nn, st->k);
    const int kb = (keep + TPB - 1) / TPB;
    topk_gather_kernel<<<kb, TPB>>>(score, gid, cols, st->d_v2o, keep, st->dv,
                              st->keep_cols ? 1 : 0,
                              st->d_score2, st->d_gid2, st->d_cols2);
    TOPK_CHECK(cudaGetLastError());

    TOPK_CHECK(cudaMemcpy(score, st->d_score2, (size_t)keep * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    TOPK_CHECK(cudaMemcpy(gid, st->d_gid2, (size_t)keep * sizeof(long long),
                         cudaMemcpyDeviceToDevice));
    if (st->keep_cols){
        TOPK_CHECK(cudaMemcpy(cols, st->d_cols2,
                             (size_t)keep * st->dv * sizeof(float),
                             cudaMemcpyDeviceToDevice));
    }
    topk_set_state_kernel<<<1, 1>>>(st->d_count, st->d_thr, q, keep, st->k, score);
    TOPK_CHECK(cudaGetLastError());
    TOPK_CHECK(cudaDeviceSynchronize());

    st->h_count[(size_t)q] = (unsigned int)keep;
    if (keep >= st->k){
        float thr = 0.0f;
        TOPK_CHECK(cudaMemcpy(&thr, score + (st->k - 1), sizeof(float),
                             cudaMemcpyDeviceToHost));
        st->bytes_d2h += sizeof(float);
        st->h_thr[(size_t)q] = thr;
    } else {
        st->h_thr[(size_t)q] = -std::numeric_limits<float>::infinity();
    }
    st->n_select += 1;
}

void topk_add_block(DeviceTopKState * st, const float * d_scores, long n_rows,
                        const int * d_out_q, const int * d_out_t,
                        long dense_rows_per_query,
                        long long gid_base, long max_rows_per_query,
                        int device_id){
    // `st->device_id`, not the `device_id` argument: the state owns its
    // buffers' device, and several bindings default that argument to 0.
    ScopedDevice on_device(st->device_id);
    if (n_rows <= 0) return;
    if (max_rows_per_query <= 0) max_rows_per_query = n_rows;

    // Headroom: after a select a buffer holds at most k, so
    // cap >= k + 2 * max_rows_per_query lets a block land whole, twice, before
    // a select is even considered. Sized once, on the first block; a later
    // block that needs more is a hard error rather than a silent drop.
    if (st->cap == 0){
        long want = (long)st->k + 2 * max_rows_per_query;
        if (want > (long)INT32_MAX)
            throw std::runtime_error("roshambo3.cuda: capacity overflow");
        st->cap = (int)want;
        topk_alloc(st);
    } else if ((long)st->k + max_rows_per_query > (long)st->cap){
        throw std::runtime_error(
            "roshambo3.cuda: block of " + std::to_string(max_rows_per_query)
            + " rows/query does not fit the capacity sized for the first block ("
            + std::to_string(st->cap) + "); construct the state per screen, not per run");
    }

    // Select any query that could not absorb this block.
    for (int q = 0; q < st->n_query; ++q){
        if ((long)st->h_count[(size_t)q] + max_rows_per_query > (long)st->cap){
            topk_select(st, q);
        }
    }

    const int TPB = 256;
    if (d_out_q == nullptr){
        // Dense: the block is n_query x dense_rows_per_query, query-major.
        if (dense_rows_per_query <= 0
                || n_rows != (long)st->n_query * dense_rows_per_query)
            throw std::runtime_error(
                "roshambo3.cuda: a dense top-K block must be n_query x "
                "rows_per_query rows; got " + std::to_string(n_rows) + " rows for "
                + std::to_string(st->n_query) + " queries x "
                + std::to_string(dense_rows_per_query));
        int nblk = (int)std::min<long>(65535, (dense_rows_per_query + TPB - 1) / TPB);
        if (nblk < 1) nblk = 1;
        dim3 grid((unsigned)nblk, (unsigned)std::min(65535, st->n_query));
        topk_compact_dense_kernel<<<grid, TPB>>>(d_scores, st->n_query,
                                     dense_rows_per_query, st->dv, st->rank_col,
                                     gid_base, st->d_score, st->d_gid, st->d_cols,
                                     st->d_count, st->d_thr, st->cap,
                                     st->keep_cols ? 1 : 0, st->d_over);
    } else {
        int nblk = (int)std::min<long>(65535, (n_rows + TPB - 1) / TPB);
        if (nblk < 1) nblk = 1;
        topk_compact_kernel<<<nblk, TPB>>>(d_scores, n_rows, st->dv, st->rank_col,
                                     d_out_q, d_out_t, gid_base,
                                     st->d_score, st->d_gid, st->d_cols,
                                     st->d_count, st->d_thr, st->cap,
                                     st->keep_cols ? 1 : 0, st->d_over);
    }
    TOPK_CHECK(cudaGetLastError());
    TOPK_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> before = st->h_count;
    TOPK_CHECK(cudaMemcpy(st->h_count.data(), st->d_count,
                         (size_t)st->n_query * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost));
    st->bytes_d2h += (long long)st->n_query * sizeof(unsigned int);
    unsigned int over = 0;
    TOPK_CHECK(cudaMemcpy(&over, st->d_over, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    st->bytes_d2h += sizeof(unsigned int);
    if (over){
        throw std::runtime_error("roshambo3.cuda: candidate buffer overflowed; "
                                 "rows that could enter the top-K were dropped");
    }
    st->n_offered += (long long)n_rows;
    for (int q = 0; q < st->n_query; ++q){
        st->n_admitted += (long long)st->h_count[(size_t)q] - (long long)before[(size_t)q];
    }
}

void topk_flush(DeviceTopKState * st, int device_id){
    ScopedDevice on_device(st->device_id);
    if (st->cap == 0) return;
    for (int q = 0; q < st->n_query; ++q) topk_select(st, q);
}

void topk_result(DeviceTopKState * st, float * h_scores, long long * h_ids,
                     int * h_counts, int device_id){
    ScopedDevice on_device(st->device_id);
    for (int q = 0; q < st->n_query; ++q) h_counts[q] = 0;
    if (st->cap == 0) return;
    topk_flush(st, device_id);
    for (int q = 0; q < st->n_query; ++q){
        int n = (int)st->h_count[(size_t)q];
        h_counts[q] = n;
        if (n == 0) continue;
        const size_t base = (size_t)q * (size_t)st->cap;
        TOPK_CHECK(cudaMemcpy(h_ids + (size_t)q * st->k, st->d_gid + base,
                             (size_t)n * sizeof(long long), cudaMemcpyDeviceToHost));
        st->bytes_d2h += (long long)n * sizeof(long long);
        if (st->keep_cols && h_scores){
            TOPK_CHECK(cudaMemcpy(h_scores + (size_t)q * st->k * st->dv,
                                 st->d_cols + base * st->dv,
                                 (size_t)n * st->dv * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            st->bytes_d2h += (long long)n * st->dv * sizeof(float);
        } else if (h_scores){
            TOPK_CHECK(cudaMemcpy(h_scores + (size_t)q * st->k, st->d_score + base,
                                 (size_t)n * sizeof(float), cudaMemcpyDeviceToHost));
            st->bytes_d2h += (long long)n * sizeof(float);
        }
    }
}

void topk_thresholds(DeviceTopKState * st, float * h_thr){
    for (int q = 0; q < st->n_query; ++q) h_thr[q] = st->h_thr.empty()
        ? -std::numeric_limits<float>::infinity() : st->h_thr[(size_t)q];
}

void topk_counters(DeviceTopKState * st, long long * n_offered, long long * n_admitted,
                       long long * n_select, long long * bytes_d2h){
    *n_offered = st->n_offered; *n_admitted = st->n_admitted;
    *n_select = st->n_select;   *bytes_d2h = st->bytes_d2h;
}

int topk_n_query(DeviceTopKState * st){ return st->n_query; }
int topk_k(DeviceTopKState * st){ return st->k; }
int topk_dv(DeviceTopKState * st){ return st->dv; }
long long topk_capacity_bytes(DeviceTopKState * st){
    if (st->cap == 0) return 0;
    long long per = (long long)st->cap * (long long)(sizeof(float) + sizeof(long long));
    if (st->keep_cols) per += (long long)st->cap * st->dv * sizeof(float);
    return per * st->n_query;
}


////////////////////////////////////////////////////////////////////////////////
// The self-overlap bound scan (see topk.cuh). One thread per row, grid-stride.
////////////////////////////////////////////////////////////////////////////////
namespace {
// Column indices of roshambo3.scores: OVERLAP_VOLUME, SELF_OVERLAP_QUERY,
// SELF_OVERLAP_LIBRARY. The same three the CPU sink reads.
constexpr int kOverlapVolume = 3;
constexpr int kSelfOverlapQuery = 5;
constexpr int kSelfOverlapLibrary = 6;
}

__global__ void self_overlap_bound_kernel(const float * __restrict__ scores,
                                          long n_rows, int dv, double margin,
                                          unsigned long long * counters){
    for (long r = (long)blockIdx.x * blockDim.x + threadIdx.x; r < n_rows;
         r += (long)gridDim.x * blockDim.x){
        const float * row = scores + (size_t)r * dv;
        const double overlap = (double)row[kOverlapVolume];
        const double selves = (double)row[kSelfOverlapQuery]
                            + (double)row[kSelfOverlapLibrary];
        if (!(2.0 * overlap > selves * (1.0 + margin))) continue;
        atomicAdd(&counters[0], 1ull);
        atomicMin(&counters[1], (unsigned long long)r);
    }
}

void self_overlap_bound_scan(const float * d_scores, long n_rows, int dv,
                             double margin, unsigned long long * d_counters,
                             unsigned long long * h_flagged,
                             long long * h_first_row, int device_id){
    ScopedDevice on_device(device_id);
    if (dv <= kSelfOverlapLibrary)
        throw std::runtime_error(
            "roshambo3.cuda: the self-overlap bound scan needs the overlap and "
            "self-overlap columns; this block has " + std::to_string(dv));
    TOPK_CHECK(cudaMemset(d_counters, 0, sizeof(unsigned long long)));
    TOPK_CHECK(cudaMemset(d_counters + 1, 0xFF, sizeof(unsigned long long)));
    if (n_rows > 0){
        const int TPB = 256;
        int nblk = (int)std::min<long>(65535, (n_rows + TPB - 1) / TPB);
        if (nblk < 1) nblk = 1;
        self_overlap_bound_kernel<<<nblk, TPB>>>(d_scores, n_rows, dv, margin,
                                                 d_counters);
        TOPK_CHECK(cudaGetLastError());
    }
    unsigned long long h[2] = {0, 0};
    TOPK_CHECK(cudaMemcpy(h, d_counters, sizeof(h), cudaMemcpyDeviceToHost));
    *h_flagged = h[0];
    *h_first_row = (h[1] == ~0ull) ? -1 : (long long)h[1];
}
