/// Molecule-level top-K -- a segmented argmax in front of the device top-K.
/// See molecule_topk.cuh for the design, the tie-break and the determinism
/// argument.

#include "molecule_topk.cuh"
#include "device_guard.cuh"
#include "topk.cuh"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define MOLTOPK_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess){                                                 \
            throw std::runtime_error(std::string("roshambo3.cuda: ")            \
                + cudaGetErrorString(_e) + " at " __FILE__ ":"                  \
                + std::to_string(__LINE__));                                    \
        }                                                                       \
    } while (0)

//: Warps per CUDA block in the segmented argmax. One warp per (query, segment):
//: segments are short -- a molecule holds 1 to a few tens of conformers -- so a
//: whole thread block per segment would leave most of it idle.
static const int MOLTOPK_WARPS = 8;


struct MoleculeTopKState {
    DeviceTopKState * inner = nullptr;
    int n_query = 0;
    int dv = 20;
    int rank_col = 0;
    int device_id = 0;
    long max_nb = 0;

    // per-block staging
    long long * d_mol_id    = nullptr;   // [max_nb]
    int *       d_flag      = nullptr;   // [max_nb]
    int *       d_seg_of    = nullptr;   // [max_nb], inclusive scan of d_flag
    int *       d_seg_start = nullptr;   // [max_nb + 1]
    int *       d_info      = nullptr;   // [2]: n_segments, trailing segment start
    void *      d_tmp       = nullptr;
    size_t      tmp_bytes   = 0;

    // compacted molecule rows handed to the conformer-level top-K
    float *     d_out_score = nullptr;   // [n_query * max_nb * dv]
    int *       d_out_q     = nullptr;   // [n_query * max_nb]
    int *       d_out_t     = nullptr;   // [n_query * max_nb]

    // the molecule left open at a block boundary, double-buffered so the warp
    // that reads it and the warp that replaces it cannot race
    float *     d_carry_score = nullptr; // [2 * n_query * dv]
    long long * d_carry_gid   = nullptr; // [2 * n_query]
    int         carry_slot = 0;
    bool        carry_live = false;
    long long   carry_mol_id = 0;

    unsigned int * d_bad = nullptr;      // [1]

    // staged block
    bool      staged = false;
    long      nb = 0;
    long long gid_base = 0;
    long long base_gid = 0;
    int       tail_open = 0;
    int       n_seg = 0;
    int       tail_seg_start = 0;
    long long tail_mol_id = 0;

    long long n_conformers = 0;
    long long n_molecules  = 0;
    long long n_carried    = 0;
    long long n_blocks     = 0;
};


////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////

/// The one ordering rule: higher score wins, ties go to the lower global
/// conformer id, and a NaN loses to anything that is not one.
///
/// A strict weak ordering on `(score, gid)`, which is what makes the warp
/// reduction, the per-lane serial scan and the carry merge give the same answer
/// whatever order they visit rows in. `roshambo3.numerics.rank_order`'s
/// tie-break, applied to choosing which conformer represents a molecule.
__device__ __forceinline__ bool moltopk_beats(float bs, long long bg,
                                              float as, long long ag){
    const bool an = isnan(as), bn = isnan(bs);
    if (an != bn) return an;          // a NaN incumbent loses to a real score
    if (an)       return bg < ag;     // all-NaN molecule: the lowest id names it
    if (bs != as) return bs > as;
    return bg < ag;
}

__global__ void moltopk_flag_kernel(const long long * __restrict__ mol, long n,
                                    int * __restrict__ flag){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flag[i] = (i == 0 || mol[i] != mol[i - 1]) ? 1 : 0;
}

__global__ void moltopk_starts_kernel(const int * __restrict__ flag,
                                      const int * __restrict__ seg_of, long n,
                                      int * __restrict__ seg_start){
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flag[i]) seg_start[seg_of[i] - 1] = (int)i;
}

/// Close the segment table and report it in one place, so the host needs a
/// single 8 B copy rather than two that each synchronize.
__global__ void moltopk_close_kernel(const int * __restrict__ seg_of, long n,
                                     int * __restrict__ seg_start,
                                     int * __restrict__ info){
    if (threadIdx.x || blockIdx.x) return;
    const int m = seg_of[n - 1];
    seg_start[m] = (int)n;          // sentinel, so segment j is [start[j], start[j+1])
    info[0] = m;                    // segments in this block
    info[1] = seg_start[m - 1];     // where the trailing one starts
}

/// One warp per (query, segment): reduce the segment to its best conformer,
/// merge the carried molecule into segment 0, and either emit the row or hold
/// the trailing segment back as the new carry.
__global__ void moltopk_segment_argmax_kernel(
        const float * __restrict__ scores, long nb, int dv, int rank_col,
        int n_seg, const int * __restrict__ seg_start,
        long long gid_base, long long base_gid,
        int merge_carry, int tail_open, int n_out,
        const float * __restrict__ carry_in_score,
        const long long * __restrict__ carry_in_gid,
        float * __restrict__ carry_out_score,
        long long * __restrict__ carry_out_gid,
        float * __restrict__ out_score, int * __restrict__ out_q,
        int * __restrict__ out_t, unsigned int * __restrict__ bad){
    const int lane = threadIdx.x & 31;
    const int seg  = blockIdx.x * (int)(blockDim.x >> 5) + (int)(threadIdx.x >> 5);
    const int q    = blockIdx.y;
    if (seg >= n_seg) return;        // uniform across the warp

    const int s0 = seg_start[seg];
    const int s1 = seg_start[seg + 1];
    const float * qs = scores + (size_t)q * (size_t)nb * (size_t)dv;

    // A lane with no row of its own carries the identity of `moltopk_beats`:
    // NaN loses to every real score, and the largest id loses every NaN tie.
    float     bs = nanf("");
    long long bg = 0x7fffffffffffffffLL;
    for (int t = s0 + lane; t < s1; t += 32){
        const float s = qs[(size_t)t * dv + rank_col];
        const long long g = gid_base + (long long)t;
        if (moltopk_beats(s, g, bs, bg)){ bs = s; bg = g; }
    }
    for (int off = 16; off > 0; off >>= 1){
        const float     os = __shfl_down_sync(0xffffffffu, bs, off);
        const long long og = __shfl_down_sync(0xffffffffu, bg, off);
        if (moltopk_beats(os, og, bs, bg)){ bs = os; bg = og; }
    }

    int from_carry = 0;
    if (lane == 0 && seg == 0 && merge_carry){
        const float     cs = carry_in_score[(size_t)q * dv + rank_col];
        const long long cg = carry_in_gid[q];
        // A carried molecule started in an earlier block, so its winning
        // conformer precedes this one. If it does not, block order was not what
        // the carry assumes and the answer would be wrong rather than slow.
        if (cg >= gid_base) atomicExch(bad, 1u);
        if (moltopk_beats(cs, cg, bs, bg)){ bs = cs; bg = cg; from_carry = 1; }
    }
    bs         = __shfl_sync(0xffffffffu, bs, 0);
    bg         = __shfl_sync(0xffffffffu, bg, 0);
    from_carry = __shfl_sync(0xffffffffu, from_carry, 0);

    const bool to_carry = (seg == n_seg - 1) && tail_open;
    float * dst;
    if (to_carry){
        dst = carry_out_score + (size_t)q * dv;
        if (lane == 0) carry_out_gid[q] = bg;
    } else {
        const long long rel = bg - base_gid;
        if (rel < 0 || rel > 0x7fffffffLL){
            if (lane == 0) atomicExch(bad, 1u);
            return;
        }
        const size_t o = (size_t)q * (size_t)n_out + (size_t)seg;
        dst = out_score + o * dv;
        if (lane == 0){ out_q[o] = q; out_t[o] = (int)rel; }
    }
    const float * src = from_carry ? (carry_in_score + (size_t)q * dv)
                                   : (qs + (size_t)(bg - gid_base) * dv);
    for (int c = lane; c < dv; c += 32) dst[c] = src[c];
}


////////////////////////////////////////////////////////////////////////////////
// Host side
////////////////////////////////////////////////////////////////////////////////

MoleculeTopKState * moltopk_create(int n_query, int k, int rank_col, int dv,
                                   bool keep_cols, int device_id,
                                   long max_block_conformers){
    if (max_block_conformers <= 0)
        throw std::runtime_error("roshambo3.cuda: max_block_conformers must be > 0");
    if (!keep_cols)
        throw std::runtime_error(
            "roshambo3.cuda: the molecule-level top-K keeps all score columns; "
            "it emits a winning conformer's whole score row");

    // Every allocation below lands on whichever device is current, and every
    // kernel that later reads them runs on `device_id` because `score_pairs`
    // sets it. Those are the same device only by accident on a one-GPU host.
    ScopedDevice on_device(device_id);

    MoleculeTopKState * st = new MoleculeTopKState();
    try {
        st->inner = topk_create(n_query, k, rank_col, dv, keep_cols, device_id);
        st->n_query = n_query;
        st->dv = dv;
        st->rank_col = rank_col;
        st->device_id = device_id;
        st->max_nb = max_block_conformers;

        const size_t nb = (size_t)st->max_nb, nq = (size_t)n_query, d = (size_t)dv;
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_mol_id,    nb * sizeof(long long)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_flag,      nb * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_seg_of,    nb * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_seg_start, (nb + 1) * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_info,       2 * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_out_score, nq * nb * d * sizeof(float)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_out_q,     nq * nb * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_out_t,     nq * nb * sizeof(int)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_carry_score, 2 * nq * d * sizeof(float)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_carry_gid,   2 * nq * sizeof(long long)));
        MOLTOPK_CHECK(cudaMalloc((void **)&st->d_bad, sizeof(unsigned int)));
        MOLTOPK_CHECK(cudaMemset(st->d_bad, 0, sizeof(unsigned int)));

        size_t bytes = 0;
        MOLTOPK_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, st->d_flag,
                                                    st->d_seg_of, (int)st->max_nb));
        st->tmp_bytes = bytes;
        MOLTOPK_CHECK(cudaMalloc(&st->d_tmp, bytes));
    } catch (...){
        moltopk_destroy(st);
        throw;
    }
    return st;
}

void moltopk_destroy(MoleculeTopKState * st){
    if (!st) return;
    // No-throw: this runs from ~MoleculeTopKHandle, where an exception would
    // terminate rather than report.
    ScopedDevice on_device(st->device_id, std::nothrow);
    void * ps[] = {st->d_mol_id, st->d_flag, st->d_seg_of, st->d_seg_start,
                   st->d_info, st->d_tmp, st->d_out_score, st->d_out_q,
                   st->d_out_t, st->d_carry_score, st->d_carry_gid, st->d_bad};
    for (void * p : ps) if (p) cudaFree(p);
    if (st->inner) topk_destroy(st->inner);
    delete st;
}

DeviceTopKState * moltopk_inner(MoleculeTopKState * st){ return st->inner; }

static void moltopk_check_bad(MoleculeTopKState * st){
    ScopedDevice on_device(st->device_id);
    unsigned int bad = 0;
    MOLTOPK_CHECK(cudaMemcpy(&bad, st->d_bad, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
    if (bad){
        MOLTOPK_CHECK(cudaMemset(st->d_bad, 0, sizeof(unsigned int)));
        throw std::runtime_error(
            "roshambo3.cuda: the molecule reduction saw a global conformer id "
            "outside the range its base was chosen for, which means blocks did "
            "not arrive in ascending id order or a carried molecule was matched "
            "to the wrong block. The molecule-level answer would be wrong, not "
            "merely mis-ordered.");
    }
}

void moltopk_stage_block(MoleculeTopKState * st, const long long * h_mol_id,
                         long n_conformers, long long gid_base, int tail_open,
                         long long base_gid, int device_id){
    // `device_id` is the caller's idea of the device; `st->device_id` is where
    // the buffers actually are, and several bindings default the argument to 0.
    ScopedDevice on_device(st->device_id);
    if (n_conformers <= 0)
        throw std::runtime_error("roshambo3.cuda: a staged block must hold conformers");
    if (n_conformers > st->max_nb)
        throw std::runtime_error(
            "roshambo3.cuda: block of " + std::to_string(n_conformers)
            + " conformers exceeds the " + std::to_string(st->max_nb)
            + " the molecule-level top-K was sized for; size it from the "
              "source's largest block, not the first one");
    if (base_gid > gid_base)
        throw std::runtime_error("roshambo3.cuda: base_gid must not exceed gid_base");
    if (gid_base + n_conformers - base_gid > (long long)INT32_MAX)
        throw std::runtime_error(
            "roshambo3.cuda: the span from base_gid to the end of this block "
            "does not fit an int32 output index");
    if (st->carry_live && h_mol_id[0] != st->carry_mol_id)
        throw std::runtime_error(
            "roshambo3.cuda: a molecule was left open by the previous block but "
            "this block does not continue it. Blocks must be staged in ascending "
            "global-id order, and tail_open must be 0 at the end of a group.");

    MOLTOPK_CHECK(cudaMemcpy(st->d_mol_id, h_mol_id,
                             (size_t)n_conformers * sizeof(long long),
                             cudaMemcpyHostToDevice));

    const int TPB = 256;
    const int nblk = (int)((n_conformers + TPB - 1) / TPB);
    moltopk_flag_kernel<<<nblk, TPB>>>(st->d_mol_id, n_conformers, st->d_flag);
    MOLTOPK_CHECK(cudaGetLastError());
    size_t bytes = st->tmp_bytes;
    MOLTOPK_CHECK(cub::DeviceScan::InclusiveSum(st->d_tmp, bytes, st->d_flag,
                                                st->d_seg_of, (int)n_conformers));
    moltopk_starts_kernel<<<nblk, TPB>>>(st->d_flag, st->d_seg_of, n_conformers,
                                         st->d_seg_start);
    MOLTOPK_CHECK(cudaGetLastError());
    moltopk_close_kernel<<<1, 1>>>(st->d_seg_of, n_conformers, st->d_seg_start,
                                   st->d_info);
    MOLTOPK_CHECK(cudaGetLastError());

    int info[2] = {0, 0};
    MOLTOPK_CHECK(cudaMemcpy(info, st->d_info, 2 * sizeof(int),
                             cudaMemcpyDeviceToHost));
    const int n_seg = info[0], tail_start = info[1];
    if (n_seg < 1 || n_seg > (int)n_conformers)
        throw std::runtime_error("roshambo3.cuda: implausible segment count "
                                 + std::to_string(n_seg));

    st->staged = true;
    st->nb = n_conformers;
    st->gid_base = gid_base;
    st->base_gid = base_gid;
    st->tail_open = tail_open ? 1 : 0;
    st->n_seg = n_seg;
    st->tail_seg_start = tail_start;
    st->tail_mol_id = h_mol_id[n_conformers - 1];
}

void moltopk_add_block(MoleculeTopKState * st, const float * d_scores, long n_rows,
                       const int * h_out_q, const int * h_out_t,
                       long max_rows_per_query, int device_id){
    ScopedDevice on_device(st->device_id);
    if (!st->staged)
        throw std::runtime_error(
            "roshambo3.cuda: no block is staged. The molecule-level top-K needs "
            "this launch's mol_id sidecar slice, staged before the launch.");
    const long nb = st->nb;
    const int  nq = st->n_query;
    if (n_rows != (long)nq * nb)
        throw std::runtime_error(
            "roshambo3.cuda: the molecule-level top-K needs the full "
            + std::to_string(nq) + " x " + std::to_string(nb)
            + " query-major pair block, got " + std::to_string(n_rows) + " rows");
    // A dense launch passes no pair list (null), and its rows ARE this layout
    // by construction. A pair list is verified, not assumed:
    // `optimize_pairs` takes an arbitrary one, and a caller who passed a
    // different one would get a confidently wrong answer rather than an error.
    // Two compares per row over 128 KB, against a launch that runs an optimizer
    // over the same block -- but written as a nested loop rather than `r / nb`
    // and `r % nb`, because two integer divisions per row *was* measurable: 80 M
    // of them over a 10 M-conformer screen.
    if ((h_out_q == nullptr) != (h_out_t == nullptr))
        throw std::runtime_error(
            "roshambo3.cuda: the molecule-level top-K needs both halves of a pair "
            "list, or neither (a dense launch)");
    if (h_out_q != nullptr){
        long r = 0;
        for (int q = 0; q < nq; ++q){
            for (long t = 0; t < nb; ++t, ++r){
                if (h_out_q[r] != q || h_out_t[r] != (int)t)
                    throw std::runtime_error(
                        "roshambo3.cuda: the molecule-level top-K needs the "
                        "dense query-major pair block (row q*n + t is query q "
                        "against library row t); this pair list is not that, so "
                        "the segmented reduction cannot locate a conformer's "
                        "score");
            }
        }
    }

    // n_out == 0 means the whole block is one still-open molecule: it reduces
    // into the carry and emits nothing. Only the *trailing* segment is ever
    // held back, so the emitted rows are segments 0..n_out-1 in order.
    const int n_out = st->n_seg - st->tail_open;
    const int in = st->carry_slot, out = 1 - st->carry_slot;
    dim3 grid((unsigned)((st->n_seg + MOLTOPK_WARPS - 1) / MOLTOPK_WARPS),
              (unsigned)nq);
    moltopk_segment_argmax_kernel<<<grid, 32 * MOLTOPK_WARPS>>>(
        d_scores, nb, st->dv, st->rank_col, st->n_seg, st->d_seg_start,
        st->gid_base, st->base_gid, st->carry_live ? 1 : 0, st->tail_open,
        n_out > 0 ? n_out : 1,
        st->d_carry_score + (size_t)in * nq * st->dv,
        st->d_carry_gid + (size_t)in * nq,
        st->d_carry_score + (size_t)out * nq * st->dv,
        st->d_carry_gid + (size_t)out * nq,
        st->d_out_score, st->d_out_q, st->d_out_t, st->d_bad);
    MOLTOPK_CHECK(cudaGetLastError());
    moltopk_check_bad(st);          // the D2H orders this behind the kernel
    if (n_out > 0){
        topk_add_block(st->inner, st->d_out_score, (long)nq * n_out,
                       st->d_out_q, st->d_out_t, /*dense_rows_per_query=*/0,
                       st->base_gid,
                       max_rows_per_query, device_id);
    }

    if (st->tail_open){
        st->carry_slot = 1 - st->carry_slot;
        st->carry_live = true;
        st->carry_mol_id = st->tail_mol_id;
        st->n_carried += 1;
    } else {
        st->carry_live = false;
    }
    st->n_conformers += nb;
    st->n_molecules  += n_out;
    st->n_blocks     += 1;
    st->staged = false;
}

void moltopk_finish(MoleculeTopKState * st, int device_id){
    if (st->carry_live)
        throw std::runtime_error(
            "roshambo3.cuda: the screen ended with a molecule still open across "
            "a block boundary, so that molecule is missing from the answer. A "
            "partition must end at a molecule boundary -- which a group boundary "
            "always is.");
    if (st->staged)
        throw std::runtime_error(
            "roshambo3.cuda: a block was staged and never scored");
}

void moltopk_counters(MoleculeTopKState * st, long long * n_conformers,
                      long long * n_molecules, long long * n_carried,
                      long long * n_blocks){
    *n_conformers = st->n_conformers;
    *n_molecules  = st->n_molecules;
    *n_carried    = st->n_carried;
    *n_blocks     = st->n_blocks;
}

void moltopk_last_block(MoleculeTopKState * st, int * n_segments,
                        int * tail_segment_start){
    *n_segments = st->n_seg;
    *tail_segment_start = st->tail_seg_start;
}
