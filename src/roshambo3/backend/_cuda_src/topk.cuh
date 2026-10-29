#pragma once
////////////////////////////////////////////////////////////////////////////////
/// The device-resident top-K.
///
/// **Why this exists.** Without it the *sparse* launcher writes a dense
/// `[n_rows, 20]` score block, copies all of it over PCIe, and the host does
/// the top-K. Measured on 99,970,265 conformers x 8 queries, that is
/// 799,762,120 rows offered to the host, 63.98 GB of D2H, and -- once the
/// kernel was cut from 74.1 s to 21.7 s -- **16.708 s of exposed host
/// reduce on a 45.431 s critical path (36.8%)**. The reduce work did not grow;
/// it stopped fitting in the kernel's shadow.
///
/// This module moves it. The scores never leave the device: a compaction kernel
/// admits only rows that can still enter the top-K, and the running top-K
/// itself stays resident across batches. The host sees K rows per query, once,
/// at the end.
///
/// **The algorithm.**
///
///  * Per query, a candidate buffer of `cap` rows plus a scalar threshold
///    `thr[q]`, both on the device. A row is admitted iff `score >= thr[q]`,
///    claimed with one `atomicAdd`.
///  * When a buffer would not have room for the next block, it is *selected*:
///    reduced on the device to the best `k` rows in canonical order, and
///    `thr[q]` is raised to the score of the new k-th.
///  * `cap >= k + 2 * max_rows_per_query` guarantees a block can never overflow
///    a buffer, so no row that could enter the top-K is ever dropped. An
///    overflow flag is carried anyway and raised as an exception rather than
///    silently losing rows.
///
/// **Determinism** (a standing requirement; see also
/// `roshambo3/numerics.py::rank_order` and `doc/NUMERICS_POLICY.md`).
///
///  The output is a pure function of the *set* of conformers screened -- not of
///  block size, grid size, launch configuration, batch split, or GPU count.
///  Three facts compose to give that:
///
///  1. **The admitted set is order-independent.** On a launch that does not
///     overflow, the buffer's contents are exactly `{rows : score >= thr[q]}`.
///     Which slot a row lands in depends on `atomicAdd` arbitration and is not
///     reproducible; the *set* is.
///  2. **The selection is a function of the set, not of the buffer order.**
///     `select_one()` sorts twice: first ascending by conformer id, then
///     ascending by the bitwise complement of an order-preserving u32 encoding
///     of the score. Conformer ids are unique, so after the first pass the
///     sequence is fully determined by the set. Both passes use CUB's radix
///     sort, which is stable, so the second pass leaves equal scores in
///     ascending-id order. That is exactly `rank_order`'s canonical order
///     (descending score, ties by ascending id) -- computed on the device.
///  3. **Any valid threshold gives the same answer.** `thr[q]` only has to
///     satisfy `thr <= (true k-th largest score seen so far)`. Selection keeps
///     the exact best `k` of a superset of the true top-K, so the invariant
///     "true top-K so far is a subset of what is held" holds inductively, and
///     the k-th score of a superset's top-K is >= the true k-th. When the
///     selection points fall differently -- different block size, different
///     batch split -- the thresholds differ and the final answer does not.
///
///  A NaN score fails `>=` and is dropped rather than ranked.
///
/// Verified against a dense host reference on a 1e7-pair sample.
////////////////////////////////////////////////////////////////////////////////

#include <cstddef>

struct DeviceTopKState;

/// Allocate a top-K state on `device_id`.
/// @param n_query   number of query rows tracked independently
/// @param k         K
/// @param rank_col  which of the DV score columns is ranked on
/// @param dv        columns per score row (20 on every shipped path)
/// @param keep_cols keep all `dv` columns for surviving rows (else score+id only)
DeviceTopKState * topk_create(int n_query, int k, int rank_col, int dv,
                          bool keep_cols, int device_id);

void topk_destroy(DeviceTopKState * st);

/// Fold one launch's dense score block into the running top-K, on the device.
///
/// @param d_scores  `[n_rows, dv]` device scores, as the sparse launcher wrote them
/// @param n_rows    rows in this launch (all queries together)
/// @param d_out_q   `[n_rows]` device: output row -> query row, or null for a
///                  dense block (then `d_out_t` must be null too)
/// @param d_out_t   `[n_rows]` device: output row -> library row within the block
/// @param dense_rows_per_query  for a dense block only: the library rows per
///        query. Row `q * dense_rows_per_query + t` is query `q` against library
///        row `t`, and `n_rows` must be `n_query * dense_rows_per_query`.
///        Ignored when a pair list is given.
/// @param gid_base  added to the library row to give the global conformer id
/// @param max_rows_per_query  upper bound on rows of any single query in this
///        launch; sizes the overflow headroom. Pass `n_rows` if unknown.
void topk_add_block(DeviceTopKState * st, const float * d_scores, long n_rows,
                        const int * d_out_q, const int * d_out_t,
                        long dense_rows_per_query,
                        long long gid_base, long max_rows_per_query,
                        int device_id);

/// Reduce every query's buffer to its best `k` rows in canonical order.
void topk_flush(DeviceTopKState * st, int device_id);

/// Flush, then copy the result out. `h_scores` is `[n_query, k, dv]` (only when
/// the state was created with `keep_cols`), `h_ids` is `[n_query, k]`,
/// `h_counts` is `[n_query]` and receives the number of rows actually held
/// (`<= k`). Rows beyond `h_counts[q]` are left untouched.
void topk_result(DeviceTopKState * st, float * h_scores, long long * h_ids,
                     int * h_counts, int device_id);

/// Running thresholds, one per query (-inf until `k` rows are held).
void topk_thresholds(DeviceTopKState * st, float * h_thr);

void topk_counters(DeviceTopKState * st, long long * n_offered,
                       long long * n_admitted, long long * n_select,
                       long long * bytes_d2h);

int  topk_n_query(DeviceTopKState * st);
int  topk_k(DeviceTopKState * st);
int  topk_dv(DeviceTopKState * st);
long long topk_capacity_bytes(DeviceTopKState * st);


////////////////////////////////////////////////////////////////////////////////
/// The self-overlap bound scan: a count of the rows in a dense score block whose
/// pair overlap exceeds the mean of their self-overlaps.
///
/// Gaussian overlaps obey V_AB <= sqrt(V_AA V_BB) <= (V_AA + V_BB) / 2,
/// so `2 V_AB > (V_AA + V_BB)(1 + margin)` cannot happen on consistent inputs.
/// It is exactly a shape Tanimoto above 1 and holds wherever the Tanimoto
/// denominator is not positive -- which every backend reports as 0, so such a
/// row ranks LAST and no check of the returned top-K can see it. This scan runs
/// over every row of the block, on the device, before any top-K reduction or
/// copy-out, so the clamped rows are counted too.
///
/// The predicate is the CPU worker route's, term for term
/// (`_cpu_src/worker_local_topk_sink.hpp::check_self_overlap_bound` and
/// `roshambo3.scores.check_self_overlap_bound`): the float32 columns
/// OVERLAP_VOLUME (3), SELF_OVERLAP_QUERY (5) and SELF_OVERLAP_LIBRARY (6),
/// widened to double, compared in double. `margin` is the recompute route's
/// `roshambo3.scores.self_overlap_bound_margin("recompute")`, passed in by
/// the caller. A NaN
/// row fails the comparison and is not counted, as on the CPU.
///
/// It reads the score block and writes only `d_counters`, so it cannot change
/// a score; a caller that does not ask for it launches nothing. The result is
/// order-independent (an atomic count and an atomic minimum), so it is
/// deterministic whatever the launch configuration.
///
/// @param d_scores    `[n_rows, dv]` device scores
/// @param d_counters  device scratch of two `unsigned long long`, overwritten
/// @param h_flagged   receives the number of rows past the bound
/// @param h_first_row receives the lowest flagged row index, or -1 if none
void self_overlap_bound_scan(const float * d_scores, long n_rows, int dv,
                             double margin, unsigned long long * d_counters,
                             unsigned long long * h_flagged,
                             long long * h_first_row, int device_id);
