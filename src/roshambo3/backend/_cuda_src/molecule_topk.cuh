#pragma once
////////////////////////////////////////////////////////////////////////////////
/// Molecule-level top-K: a segmented argmax in front of the device top-K.
///
/// **Why this exists.** A screen returns the top K *conformers*. "Give me the
/// top 100 molecules" is a different question and has a different answer: a
/// molecule that would rank 40th on its best conformer can be missing from a
/// conformer-level top 100 that another molecule's near-duplicates filled.
/// Every enrichment factor is a compound-level statistic, so nothing
/// downstream can be computed from the conformer-level list.
///
/// **The algorithm.** A molecule scores as the best of its conformers. All of a
/// molecule's conformers are contiguous in global id order -- the builder
/// guarantees it, `roshambo3 verify` checks it, and `prep/build.py` asserts it
/// while writing -- so a molecule is a *contiguous run* of library rows, and
/// the reduction is a segmented argmax:
///
///  * Per launch, the caller stages that block's `mol_id` sidecar slice on the
///    device (8 B per conformer, and nothing else in the pipeline sends it).
///    A scan turns it into segment offsets; one warp per (query, segment)
///    reduces to the segment's best conformer.
///  * A molecule may straddle a block boundary, so the trailing segment of a
///    block is held back as a **carry** -- its winning score row and global
///    conformer id -- and merged with the leading segment of the next block.
///    The caller says whether the trailing segment is open, because only it can
///    see the sidecar past the end of the block.
///  * The surviving `(score row, winning gid)` pairs are fed to the ordinary
///    device top-K (`topk.cuh`) as a compacted dense block. That is the whole
///    integration: the top-K is not modified and does not know the rows it is
///    ranking are molecules.
///
/// **The tie-break, which is where this can go wrong.** A max over floats is
/// order-independent; an argmax is not. `beats()` below is a strict weak
/// ordering on `(score, gid)` -- higher score wins, ties go to the *lower*
/// global conformer id -- so the warp reduction, the per-lane serial scan and
/// the carry merge all compose to the same answer whatever order they visit
/// rows in. That is `roshambo3.numerics.rank_order`'s canonical tie-break,
/// applied one level down: it decides *which conformer represents a molecule*,
/// and the top-K then applies the same rule again to decide which molecules
/// win. Float32 ties are common at scale (50,737 in one measured run), so this is
/// not a theoretical concern.
///
/// A NaN never represents a molecule unless every one of its conformers is NaN,
/// in which case the emitted row is NaN and the top-K drops it -- `!(NaN >=
/// thr)` -- the conformer-level top-K's rule unchanged.
///
/// **Determinism.** The output is a pure function of the *set* of conformers
/// screened, not of block size, batch split or GPU count:
///
///  1. Within a block the segmented argmax is `beats()`-reduced over a set, so
///     it is independent of the grid and of lane assignment.
///  2. The carry makes the answer independent of *where* the block boundaries
///     fall: a molecule's segment is reduced with `beats()` whether it arrives
///     in one piece or several, because `beats()` is associative over the
///     molecule's rows.
///  3. A molecule never straddles a *group*, and a batch split is by group
///     (`R3gSource(...)`), so each partition emits each of its
///     molecules exactly once and `TopK.merge` sees no duplicates. The state
///     refuses to finish with a live carry, which is what a split through the
///     middle of a molecule would look like.
///  4. Downstream of the compaction the device top-K's argument (topk.cuh) applies verbatim: the
///     admitted set is order-independent, both sort passes are stable, and ids
///     are unique because a molecule contributes exactly one row.
////////////////////////////////////////////////////////////////////////////////

#include <cstddef>

struct DeviceTopKState;
struct MoleculeTopKState;

/// Allocate a molecule-level top-K on `device_id`. It owns the conformer-level
/// `DeviceTopKState` it feeds; the arguments are that state's, unchanged.
///
/// @param max_block_conformers  the largest library block that will be staged.
///        Sizes the segment and staging buffers once, so a later, larger block
///        is a hard error rather than a silent overrun.
MoleculeTopKState * moltopk_create(int n_query, int k, int rank_col, int dv,
                                   bool keep_cols, int device_id,
                                   long max_block_conformers);

void moltopk_destroy(MoleculeTopKState * st);

/// The wrapped conformer-level state, for readback and counters.
DeviceTopKState * moltopk_inner(MoleculeTopKState * st);

/// Stage one block's `mol_id` sidecar slice, before the launch that scores it.
///
/// @param h_mol_id   `[n_conformers]` host `mol_id`, in library-row order
/// @param gid_base   global conformer id of this block's first row
/// @param tail_open  1 if this block's last molecule continues into the next
///        block. Only the caller can know: it needs the sidecar entry one past
///        the end of the block, and it must be 0 at the end of a group because
///        a molecule never straddles one.
/// @param base_gid   the id `out_t` will be measured from. It must be <= every
///        gid this block can emit, which means the first gid of the *carried*
///        molecule when a carry is live, not this block's `gid_base`.
void moltopk_stage_block(MoleculeTopKState * st, const long long * h_mol_id,
                         long n_conformers, long long gid_base, int tail_open,
                         long long base_gid, int device_id);

/// Reduce one launch's dense score block to molecule rows and fold them in.
///
/// `d_scores` is `[n_rows, dv]` with `n_rows == n_query * n_conformers` and row
/// `q * n_conformers + t` holding query `q` against library row `t`. That
/// layout is *verified* against `h_out_q`/`h_out_t` on every call rather than
/// assumed: `optimize_pairs` takes an arbitrary pair list, and a caller that
/// passed a different one would otherwise get a confidently wrong answer. Both
/// null means a dense launch, which has that layout by construction.
void moltopk_add_block(MoleculeTopKState * st, const float * d_scores, long n_rows,
                       const int * h_out_q, const int * h_out_t,
                       long max_rows_per_query, int device_id);

/// Refuse to finish with a molecule still open. A live carry means the screen
/// stopped in the middle of a molecule, so that molecule is absent from the
/// answer and the next partition will emit it as well.
void moltopk_finish(MoleculeTopKState * st, int device_id);

void moltopk_counters(MoleculeTopKState * st, long long * n_conformers,
                      long long * n_molecules, long long * n_carried,
                      long long * n_blocks);

/// Segment count and trailing-segment start of the most recent block, read back
/// so the caller can check the device's reading of `mol_id` against its own.
void moltopk_last_block(MoleculeTopKState * st, int * n_segments,
                        int * tail_segment_start);
