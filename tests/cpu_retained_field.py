"""The retained-query-field shape route through `cpu.score_pairs`.

Shared by tests that need a block scored on the pipelined field and,
optionally, rescored for color at the winning shape pose. It uses the worker
route's field knobs but recomputes library self-overlaps from the supplied
coordinates; authenticated format-3 self values use the worker handle path.
"""
import numpy as np

from roshambo3.optimizer import lr_t_for_steps

#: The fast shape knobs the CPU worker route passes to `cpu.score_pairs`.
FAST_SHAPE = dict(field_h=.5, field_pad=8., field_width=8, field_unroll=0,
                  field_pipeline=True, field_gridxf=1, field_allin=1,
                  field_batched=True)


def shape_then_rescore(cpu, query, coords, counts, vaa=None, *, rescore=False,
                       diagnostic=False, query_fields=None,
                       final_start_policy=None):
    """Score one block on the retained shape field; optionally rescore color
    at the winning pose, refusing if that rescore moved the pose."""
    pq = np.zeros(counts.shape[0], np.int32)
    pl = np.arange(counts.shape[0], dtype=np.int32)
    rows = cpu.score_pairs(query_coords=query.coords, query_n_atoms=query.n_real,
        library_coords=coords, library_counts=counts, pair_query_row=pq,
        pair_library_row=pl, color=False, mixing=0., steps=25, start_mode=1,
        lr_q=.1, lr_t=lr_t_for_steps(25), threads=1,
        with_phase_timers=diagnostic,
        prepared_query_fields=query_fields, final_start_policy=final_start_policy,
        **FAST_SHAPE)
    shape_phases, color_phases = {}, {}
    if diagnostic:
        rows, shape_phases = rows
    if rescore:
        shaped = rows
        rows, color_phases = cpu.extension().rescore_scalar_color(
            query.coords, query.types, query.n_real, coords, counts, pq, pl,
            query.interaction_r, query.interaction_p, rows, .5,
            False, diagnostic, False, 0)
        if not np.array_equal(rows[:, 9:20], shaped[:, 9:20]):
            raise RuntimeError("color rescore changed winning shape pose")
    if diagnostic:
        return rows, {"shape_phases": shape_phases,
                      "color_phases": dict(color_phases)}
    return rows
