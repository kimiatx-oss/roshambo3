"""Independent physical-frame checks using the published forward-error budget."""
import numpy as np
from roshambo3.reference import color as reference
from roshambo3.scores import COLUMN_INDEX


def assert_reported_color_scores(rows, query_coords, query_types, query_n_atoms,
                                 library_coords, library_counts, query_rows,
                                 library_rows, interaction_r, interaction_p,
                                 mixing=0.5):
    for row, qi, li in zip(rows, query_rows, library_rows):
        nq = int(query_n_atoms[qi])
        counts = library_counts[li].astype(int)
        nl = int(counts[0])
        qmask = query_types[qi] > 0
        qs = query_coords[qi, :nq, :3]
        qc = query_coords[qi, qmask, :3]
        qt = query_types[qi, qmask]
        ls = library_coords[li, :nl, :3]
        lc = library_coords[li, nl:int(counts.sum()), :3]
        lt = np.repeat(np.arange(1, len(counts)), counts[1:])
        quat, translation = row[9:13], row[13:16]
        # Start-mode rotation is applied to the query. Undo it to express the
        # returned transform in the original query frame.
        start_inverse = row[16:20].astype(np.float64).copy()
        start_inverse[1:] *= -1
        quat = reference.quat_multiply(start_inverse, quat)
        translation = reference.apply_pose_elementwise(
            translation[None], start_inverse, np.zeros(3))[0]
        posed_shape = reference.apply_pose_elementwise(ls, quat, translation)
        posed_color = reference.apply_pose_elementwise(lc, quat, translation)
        expected = reference.score_columns(qs, ls, qc, qt, lc, lt,
            interaction_r, interaction_p, mixing, quat, translation)
        budget = reference.fixed_pose_tolerance(qs, posed_shape, qc, qt,
            posed_color, lt, interaction_r, interaction_p, mixing,
            include_input_rounding=True)
        for name in ('tanimoto_combination', 'tanimoto_color', 'overlap_color',
                     'self_color_query', 'self_color_library'):
            error = abs(float(row[COLUMN_INDEX[name]]) - expected[name])
            bound = budget[name] + abs(expected[name]) * 2.0 ** -23
            assert error <= bound, (name, error, bound)
