"""Compare complete corpus outputs by GID, independently of returned ordering."""
import argparse
import json
from pathlib import Path

import numpy as np

from bench.apple_silicon import LEGACY_CAPTURE_GIDS as GIDS


def distribution(values):
    return {"median": float(np.median(values)),
            "p99": float(np.quantile(values, .99)),
            "max": float(np.max(values))}


def compare(reference, candidate, reference_workers, candidate_workers):
    identity = [json.loads((root / "provenance.json").read_text())
                for root in (reference, candidate)]
    for key in ("queries_sha256", "corpus_sha256"):
        if identity[0][key] != identity[1][key]:
            raise ValueError(f"unmatched {key}")
    report = {"reference": identity[0], "candidate": identity[1], "queries": []}
    for mode in ("combo", "shape", "rescore"):
        for gid in GIDS:
            paths = [root / f"{mode}-inherit-{workers}-r0-g{gid}.npz"
                     for root, workers in ((reference, reference_workers),
                                           (candidate, candidate_workers))]
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                ai, bi = np.argsort(a["gids"]), np.argsort(b["gids"])
                ag, bg = a["gids"][ai], b["gids"][bi]
                if not np.array_equal(ag, bg):
                    raise ValueError(f"query {gid}/{mode} does not contain identical GIDs")
                aa, bb = a["scores"][ai], b["scores"][bi]
                if aa.shape[1] != 20 or bb.shape[1] != 20 or not (
                        np.isfinite(aa).all() and np.isfinite(bb).all()):
                    raise ValueError("invalid or nonfinite score/pose rows")
                am, bm = a["mol_ids"][ai], b["mol_ids"][bi]
                if not np.array_equal(am, bm):
                    raise ValueError("unmatched molecule identities")
                delta = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
                result = {"mode": mode, "gid": gid, "n_pairs": len(ag),
                          "identical_rows": int(np.all(aa == bb, axis=1).sum()),
                          "bit_identical_rows": int(np.all(aa.view(np.uint32) == bb.view(np.uint32), axis=1).sum()),
                          "columns": [distribution(delta[:, col]) for col in range(20)],
                          "pose_translation_delta": distribution(np.linalg.norm(aa[:, 13:16] - bb[:, 13:16], axis=1)),
                          "start_quaternion_changed": int(np.any(aa[:, 16:20] != bb[:, 16:20], axis=1).sum())}
                result["shape_score_strata"] = {}
                for lo, hi in ((0., .4), (.4, .55), (.55, .7), (.7, 1.01)):
                    mask = (aa[:, 1] >= lo) & (aa[:, 1] < hi)
                    if mask.any():
                        result["shape_score_strata"][f"{lo}-{hi}"] = distribution(delta[mask, 1])
                # Canonical score/GID order, then one winning conformer per molecule.
                lists = []
                for scores in (aa, bb):
                    order = np.lexsort((ag, -scores[:, 0]))
                    _, first = np.unique(am[order], return_index=True)
                    lists.append(order[np.sort(first)])
                result["ranking"] = {}
                for k in (1, 10, 100):
                    left, right = am[lists[0][:k]], am[lists[1][:k]]
                    left_winners = dict(zip(left.tolist(), ag[lists[0][:k]].tolist()))
                    right_winners = dict(zip(right.tolist(), ag[lists[1][:k]].tolist()))
                    result["ranking"][str(k)] = {
                        "overlap": len(set(left.tolist()) & set(right.tolist())) / len(left),
                        "identical_order": bool(np.array_equal(left, right)),
                        "reference": left.tolist(), "candidate": right.tolist(),
                        "winning_gid_changes": sum(left_winners[mid] != right_winners[mid]
                            for mid in left_winners.keys() & right_winners.keys())}
                changed = result["ranking"]["100"]
                result["review_required"] = not changed["identical_order"] or bool(changed["winning_gid_changes"])
                report["queries"].append(result)
    report["review_required"] = any(row["review_required"] for row in report["queries"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--reference-workers", type=int, default=16)
    p.add_argument("--candidate-workers", type=int, default=14)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = compare(a.reference, a.candidate, a.reference_workers, a.candidate_workers)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"review_required": report["review_required"],
                      "n_queries": len(report["queries"])}))


if __name__ == "__main__":
    main()
