#!/usr/bin/env python3
"""Replay the already accepted q001 -> q2m8 geometry change on frozen regression panels.

This is a matched historical-cache analysis at the original 25-step analytic
configuration. It is deliberately separate from the 100-step routine caches
and never imports these scores through the routine cache API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (str(ROOT), str(SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from roshambo3.metrics import (  # noqa: E402
    canonical_ids,
    rank_agreement_report,
    rank_g,
    rank_order,
    recall_at,
    reduce_over_conformers,
    replicate_summary,
    signed_summary,
)
from bench import screening_regression as core  # noqa: E402

EVAL_TARGETS = ("ADA", "AMPC", "MAPK2", "ANDR", "PARP1", "RENI", "EGFR", "HS90A")
LANES = ("shape", "combo")
PROTOCOL = "historical-accepted-geometry-change-replay-v1"
EXPECTED_STEP = {"q001": 0.001, "q2m8": 0.00390625}
N_GOLD = 8
K_CANDIDATE = 16
RECALL_LIMITS = {"mean_min": 0.95, "p10_min": 0.75, "min_min": 0.75}
BIAS_LIMIT = 0.02
BOOTSTRAP_REPLICATES = 10_000


class ReceiptError(ValueError):
    """A published provenance/configuration receipt did not validate."""


class CoverageUnavailable(RuntimeError):
    """An input cache does not cover a frozen panel row."""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums(path):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    expected = {}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ReceiptError(f"malformed checksum line {line_number} in {path}")
        rel = pieces[1].strip().lstrip("*")
        if rel in expected and expected[rel] != pieces[0]:
            raise ReceiptError(f"conflicting checksum entries for {rel}")
        expected[rel] = pieces[0]
    return expected


def verify_sum(path, relative_name, checksums):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    recorded = checksums.get(relative_name)
    if recorded is None:
        raise ReceiptError(f"no SCREEN_SHA256SUMS entry for {relative_name}")
    if actual != recorded:
        raise ReceiptError(f"SHA256 mismatch for {relative_name}: {actual} != {recorded}")
    return actual


def validate_screen_manifest(tag, lane, checksums, screens_root):
    manifest_path = Path(screens_root) / tag / "dudez" / "screens.json"
    rel = f"{tag}/dudez/screens.json"
    manifest_sha = verify_sum(manifest_path, rel, checksums)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("tag") != tag or manifest.get("arm") != tag:
        raise ReceiptError(f"screen manifest tag/arm mismatch for {tag}")
    if manifest.get("pose_search") != "analytic":
        raise ReceiptError(f"{tag} is not an analytic screen")
    if not math.isclose(float(manifest.get("step", -1)), EXPECTED_STEP[tag], rel_tol=0, abs_tol=1e-15):
        raise ReceiptError(f"{tag} has unexpected coordinate step {manifest.get('step')}")
    if manifest.get("start_modes", {}).get(lane) != 1:
        raise ReceiptError(f"{tag}/{lane} did not use start_mode=1")
    cfg = manifest.get("configs", {}).get(lane)
    if not isinstance(cfg, str):
        raise ReceiptError(f"{tag}/{lane} lacks its serialized ScreenConfig")
    required = (
        "pose_search=<PoseSearch.ANALYTIC: 1>", "steps=25", "lr_t=None",
        "lr_q=0.1", "start_mode=1", "update_rule=<UpdateRule.ADAGRAD_EXP: 1>",
    )
    if any(token not in cfg for token in required):
        raise ReceiptError(f"{tag}/{lane} does not match the historical 25-step analytic config")
    if lane == "shape":
        lane_required = ("color=False",)
    else:
        lane_required = (
            "color=True", "mixing=0.5", "color_residency=<CoordResidency.GLOBAL: 1>",
            "color_stage=0", "color_placement=<ColorPlacement.IN_GRADIENT: 0>",
        )
    if any(token not in cfg for token in lane_required):
        raise ReceiptError(f"{tag}/{lane} has unexpected lane settings")
    return {
        "tag": tag,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "arm": manifest["arm"],
        "wire_quant_step": float(manifest["step"]),
        "start_mode": int(manifest["start_modes"][lane]),
        "lane_config": cfg,
        "candidate_source_sha256": manifest.get("build", {}).get(
            "cuda_src_sha256" if lane in LANES else "source_sha256"),
        "gpu_lock_held_in_original_screen": manifest.get("gpu_lock_held"),
    }


def panel_scores(cache_path, checksum_rel, checksums, target, lane, panel):
    cache_path = Path(cache_path)
    cache_sha = verify_sum(cache_path, checksum_rel, checksums)
    column = "t_shape" if lane == "shape" else "t_combination"
    with np.load(cache_path, allow_pickle=False) as archive:
        required = ("mol_id", "conf_idx", column)
        if any(key not in archive for key in required):
            raise ReceiptError(f"{cache_path} lacks required identity/score columns")
        mol = np.asarray(archive["mol_id"], np.int64)
        conf = np.asarray(archive["conf_idx"], np.int64)
        scores = np.asarray(archive[column], np.float64)
    if mol.ndim != 1 or mol.shape != conf.shape or mol.shape != scores.shape:
        raise ReceiptError(f"misaligned identity/score arrays in {cache_path}")
    if not np.isfinite(scores).all():
        raise ReceiptError(f"nonfinite scores in {cache_path}")
    order = np.lexsort((conf, mol))
    sorted_mol, sorted_conf = mol[order], conf[order]
    duplicates = ((sorted_mol[1:] == sorted_mol[:-1])
                  & (sorted_conf[1:] == sorted_conf[:-1]))
    if np.any(duplicates):
        raise ReceiptError(f"duplicate (mol_id, conf_idx) identities in {cache_path}")
    pairs = list(zip(map(int, panel["mol_id"]), map(int, panel["conf_idx"])))
    if len(set(pairs)) != len(pairs):
        raise ReceiptError(f"frozen panel has duplicate pair identities for {target}")
    selected = {(int(mol[i]), int(conf[i])): float(scores[i]) for i in range(len(mol))}
    missing = [pair for pair in pairs if pair not in selected]
    if missing:
        raise CoverageUnavailable(
            f"{cache_path} omits {len(missing)} of 128 frozen panel identities")
    result = np.asarray([selected[pair] for pair in pairs], np.float64)
    return result, {
        "path": str(cache_path),
        "sha256": cache_sha,
        "checksum_path": checksum_rel,
        "score_column": column,
        "full_cache_rows": int(len(scores)),
        "panel_rows_aligned": int(len(result)),
    }


def _hac_bins(hac):
    hac = np.asarray(hac, np.int64)
    order = np.lexsort((np.arange(hac.size), hac))
    bins = np.empty(hac.size, np.int8)
    bins[order] = np.minimum(2, np.arange(hac.size) * 3 // hac.size)
    return bins


def _query_summary(values, seed):
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return {"n": 0, "status": "unavailable"}
    if values.size == 1:
        v = float(values[0])
        return {"n": 1, "mean": v, "p10": v, "min": v, "max": v,
                "median": v, "sd": 0.0, "mean_ci95_bca": None,
                "ci_status": "singleton_no_inferential_ci"}
    return replicate_summary(values, B=BOOTSTRAP_REPLICATES,
                             rng=np.random.default_rng(seed))


def _stable_seed(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")


def _level_metrics(gold, candidate, tie_ids, hac, target):
    gold = np.asarray(gold, np.float64)
    candidate = np.asarray(candidate, np.float64)
    tie_ids = np.asarray(tie_ids)
    hac = np.asarray(hac, np.int64)
    gold_ranks = rank_g(gold, tie_ids)
    candidate_ranks = rank_g(candidate, tie_ids)
    gold_order = rank_order(gold, tie_ids)
    candidate_order = rank_order(candidate, tie_ids)
    gold_top8 = gold_order[:N_GOLD]
    candidate_top16 = candidate_order[:K_CANDIDATE]
    candidate_top16_set = set(map(int, candidate_top16))
    top1_idx = int(gold_order[0])
    exact = {str(k): bool(np.array_equal(tie_ids[gold_order[:k]],
                                        tie_ids[candidate_order[:k]]))
             for k in (1, 5, 10)}
    hac_bins = _hac_bins(hac)
    top_hac = {}
    for bucket in range(3):
        subset = gold_top8[hac_bins[gold_top8] == bucket]
        if subset.size:
            top_hac[f"tercile_{bucket + 1}"] = {
                "desired_n": int(subset.size),
                "recall_at_candidate_16": float(np.mean([
                    int(index) in candidate_top16_set for index in subset])),
            }
    delta = candidate - gold
    bias_masks = {"global": np.ones(len(delta), bool),
                  "reference_top8": np.isin(np.arange(len(delta)), gold_top8)}
    for bucket in range(3):
        mask = hac_bins == bucket
        if np.any(mask):
            bias_masks[f"target_hac_tercile_{bucket + 1}"] = mask
    bias = {}
    for name, mask in bias_masks.items():
        summary = signed_summary(delta[mask])
        summary["abs_mean_bias_gate"] = abs(summary["mean_signed"]) <= BIAS_LIMIT
        bias[name] = summary
    agreement = rank_agreement_report(
        gold[None, :], candidate[None, :], N_GOLD,
        ids=tie_ids, query_ids=[target])
    return {
        "M": int(len(gold)),
        "survivor_fraction_at_16": float(min(K_CANDIDATE, len(gold))) / float(len(gold)),
        "recall_at_8_keep_16": recall_at(gold, candidate, N_GOLD,
                                           K_CANDIDATE, ids=tie_ids),
        "top1_in_candidate_top8": int(candidate_ranks[top1_idx]) <= 8,
        "reference_top1_id": int(tie_ids[top1_idx]),
        "reference_top1_candidate_rank": int(candidate_ranks[top1_idx]),
        "exact_top_order": exact,
        "target_hac_recall": top_hac,
        "signed_score_movement": bias,
        "rank_agreement": agreement,
        "reference_top8_ids": [int(tie_ids[i]) for i in gold_top8],
        "candidate_top16_ids": [int(tie_ids[i]) for i in candidate_top16],
    }


def analyze_target(target, lane, panel_dir, screens_root, checksums, source_manifests):
    panel_meta, panel = core.load_panel(panel_dir, target)
    pairs = list(zip(map(int, panel["mol_id"]), map(int, panel["conf_idx"])))
    if len(set(pairs)) != len(pairs):
        raise ReceiptError(f"panel has duplicate (mol_id, conf_idx) for {target}")
    identities = canonical_ids(panel["mol_id"], panel["conf_idx"])
    query_path = Path(panel_meta["query_sdf"])
    query_sha = sha256_file(query_path) if query_path.is_file() else None
    if query_sha is None or query_sha != panel_meta.get("query_sdf_sha256"):
        raise ReceiptError(f"query SDF hash does not match frozen panel for {target}")
    lane_sources = {}
    panel_score_vectors = {}
    for tag in ("q001", "q2m8"):
        source = Path(screens_root) / tag / "dudez" / f"{target}.{lane}.conf.npz"
        relative = f"{tag}/dudez/{target}.{lane}.conf.npz"
        panel_score_vectors[tag], lane_sources[tag] = panel_scores(
            source, relative, checksums, target, lane, panel)
    q001 = panel_score_vectors["q001"]
    q2m8 = panel_score_vectors["q2m8"]
    levels = {
        "conformer": _level_metrics(q001, q2m8, identities,
                                    panel["hac"], target),
    }
    unique_mol, q001_mol, conf_count, q001_arg = reduce_over_conformers(
        q001, panel["mol_id"], identities)
    unique_mol_2, q2m8_mol, conf_count_2, q2m8_arg = reduce_over_conformers(
        q2m8, panel["mol_id"], identities)
    if not np.array_equal(unique_mol, unique_mol_2) or not np.array_equal(conf_count, conf_count_2):
        raise ReceiptError(f"molecule identity reduction differs across cached configs for {target}")
    mol_hac = np.asarray(panel["hac"])[q001_arg]
    molecule = _level_metrics(q001_mol, q2m8_mol, unique_mol, mol_hac, target)
    ref_top = rank_order(q001_mol, unique_mol)[:N_GOLD]
    molecule["n_selected_conformers_of_reference_top8"] = [
        int(conf_count[i]) for i in ref_top]
    molecule["median_selected_conformer_count_of_reference_top8"] = float(
        np.median(conf_count[ref_top]))
    levels["molecule"] = molecule
    comparison = {
        "levels": levels,
        "reference_source_tag": "q001",
        "fixed_and_current_are_same_historical_q001_cache": True,
    }
    source_records = {
        "q001": {**source_manifests["q001"], **lane_sources["q001"]},
        "q2m8": {**source_manifests["q2m8"], **lane_sources["q2m8"]},
    }
    return {
        "target": target,
        "query_hac": int(panel_meta["query_hac"]),
        "query_sdf": str(query_path),
        "query_sdf_sha256": query_sha,
        "panel_sha256": panel_meta["npz_sha256"],
        "panel_geometry_sha256": panel_meta["geometry_sha256"],
        "M_conformers": int(len(panel["gid"])),
        "source_cache_provenance": source_records,
        "geometry_change": {
            "reference_coordinate_step": EXPECTED_STEP["q001"],
            "candidate_coordinate_step": EXPECTED_STEP["q2m8"],
            "intentional_representation_change": True,
            "same_original_query_sdf": True,
            "reference_geometry_digest": "not present in score-only historical cache",
            "candidate_panel_geometry_sha256": panel_meta["geometry_sha256"],
        },
        "comparisons": {
            "fixed": json.loads(json.dumps(comparison)),
            "current": json.loads(json.dumps(comparison)),
        },
    }


def _all_bias_keys(rows, role, level):
    keys = set()
    for row in rows:
        keys.update(row["comparisons"][role]["levels"][level]["signed_score_movement"])
    return sorted(keys, key=lambda k: (k != "global", k != "reference_top8", k))


def summarize_lane(lane, query_rows, unavailable, expected_targets):
    result = {"lane": lane, "per_query": query_rows, "unavailable": unavailable}
    comparisons = {}
    for role in ("fixed", "current"):
        role_summary = {"levels": {}, "signed_score_movement": {}}
        for level in ("conformer", "molecule"):
            rows_with_level = [r for r in query_rows if level in r["comparisons"][role]["levels"]]
            values = [r["comparisons"][role]["levels"][level]["recall_at_8_keep_16"]
                      for r in rows_with_level]
            stats = _query_summary(values, _stable_seed((lane, role, level, "recall")))
            stats["query_targets"] = [r["target"] for r in rows_with_level]
            stats["M_by_query"] = [r["comparisons"][role]["levels"][level]["M"]
                                   for r in rows_with_level]
            stats["survivor_fraction_by_query"] = [
                r["comparisons"][role]["levels"][level]["survivor_fraction_at_16"]
                for r in rows_with_level]
            stats["gate_definition"] = dict(RECALL_LIMITS)
            stats["gate"] = (len(rows_with_level) == len(expected_targets)
                and stats.get("mean", -math.inf) >= RECALL_LIMITS["mean_min"]
                and stats.get("p10", -math.inf) >= RECALL_LIMITS["p10_min"]
                and stats.get("min", -math.inf) >= RECALL_LIMITS["min_min"])
            top1 = [r["comparisons"][role]["levels"][level]["top1_in_candidate_top8"]
                    for r in rows_with_level]
            stats["top1_lands_within_candidate_top8_all_queries"] = (
                len(top1) == len(expected_targets) and all(top1))
            rbo_values = {}
            for rbo_key in ("rbo_ext_p0.98", "rbo_ext_p0.998"):
                rbo_values[rbo_key] = [
                    r["comparisons"][role]["levels"][level]["rank_agreement"]["per_query"][0][rbo_key]
                    for r in rows_with_level]
            stats["rank_agreement"] = {
                key: _query_summary(vals, _stable_seed((lane, role, level, key)))
                for key, vals in rbo_values.items()}
            role_summary["levels"][level] = stats
            bias_summary = {}
            for key in _all_bias_keys(rows_with_level, role, level):
                per_query = [(r["target"], r["comparisons"][role]["levels"][level]
                    ["signed_score_movement"][key]["mean_signed"])
                    for r in rows_with_level
                    if key in r["comparisons"][role]["levels"][level]["signed_score_movement"]]
                values_bias = [v for _, v in per_query]
                stats_bias = _query_summary(values_bias, _stable_seed((lane, role, level, key)))
                stats_bias["query_signed_means"] = [
                    {"target": t, "mean_signed": v} for t, v in per_query]
                stats_bias["gate_definition"] = {"abs_mean_bias_max": BIAS_LIMIT,
                                                   "all_populated_queries_must_pass": True}
                stats_bias["all_populated_queries_pass"] = (
                    len(per_query) == len(rows_with_level)
                    and all(abs(v) <= BIAS_LIMIT for _, v in per_query))
                stats_bias["aggregate_mean_gate"] = (
                    stats_bias.get("n", 0) > 0
                    and abs(stats_bias.get("mean", math.inf)) <= BIAS_LIMIT)
                stats_bias["gate"] = (stats_bias["all_populated_queries_pass"]
                                       and stats_bias["aggregate_mean_gate"])
                bias_summary[key] = stats_bias
            role_summary["signed_score_movement"][level] = bias_summary
        role_summary["top1_landing_all_queries_both_levels"] = all(
            role_summary["levels"][level]["top1_lands_within_candidate_top8_all_queries"]
            for level in ("conformer", "molecule"))
        comparisons[role] = role_summary
    query_hac_order = sorted(query_rows, key=lambda r: (r["query_hac"], r["target"]))
    query_hac_groups = {}
    for i, group in enumerate(np.array_split(np.arange(len(query_hac_order)), 3), 1):
        chosen = [query_hac_order[int(j)] for j in group]
        query_hac_groups[f"tercile_{i}"] = {
            "targets": [r["target"] for r in chosen],
            "query_hac": [r["query_hac"] for r in chosen],
            "by_role": {role: {level: _query_summary([
                r["comparisons"][role]["levels"][level]["recall_at_8_keep_16"]
                for r in chosen], _stable_seed((lane, "query_hac", i, role, level)))
                for level in ("conformer", "molecule")} for role in ("fixed", "current")},
        }
    conf_count_groups = {}
    mol_rows = [r for r in query_rows
                if "median_selected_conformer_count_of_reference_top8"
                in r["comparisons"]["fixed"]["levels"]["molecule"]]
    mol_rows.sort(key=lambda r: (r["comparisons"]["fixed"]["levels"]["molecule"]
                                   ["median_selected_conformer_count_of_reference_top8"], r["target"]))
    for i, group in enumerate(np.array_split(np.arange(len(mol_rows)), 3), 1):
        chosen = [mol_rows[int(j)] for j in group]
        conf_count_groups[f"tercile_{i}"] = {
            "targets": [r["target"] for r in chosen],
            "median_selected_conformer_count_of_reference_top8": [
                r["comparisons"]["fixed"]["levels"]["molecule"]
                 ["median_selected_conformer_count_of_reference_top8"] for r in chosen],
            "by_role": {role: _query_summary([
                r["comparisons"][role]["levels"][level]["recall_at_8_keep_16"]
                for r in chosen], _stable_seed((lane, "selected_conf_count", i, role, level)))
                for level in ("conformer", "molecule") for role in ("fixed", "current")},
        }
    target_hac_recall = {}
    for role in ("fixed", "current"):
        target_hac_recall[role] = {}
        for level in ("conformer", "molecule"):
            bins = {}
            for bucket in range(1, 4):
                vals = []
                for row in query_rows:
                    hrec = row["comparisons"][role]["levels"][level]["target_hac_recall"]
                    key = f"tercile_{bucket}"
                    if key in hrec:
                        vals.append((row["target"], hrec[key]["recall_at_candidate_16"]))
                bins[f"tercile_{bucket}"] = {
                    "query_targets": [t for t, _ in vals],
                    "replicate_summary": _query_summary([v for _, v in vals],
                        _stable_seed((lane, role, level, "target_hac", bucket))),
                }
            target_hac_recall[role][level] = bins
    detector_pass = (not unavailable and len(query_rows) == len(expected_targets)
                     and tuple(expected_targets) == EVAL_TARGETS)
    detector_pass = detector_pass and all(
        comparisons[role]["levels"][level]["gate"]
        and comparisons[role]["levels"][level]["top1_lands_within_candidate_top8_all_queries"]
        for role in ("fixed", "current") for level in ("conformer", "molecule"))
    detector_pass = detector_pass and all(
        data["gate"] for role in ("fixed", "current")
        for levels in comparisons[role]["signed_score_movement"].values()
        for data in levels.values())
    result.update({
        "status": "unavailable" if unavailable else "complete",
        "historical_fixed_current_relation": "same q001 cache for both labels; not independent routine caches",
        "comparisons": comparisons,
        "query_hac_terciles": query_hac_groups,
        "selected_conformer_count_terciles": conf_count_groups,
        "target_hac_top8_recall_diagnostics": target_hac_recall,
        "detector_pass": bool(detector_pass),
        "frozen_local_gates": {"recall": dict(RECALL_LIMITS),
                                "top1_in_candidate_top8": True,
                                "absolute_signed_mean_bias_max": BIAS_LIMIT,
                                "bias_gated_per_query_and_aggregate": True},
    })
    return result


def run(panel_dir, screens_root, checksums_path, targets, lanes):
    checksums = read_checksums(checksums_path)
    checksum_file_sha = sha256_file(checksums_path)
    manifests = {}
    for tag in ("q001", "q2m8"):
        # Validate a lane-specific manifest below; keep its shared provenance here.
        path = Path(screens_root) / tag / "dudez" / "screens.json"
        rel = f"{tag}/dudez/screens.json"
        sha = verify_sum(path, rel, checksums)
        manifests[tag] = json.loads(path.read_text())
        if manifests[tag].get("tag") != tag or manifests[tag].get("arm") != tag:
            raise ReceiptError(f"screen manifest tag/arm mismatch for {tag}")
        if manifests[tag].get("pose_search") != "analytic":
            raise ReceiptError(f"{tag} is not an analytic screen")
        if not math.isclose(float(manifests[tag].get("step", -1)), EXPECTED_STEP[tag],
                            rel_tol=0, abs_tol=1e-15):
            raise ReceiptError(f"{tag} has unexpected coordinate step")
        manifests[tag]["_validated_path"] = str(path)
        manifests[tag]["_validated_sha256"] = sha
    output = {
        "protocol": PROTOCOL,
        "status": "running",
        "scope": "matched cached-score replay only; no native scoring and no timing claims",
        "accepted_change": "q001 coordinate step 0.001 to q2m8 coordinate step 0.00390625 (2^-8)",
        "historical_config": {
            "pose_search": "analytic", "steps": 25, "lr_q": 0.1,
            "lr_t_effective": 0.4, "start_mode": 1,
            "shape": "analytic shape", "combo": "mixing 0.5, IN_GRADIENT, color stage 0, GLOBAL color residency",
        },
        "fixed_current_reference_note": "fixed and current are both the same historical q001 cache, explicitly labeled; these are not the independent 100-step routine references",
        "checksums_receipt": {"path": str(checksums_path),
                               "sha256": checksum_file_sha},
        "source_manifests": {},
        "targets": list(targets),
        "lanes": {},
        "unavailable": [],
        "integrity_errors": [],
        "limitations": [
            "The panel is challenger-enriched and does not estimate whole-library recall.",
            "The historical q001 cache contains scores and identities, not a per-panel q001 coordinate digest; the declared wire-step change is intentional.",
            "The same original query SDF SHA is verified against each frozen panel; query preparation is otherwise inherited from the historical cached screens.",
            "The q001 cache is reused for both fixed and current labels for this accepted-transition replay.",
        ],
    }
    for tag in ("q001", "q2m8"):
        for lane in lanes:
            output["source_manifests"].setdefault(tag, {})[lane] = validate_screen_manifest(
                tag, lane, checksums, screens_root)
    for lane in lanes:
        rows, unavailable = [], []
        for target in targets:
            try:
                rows.append(analyze_target(target, lane, panel_dir, screens_root,
                                           checksums, {tag: output["source_manifests"][tag][lane]
                                                       for tag in ("q001", "q2m8")}))
            except (CoverageUnavailable, core.Unavailable) as exc:
                unavailable.append({"target": target, "reason": str(exc)})
            except FileNotFoundError as exc:
                unavailable.append({"target": target, "reason": f"missing input: {exc}"})
        output["lanes"][lane] = summarize_lane(lane, rows, unavailable, targets)
    any_unavailable = any(output["lanes"][lane]["unavailable"] for lane in lanes)
    full_scope = tuple(targets) == EVAL_TARGETS and tuple(lanes) == LANES
    all_pass = all(output["lanes"][lane]["detector_pass"] for lane in lanes)
    output["status"] = "unavailable" if any_unavailable else "complete"
    output["qualification_scope_complete"] = bool(full_scope)
    output["detector_pass"] = bool(all_pass and not any_unavailable and full_scope)
    return output


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--screens-root", type=Path, required=True)
    p.add_argument("--checksums", type=Path, required=True)
    p.add_argument("--target", choices=EVAL_TARGETS, action="append")
    p.add_argument("--lane", choices=LANES, action="append")
    p.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite accepted-change receipt: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    targets = tuple(args.target or EVAL_TARGETS)
    lanes = tuple(args.lane or LANES)
    try:
        report = run(args.panel_dir, args.screens_root, args.checksums, targets, lanes)
    except (FileNotFoundError, CoverageUnavailable) as exc:
        report = {"protocol": PROTOCOL,
                  "status": "unavailable", "detector_pass": False,
                  "reason": str(exc), "targets": list(targets), "lanes": list(lanes)}
    except ReceiptError as exc:
        report = {"protocol": PROTOCOL,
                  "status": "integrity_error", "detector_pass": False,
                  "reason": str(exc), "targets": list(targets), "lanes": list(lanes)}
    report_path = args.output / "accepted-change.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report.get("status"),
                      "detector_pass": report.get("detector_pass"),
                      "report": str(report_path)}, indent=2))
    if report.get("status") == "integrity_error":
        return 3
    if report.get("status") == "unavailable":
        return 2
    return 0 if report.get("detector_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
