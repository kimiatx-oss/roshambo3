"""Pre-registration, and the HOLDOUT ledger that makes it bite.

A pre-registration is a file written **before** a sweep runs, naming the
hypothesis, the knobs, the primary metric verbatim, the thresholds, the panel
sha256s, the dataset manifests, the planned number of runs and the planned
analysis.  Changing the metric mid-sweep does not amend it -- it starts a new
``preregistration_id``, and the old one stays on disk with whatever it
produced.  That is the whole mechanism: the record of the abandoned hypothesis
is what stops a sweep from quietly becoming a search for a metric that likes
the result.

The **HOLDOUT ledger** is the executable half.  The HOLDOUT panel is run
through :func:`touch_holdout`, which refuses to run twice on the same
``config_hash`` without a new ``preregistration_id``, and counts every touch.
The count must be reported in every write-up that quotes a HOLDOUT number,
because a held-out set touched twenty times is not held out.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

__all__ = ["PreRegistration", "HoldoutLedger", "HoldoutViolation",
           "DEFAULT_LEDGER"]

DEFAULT_LEDGER = "holdout_ledger.jsonl"


class HoldoutViolation(RuntimeError):
    pass


@dataclass
class PreRegistration:
    preregistration_id: str
    hypothesis: str
    knobs: dict                      # {knob: [levels]}
    primary_metric_verbatim: str
    thresholds: dict
    panel_sha256: dict               # {panel_name: sha256}
    dataset_manifests: dict          # {tier: manifest path or sha256}
    planned_n_runs: int
    planned_analysis: str
    created_utc: str = ""
    supersedes: str | None = None
    notes: str = ""
    selection_panel: str = "TUNE-EXT"
    confirmation_panel: str = "HOLDOUT"
    max_eval_finalists: int = 3
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_utc:
            self.created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def as_dict(self):
        return asdict(self)

    def sha256(self):
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def write(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
            if existing.get("primary_metric_verbatim") != self.primary_metric_verbatim:
                raise HoldoutViolation(
                    f"{path} already holds a pre-registration whose primary "
                    "metric differs from this one. Changing the metric is a "
                    "NEW preregistration_id, not an amendment -- write a new "
                    "file and set `supersedes`.")
        with open(path, "w") as f:
            json.dump({**self.as_dict(), "sha256": self.sha256()}, f, indent=1)
        return path

    @classmethod
    def read(cls, path):
        with open(path) as f:
            d = json.load(f)
        d.pop("sha256", None)
        return cls(**d)


class HoldoutLedger:
    """Append-only record of every touch of the held-out set."""

    def __init__(self, path=DEFAULT_LEDGER):
        self.path = path

    def entries(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def n_touches(self, panel="HOLDOUT", dataset="D3-EVAL-10M"):
        return sum(1 for e in self.entries()
                   if e.get("panel") == panel and e.get("dataset") == dataset)

    def touch(self, *, preregistration_id, config_hash, panel="HOLDOUT",
              dataset="D3-EVAL-10M", run_id=None, purpose="", force=False):
        """Record a HOLDOUT run, refusing a repeat of the same config.

        The refusal is on ``(config_hash, preregistration_id)``: re-running the
        *same* configuration under the *same* pre-registration is the exact
        motion that turns a held-out set into a tuning set, one careful
        decision at a time.  A new pre-registration id is allowed to re-touch,
        because that is a declared new experiment and the ledger will show
        both.
        """
        for e in self.entries():
            if (e.get("config_hash") == config_hash
                    and e.get("preregistration_id") == preregistration_id
                    and e.get("panel") == panel and e.get("dataset") == dataset):
                if not force:
                    raise HoldoutViolation(
                        f"{panel}/{dataset} has already been run on "
                        f"config_hash={config_hash[:12]}... under "
                        f"preregistration_id={preregistration_id} "
                        f"(at {e.get('utc')}, run_id={e.get('run_id')}). "
                        "Re-running the same configuration on held-out data "
                        "without a new pre-registration id is how a holdout "
                        "becomes a tuning set. Register a new experiment, or "
                        "quote the existing run.")
        rec = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration_id": preregistration_id,
            "config_hash": config_hash,
            "panel": panel,
            "dataset": dataset,
            "run_id": run_id,
            "purpose": purpose,
            "forced": bool(force),
            "touch_index": self.n_touches(panel, dataset) + 1,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
