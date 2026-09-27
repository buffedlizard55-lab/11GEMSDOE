"""Spatial holdout of whole fault traces: the validation that matches the contest task.

THE TASK, AS THE ORGANISERS STATE IT
------------------------------------
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
    "we have consulted with fault experts who have manually identified faults that
     are not contained within the current public USGS database. These new faults
     will comprise the test dataset for the initial prize round"
    "this set of faults is not complete and may even contain some inaccurate data"

https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516
    "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded
     from evaluation, so they do not count towards penalty terms. Re-evaluation
     will also mask/exclude the existing USGS/INGENIOUS faults."  - chrisk-dd, 2026-09-16

https://community.drivendata.org/t/where-do-you-draw-the-line/11536
    "'new fault' means 'any fault pixel not already captured by USGS/INGENIOUS'
     and can include newly mapped geometry of an existing fault system."  - chrisk-dd, 2026-09-22

WHY THE OBVIOUS SPLIT IS WRONG (measured, not assumed)
------------------------------------------------------
The first version of this module held out a random 10% of *traces* and scored the
rest of the grid with the official mask applied.  It returned a trace-holdout DTI
of 0.97 for a model that had never seen those traces.  That number is an artefact,
and the mechanism is worth writing down because it is easy to fall into:

  * the metric's kernel reaches 300 m, so a prediction on a *training* fault earns
    true-positive credit for a held-out trace that happens to run within 300 m of
    it, and
  * the official mask makes predictions on known faults free, so that credit costs
    nothing.

Held-out traces are rarely more than 300 m from a mapped neighbour, so a model can
score near 1.0 by reproducing the catalogue it was trained on.  The fix is
spatial: hold out contiguous blocks, hold out every training trace within a collar
of them, and score only inside the held-out blocks with that collar removed from
both sides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import find_objects, label

from .blocks import make_fold_plan
from .metric import DtiComponents, distance_weighted_tversky

__all__ = [
    "STRUCTURE_8",
    "extract_traces",
    "SpatialTraceHoldout",
    "make_spatial_trace_holdout",
    "leakage_probe",
]

#: 8-connectivity: a fault trace is a line, and diagonal step-through is how a
#: rasterised trace actually continues across a pixel corner.
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def extract_traces(truth) -> list[np.ndarray]:
    """Return each connected fault trace as a flat array of ``row * width + col`` indices."""
    t = np.asarray(truth).astype(bool)
    lab, n = label(t, structure=STRUCTURE_8)
    out: list[np.ndarray] = []
    if n == 0:
        return out
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == i)
        out.append((ys + sl[0].start) * t.shape[1] + (xs + sl[1].start))
    return out


@dataclass
class SpatialTraceHoldout:
    """One fold of a spatially separated, trace-level holdout."""

    train_truth: np.ndarray      # what the model may learn from
    score_truth: np.ndarray      # the "new" faults, inside the held-out blocks only
    score_region: np.ndarray     # where predictions are evaluated
    fp_exclude: np.ndarray       # known-fault pixels the official scorer removes
    n_train_traces: int
    n_score_traces: int
    train_px: int
    score_px: int
    block_px: int
    collar_px: int
    fold: int
    n_folds: int

    def score(self, pred) -> DtiComponents:
        """Official DTI of ``pred`` against the held-out traces, with the official mask."""
        return distance_weighted_tversky(pred, self.score_truth,
                                         restrict_to=self.score_region,
                                         fp_exclude=self.fp_exclude)

    def sweep(self, pred, thresholds) -> dict[float, float]:
        return {t: self.score(np.where(pred >= t, 1.0, 0.0)).dti for t in thresholds}

    def describe(self) -> dict:
        return {
            "fold": self.fold,
            "n_folds": self.n_folds,
            "block_px": self.block_px,
            "block_km_at_100m": self.block_px * 0.1,
            "collar_px": self.collar_px,
            "n_train_traces": self.n_train_traces,
            "n_score_traces": self.n_score_traces,
            "train_px": self.train_px,
            "score_px": self.score_px,
            "score_region_px": int(self.score_region.sum()),
        }


def make_spatial_trace_holdout(truth, valid, fold: int, n_folds: int = 4,
                               block_px: int = 512, collar_px: int = 12) -> SpatialTraceHoldout:
    """Hold out one spatial fold; hide every training trace inside its collar.

    ``collar_px`` must exceed the metric's R = 3 px by enough margin that no
    training trace survives within kernel range of the scored area.  The default of
    12 px (1.2 km) is deliberately generous.
    """
    from scipy.ndimage import binary_dilation

    t = np.asarray(truth).astype(bool)
    v = np.asarray(valid).astype(bool)
    plan = make_fold_plan(t.shape[0], t.shape[1], n_folds=n_folds, block_px=block_px,
                          collar_px=0, truth=t)
    held_blocks = plan.held_out_mask(fold, apply_collar=False)
    collar = binary_dilation(held_blocks, iterations=collar_px) & ~held_blocks
    scored = held_blocks & v

    train_truth = t & ~held_blocks & ~collar
    score_truth = t & scored

    tr = extract_traces(train_truth)
    sc = extract_traces(score_truth)
    return SpatialTraceHoldout(
        train_truth=train_truth,
        score_truth=score_truth,
        score_region=scored,
        fp_exclude=(t & ~scored) | ~v,
        n_train_traces=len(tr),
        n_score_traces=len(sc),
        train_px=int(train_truth.sum()),
        score_px=int(score_truth.sum()),
        block_px=block_px,
        collar_px=collar_px,
        fold=fold,
        n_folds=n_folds,
    )


def leakage_probe(truth, valid, block_px: int = 512, collar_px: int = 12) -> dict:
    """Quantify how much credit a *catalogue-only* prediction collects on held-out faults.

    If this is not small, the split is leaking and any model score on it is
    meaningless.  This is the check that the naive random-trace split fails.
    """
    from scipy.ndimage import binary_dilation

    t = np.asarray(truth).astype(bool)
    v = np.asarray(valid).astype(bool)
    plan = make_fold_plan(t.shape[0], t.shape[1], n_folds=4, block_px=block_px,
                          collar_px=0, truth=t)

    out: dict = {"block_px": block_px}
    for fold in range(4):
        held = plan.held_out_mask(fold, apply_collar=False) & v
        collar = binary_dilation(held, iterations=collar_px) & ~held
        scored_truth = t & held
        train_truth = t & ~held & ~collar
        pred = train_truth.astype(np.float32)          # the best possible "memorise" answer
        c = distance_weighted_tversky(pred, scored_truth, restrict_to=held,
                                      fp_exclude=(t & ~held) | ~v)
        out[f"fold{fold}"] = {
            "catalogue_reproduction_dti_on_heldout_faults": c.dti,
            "score_px": int(scored_truth.sum()),
        }
    out["mean_dti"] = float(np.mean([out[f"fold{i}"]["catalogue_reproduction_dti_on_heldout_faults"]
                                     for i in range(4)]))
    return out
