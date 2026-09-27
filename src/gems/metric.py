"""Distance-weighted Tversky index (DW-Tversky / "DTI") - the GEMS contest metric.

SOURCE OF EVERY CONSTANT AND FORMULA BELOW
-----------------------------------------
Official problem page, "Performance metric" section
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
(fetched 2026-09-27, quoted verbatim below):

    "The performance metric for this competition is a distance-weighted Tversky
     index. ... For this challenge, you will submit a GeoTIFF raster with fault
     predictions represented as pixel-wise probabilities or confidence scores
     between 0 and 1."

    "To mitigate these effects, we weight the contributions of true positives,
     false negatives, and false positives by the distance to the nearest ground
     truth pixel using a linear (triangular) kernel with 300 m support."

    k(d)    = (1 - d/R)_+ = max(1 - d/R, 0)          R = 300 m = 3 px at 100 m
    TP_w    = sum_{g in G}   max_{x : d(x,g) <= R}  p(x) * k(d(x,g))
    FP_w    = sum_{x : p(x) > 0}  p(x) * [1 - max_{g in G} k(d(x,g))]
    FN_w    = sum_{g in G}   [1 - max_{x : d(x,g) <= R} p(x) * k(d(x,g))]
    DTI(a,b)= TP_w / (TP_w + a*FP_w + b*FN_w + eps)

    "For this competition, we set alpha = 0.2 and beta = 0.8, which reduces the
     penalty for false positive predictions and increases the penalty for false
     negative predictions."

    Official worked example, same page: "TP_w = 3.00, FP_w = 1.89, FN_w = 2.00"
    and "TI_w = 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.60".

IMPLEMENTATION NOTES (all verifiable, no tuning)
------------------------------------------------
* ``d`` is the Euclidean distance **in pixels**.  The page defines R = 300 m and
  states the raster is 100 m, so R = 3 px; the kernel is a continuous function
  of a metric distance, and Euclidean is the only reading consistent with
  "triangular kernel over a disk".  The 29 integer offsets with
  ``dx^2 + dy^2 <= 9`` are exactly the pixels with ``d <= 3``.
* Identity used as an internal consistency check (it follows from the two sums,
  for ANY p): ``TP_w + FN_w = |G|``.
* ``brute_force`` below is a literal, loop-for-loop transcription of the three
  displayed equations.  ``distance_weighted_tversky`` is the optimised version.
  ``tests/test_metric.py`` pins them against each other and against the official
  worked example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "GtScorer",
    "ALPHA",
    "BETA",
    "R_PIXELS",
    "EPS",
    "OFFICIAL_EXAMPLE",
    "DtiComponents",
    "kernel_offsets",
    "credit_map",
    "fp_weight_map",
    "distance_weighted_tversky",
    "brute_force",
    "sanitise_prediction",
    "dti_ceiling",
]

#: alpha - false-positive penalty weight (official problem page).
ALPHA = 0.2
#: beta - false-negative penalty weight (official problem page).
BETA = 0.8
#: kernel support radius in pixels (300 m at 100 m resolution).
R_PIXELS = 3.0
#: guard so an all-zero prediction against empty truth cannot divide by zero.
EPS = 1e-7

#: The three component values printed on the official scoring-example panel,
#: together with the DTI the page reports for them.  Used as a regression test.
OFFICIAL_EXAMPLE = {
    "TP_w": 3.00,
    "FP_w": 1.89,
    "FN_w": 2.00,
    "alpha": 0.2,
    "beta": 0.8,
    "reported_dti": 0.60,
}


@dataclass(frozen=True)
class DtiComponents:
    """The three weighted sums the official page defines, plus the resulting DTI."""

    tp: float
    fp: float
    fn: float
    dti: float
    n_truth: int

    def as_dict(self) -> dict:
        return {
            "TP_w": self.tp,
            "FP_w": self.fp,
            "FN_w": self.fn,
            "DTI": self.dti,
            "n_truth_pixels": self.n_truth,
        }


def kernel_offsets(r_pixels: float = R_PIXELS) -> list[tuple[int, int, float]]:
    """Exhaustive ``(dy, dx, k)`` list for ``d(x, g) <= r_pixels``.

    At the official R = 3 this returns the 29 integer offsets inside the disk of
    radius 3, including ``(0, 0, 1.0)``.  No offset is dropped and none is added.
    """
    r = int(math.ceil(r_pixels))
    out: list[tuple[int, int, float]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = math.hypot(dy, dx)
            if d <= r_pixels + 1e-12:
                out.append((dy, dx, max(1.0 - d / r_pixels, 0.0)))
    return out


def sanitise_prediction(pred, shape: tuple[int, int]) -> np.ndarray:
    """Apply the submission-format contract once, in one place.

    Official page, "Submission format": "a single layer with datatype of 32-bit
    float (float32) with values between 0 and 1" and "data outside the bounds is
    null or nan".  So NaN/Inf become zero credit, and out-of-range values are
    clipped rather than trusted.  Clipping here is *not* a licence to ship an
    out-of-range file: ``scripts/validate_submission.py`` fails on any raw value
    outside [0, 1] before this function ever sees it.
    """
    arr = np.asarray(pred, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != tuple(shape):
        raise ValueError(f"prediction shape {arr.shape} != truth shape {tuple(shape)}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _shifted_max(acc: np.ndarray, values: np.ndarray, dy: int, dx: int, weight: float) -> None:
    """``acc[i, j] = max(acc[i, j], weight * values[i + dy, j + dx])``, zero padding.

    Zero padding is the right convention: the metric sums over pixels of the
    raster, and there is no prediction outside the raster.
    """
    h, w = values.shape
    src_y0, src_y1 = max(0, dy), min(h, h + dy)
    dst_y0, dst_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, dx), min(w, w + dx)
    dst_x0, dst_x1 = max(0, -dx), min(w, w - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return
    if weight == 1.0:
        block = values[src_y0:src_y1, src_x0:src_x1]
    else:
        block = values[src_y0:src_y1, src_x0:src_x1] * weight
    np.maximum(acc[dst_y0:dst_y1, dst_x0:dst_x1], block,
               out=acc[dst_y0:dst_y1, dst_x0:dst_x1])


def credit_map(pred: np.ndarray, r_pixels: float = R_PIXELS) -> np.ndarray:
    """``credit[g] = max_{x: d(x,g) <= R} p(x) * k(d(x,g))`` for every pixel ``g``.

    Implemented as a shift-and-max over the 29 kernel offsets, which is exactly
    the definition with the max distributed over the finite offset set.
    """
    acc = np.zeros(pred.shape, dtype=np.float64)
    for dy, dx, k in kernel_offsets(r_pixels):
        _shifted_max(acc, pred, dy, dx, k)
    return acc


def fp_weight_map(truth: np.ndarray, r_pixels: float = R_PIXELS) -> np.ndarray:
    """``1 - max_{g in G} k(d(x,g))`` for every pixel ``x`` (the FP weight).

    ``max_g k(d(x,g)) = max(1 - dist(x, G)/R, 0)`` because ``k`` is monotonically
    decreasing in ``d``; so one Euclidean distance transform of the truth suffices.
    """
    truth_bool = np.asarray(truth).astype(bool)
    if not truth_bool.any():
        return np.ones(truth_bool.shape, dtype=np.float64)
    dist = distance_transform_edt(~truth_bool)
    return 1.0 - np.maximum(1.0 - dist / r_pixels, 0.0)


def distance_weighted_tversky(
    pred,
    truth,
    alpha: float = ALPHA,
    beta: float = BETA,
    r_pixels: float = R_PIXELS,
    eps: float = EPS,
    restrict_to: np.ndarray | None = None,
    fp_exclude: np.ndarray | None = None,
) -> DtiComponents:
    """Score one prediction raster against one truth raster.

    ``restrict_to`` limits the *spatial extent* of the evaluation (used by the
    blocked holdout to score only held-out geography).  It never changes the
    distance geometry: the kernel still sees the full truth raster, which is what
    the official definition requires.

    ``fp_exclude`` implements the staff-confirmed scoring mask:

        "Pixels corresponding to known USGS/INGENIOUS faults are masked /
         excluded from evaluation, so they do not count towards penalty terms.
         Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults."
        -- chrisk-dd, DrivenData Staff, 2026-09-16
        https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516

    Pixels inside ``fp_exclude`` are removed from the false-positive sum only.
    True positives and false negatives are unchanged, because the excluded pixels
    are by definition not ground truth for this round.
    """
    truth = np.asarray(truth)
    pred_s = sanitise_prediction(pred, truth.shape)
    truth_bool = truth.astype(bool)
    n_truth_all = int(truth_bool.sum())

    if restrict_to is not None:
        keep = np.asarray(restrict_to).astype(bool)
        truth_bool = truth_bool & keep
        pred_s = np.where(keep, pred_s, 0.0)

    n_truth = int(truth_bool.sum())
    if n_truth == 0:
        scored = pred_s if fp_exclude is None else np.where(np.asarray(fp_exclude).astype(bool),
                                                            0.0, pred_s)
        fp = float(scored.sum())
        dti = 0.0 if fp > 0.0 else 1.0
        return DtiComponents(0.0, fp, 0.0, dti, 0)

    credit = credit_map(pred_s, r_pixels)
    tp = float(credit[truth_bool].sum())
    # FN_w = sum_g [1 - credit(g)] over the same truth set, so it is |G| - TP_w.
    fn = float(n_truth - tp)
    penalty = pred_s * fp_weight_map(truth_bool, r_pixels)
    if fp_exclude is not None:
        penalty = np.where(np.asarray(fp_exclude).astype(bool), 0.0, penalty)
    fp = float(penalty.sum())
    dti = tp / (tp + alpha * fp + beta * fn + eps)
    return DtiComponents(tp, fp, fn, dti, n_truth_all)


def brute_force(pred, truth, alpha: float = ALPHA, beta: float = BETA,
                r_pixels: float = R_PIXELS, eps: float = EPS) -> DtiComponents:
    """Literal transcription of the three official equations, used to check the fast path.

    Deliberately naive: it loops over ground-truth pixels and over the kernel
    offsets one at a time with no vectorisation trickery.
    """
    truth_bool = np.asarray(truth).astype(bool)
    pred_s = sanitise_prediction(pred, truth_bool.shape)
    gy, gx = np.nonzero(truth_bool)
    h, w = truth_bool.shape
    offsets = kernel_offsets(r_pixels)

    tp = 0.0
    for y, x in zip(gy, gx):
        best = 0.0
        for dy, dx, k in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                best = max(best, pred_s[yy, xx] * k)
        tp += best
    fn = float(len(gy) - tp)

    fp = 0.0
    ys, xs = np.nonzero(pred_s > 0)
    for y, x in zip(ys, xs):
        kmax = 0.0
        for dy, dx, k in offsets:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and truth_bool[yy, xx]:
                kmax = max(kmax, k)
        fp += pred_s[y, x] * (1.0 - kmax)

    dti = tp / (tp + alpha * fp + beta * fn + eps) if len(gy) else 0.0
    return DtiComponents(tp, fp, fn, dti, len(gy))


def dti_ceiling(truth: np.ndarray, r_pixels: float = R_PIXELS) -> float:
    """Best DTI achievable by *any* prediction that never predicts outside ``R`` of truth.

    A prediction that is 1.0 exactly on a dilated truth and 0 elsewhere realises
    this.  It is the honest ceiling for a "no invented geology" submission and is
    what the blocked holdout numbers should be read against.
    """
    from scipy.ndimage import binary_dilation

    truth_bool = np.asarray(truth).astype(bool)
    if not truth_bool.any():
        return 0.0
    r = int(math.ceil(r_pixels))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (yy * yy + xx * xx) <= r_pixels**2 + 1e-12
    return float(distance_weighted_tversky(binary_dilation(truth_bool, disk).astype(np.float32),
                                           truth_bool).dti)


class GtScorer:
    """Fast repeated scoring against one fixed truth raster.

    WHY THIS EXISTS
    ---------------
    The official definition is three sums, and two of them (TP_w, FN_w) are sums
    over *ground-truth pixels* - about 15,000 of them on a held-out fold, not
    12 million.  Recomputing a full-raster credit map for every one of a few
    hundred emission candidates is two orders of magnitude more work than the
    definition needs, and it is what makes a threshold/width sweep affordable on
    two cores.

    Equivalence is not assumed: ``tests/test_metric.py::test_gt_scorer_matches_the_general_path``
    compares this against ``distance_weighted_tversky`` on random arrays, and the
    identity ``TP_w + FN_w = |G|`` is checked on every use.

    The false-positive term still needs a per-pixel sum over the whole evaluated
    region, so the label distance transform is built once and reused.
    """

    def __init__(self, truth, valid_mask=None, r_pixels: float = R_PIXELS,
                 fp_exclude=None, region=None):
        from scipy.ndimage import distance_transform_edt

        t = np.asarray(truth).astype(bool)
        region_mask = (np.ones(t.shape, dtype=bool) if region is None
                       else np.asarray(region).astype(bool))
        t = t & region_mask
        self.shape = t.shape
        self.region = region_mask
        self.r = float(r_pixels)
        self.offsets = kernel_offsets(r_pixels)
        self.truth = t
        self.n_truth = int(t.sum())
        self.gy, self.gx = np.nonzero(t)
        if fp_exclude is None:
            self.exclude = np.zeros(t.shape, dtype=bool)
        else:
            self.exclude = np.asarray(fp_exclude).astype(bool)
        if valid_mask is not None:
            self.exclude = self.exclude | ~np.asarray(valid_mask).astype(bool)
        self.fp_weight = (1.0 - np.maximum(1.0 - distance_transform_edt(~t) / self.r, 0.0)
                          if self.n_truth else np.ones(t.shape, dtype=np.float64))
        # `restrict_to` in the general path also zeroes the prediction outside the
        # scored region. Omitting that here silently charges every pixel the model
        # emits outside the fold as a false positive - a defect found on 2026-09-27
        # that made a whole emission sweep read ten times too low.
        self.exclude = self.exclude | ~region_mask
        self.fp_weight[self.exclude] = 0.0

    def score(self, pred, alpha: float = ALPHA, beta: float = BETA, eps: float = EPS) -> DtiComponents:
        if self.n_truth == 0:
            p = sanitise_prediction(pred, self.shape)
            fp = float(p.sum())
            return DtiComponents(0.0, fp, 0.0, 0.0 if fp > 0 else 1.0, 0)
        p = sanitise_prediction(pred, self.shape)
        # match `restrict_to` exactly: outside the scored region the prediction is
        # zero for the true-positive gather as well as the false-positive sum
        p = np.where(self.region, p, 0.0)
        h, w = self.shape
        credit = np.zeros(self.n_truth, dtype=np.float64)
        for dy, dx, k in self.offsets:
            yy = self.gy + dy
            xx = self.gx + dx
            inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            if not inside.any():
                continue
            vals = np.zeros(self.n_truth, dtype=np.float64)
            vals[inside] = p[yy[inside], xx[inside]] * k
            np.maximum(credit, vals, out=credit)
        tp = float(credit.sum())
        fn = float(self.n_truth - tp)
        fp = float((p * self.fp_weight).sum())
        return DtiComponents(tp, fp, fn, tp / (tp + alpha * fp + beta * fn + eps), self.n_truth)
