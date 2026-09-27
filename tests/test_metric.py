"""Metric verification: the official equations, the official worked example, and parity.

Everything asserted here is traceable to
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
(Performance metric + Scoring example + Submission format).
"""

from __future__ import annotations

import numpy as np
import pytest

from gems.metric import (
    ALPHA, BETA, EPS, OFFICIAL_EXAMPLE, R_PIXELS, brute_force, credit_map,
    distance_weighted_tversky, dti_ceiling, fp_weight_map, kernel_offsets,
)


def test_constants_match_the_official_page():
    # "we set alpha = 0.2 and beta = 0.8"; "the range R is 300 meters (i.e., 3 pixels
    # at 100m resolution)".
    assert (ALPHA, BETA, R_PIXELS) == (0.2, 0.8, 3.0)


def test_kernel_offsets_are_the_full_disk_of_radius_3():
    offs = kernel_offsets(3.0)
    assert len(offs) == 29  # {(dx,dy) : dx^2 + dy^2 <= 9}
    assert (0, 0, 1.0) in offs
    assert (3, 0, 0.0) in offs  # k(3) = max(1 - 3/3, 0) = 0, and it is inside the disk
    assert (4, 0)[:2] not in [(o[0], o[1]) for o in offs]
    # the only zero-weight offsets are the four pixels at exactly d = 3, the only
    # integer solutions of dx^2 + dy^2 = 9
    zero = sorted((dy, dx) for dy, dx, k in offs if k == 0.0)
    assert zero == [(-3, 0), (0, -3), (0, 3), (3, 0)]
    for dy, dx, k in offs:
        d = np.hypot(dy, dx)
        assert k == pytest.approx(max(1.0 - d / 3.0, 0.0))


def test_official_worked_example_is_reproduced_component_by_component():
    """The page prints TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 and DTI = 0.60.

    The page's worked example is "a single vertical line" of ground truth; the
    schematic is an image, so we rebuild the smallest configuration that has the
    page's three component values and check the implementation returns them.

    Truth: 5 pixels in a vertical line, the lower 3 covered by p = 1, the upper 2
    (8 rows away) with no prediction within the 3 px kernel -> FN_w = 5 - 3 = 2.00.
    Predictions: three pixels one step to the side of the covered truth
    (k = 2/3, so each contributes p*(1-k) = 1/3) and one pixel three steps to the
    side carrying p = 0.89 (k = 0, contributes 0.89) -> FP_w = 3*(1/3) + 0.89 = 1.89.
    """
    truth = np.zeros((14, 12), dtype=bool)
    truth[[0, 1, 2, 8, 9], 0] = True

    pred = np.zeros_like(truth, dtype=np.float32)
    pred[[0, 1, 2], 0] = 1.0
    pred[[0, 1, 2], 1] = 1.0
    pred[0, 3] = 0.89

    got = distance_weighted_tversky(pred, truth)
    assert got.tp == pytest.approx(OFFICIAL_EXAMPLE["TP_w"], abs=1e-9)
    assert got.fn == pytest.approx(OFFICIAL_EXAMPLE["FN_w"], abs=1e-9)
    assert got.fp == pytest.approx(OFFICIAL_EXAMPLE["FP_w"], abs=1e-6)  # p=0.89 is float32
    # and the page's own arithmetic on those numbers
    manual = 3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00)
    assert manual == pytest.approx(0.6026, abs=5e-4)  # page rounds this to 0.60
    assert got.dti == pytest.approx(manual, abs=1e-7)  # p=0.89 stored as float32


def test_tp_plus_fn_equals_truth_count():
    """Follows from the two official sums; used here as a cheap invariant."""
    rng = np.random.default_rng(0)
    truth = rng.random((60, 60)) < 0.03
    pred = rng.random((60, 60))
    got = distance_weighted_tversky(pred, truth)
    assert got.tp + got.fn == pytest.approx(got.n_truth)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fast_path_matches_literal_transcription(seed):
    rng = np.random.default_rng(seed)
    shape = (40, 37)
    truth = rng.random(shape) < 0.05
    pred = rng.random(shape)
    fast = distance_weighted_tversky(pred, truth)
    slow = brute_force(pred, truth)
    assert fast.tp == pytest.approx(slow.tp, abs=1e-9)
    assert fast.fp == pytest.approx(slow.fp, abs=1e-9)
    assert fast.fn == pytest.approx(slow.fn, abs=1e-9)
    assert fast.dti == pytest.approx(slow.dti, abs=1e-12)


def test_nan_and_out_of_range_handling():
    rng = np.random.default_rng(7)
    truth = np.zeros((30, 30), dtype=bool)
    truth[10:13, 10:13] = True
    pred = rng.random((30, 30)).astype(np.float32)
    pred[0, 0] = np.nan
    pred[1, 1] = 5.0        # must be clipped, never trusted
    pred[2, 2] = -3.0
    got = distance_weighted_tversky(pred, truth)
    assert np.isfinite(got.dti)
    assert 0.0 <= got.dti <= 1.0


def test_perfect_prediction_scores_one():
    truth = np.zeros((40, 40), dtype=bool)
    truth[5, 5:35] = True
    got = distance_weighted_tversky(truth.astype(np.float32), truth)
    assert got.dti == pytest.approx(1.0, abs=1e-6)
    assert got.fp == pytest.approx(0.0, abs=1e-9)


def test_empty_prediction_scores_zero_against_non_empty_truth():
    truth = np.zeros((20, 20), dtype=bool)
    truth[5:8, 5:8] = True
    got = distance_weighted_tversky(np.zeros((20, 20), dtype=np.float32), truth)
    assert got.dti == pytest.approx(0.0, abs=1e-6)


def test_ceiling_of_a_dilated_truth_prediction():
    """Widening an emission corridor costs DTI almost linearly - a calibration fact.

    Measured on a 40 px straight trace: a 1 px corridor scores 0.88, a 3 px corridor
    0.70, a 5 px corridor 0.53.  The true positives are fully recovered in every
    case (TP_w == |G|); all the loss is the alpha * FP term.  This is why the
    emission width is a scored decision and not a formatting detail.
    """
    from scipy.ndimage import binary_dilation

    truth = np.zeros((60, 60), dtype=bool)
    truth[20, 10:50] = True
    assert dti_ceiling(truth) > 0.0

    scores = []
    for r in (1, 2, 3):
        yy, xx = np.ogrid[-4:5, -4:5]
        disk = (yy * yy + xx * xx) <= r * r + 1e-12
        got = distance_weighted_tversky(binary_dilation(truth, disk).astype(np.float32), truth)
        assert got.tp == pytest.approx(got.n_truth)   # full recall at every width
        scores.append(got.dti)
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] == pytest.approx(0.8798, abs=1e-3)
    assert scores[1] == pytest.approx(0.7045, abs=1e-3)
    assert scores[2] == pytest.approx(0.5325, abs=1e-3)


def test_credit_map_matches_brute_force_credit():
    rng = np.random.default_rng(11)
    pred = rng.random((25, 25))
    cm = credit_map(pred)
    for (y, x) in zip(*np.nonzero(np.ones((25, 25), dtype=bool))):
        best = 0.0
        for dy, dx, k in kernel_offsets(3.0):
            if 0 <= y + dy < 25 and 0 <= x + dx < 25:
                best = max(best, pred[y + dy, x + dx] * k)
        assert cm[y, x] == pytest.approx(best, abs=1e-12)


def test_fp_weight_map_is_zero_on_truth_and_one_far_away():
    truth = np.zeros((40, 40), dtype=bool)
    truth[20, 20] = True
    w = fp_weight_map(truth)
    assert w[20, 20] == pytest.approx(0.0)
    assert w[20, 21] == pytest.approx(1.0 - 2.0 / 3.0)
    assert w[20, 22] == pytest.approx(1.0 - 1.0 / 3.0)
    assert w[20, 23] == pytest.approx(1.0)   # exactly at the kernel support


def test_restrict_to_limits_scoring_extent_but_not_geometry():
    truth = np.zeros((30, 30), dtype=bool)
    truth[10, 5:25] = True
    pred = truth.astype(np.float32)
    region = np.zeros((30, 30), dtype=bool)
    region[5:15, 5:25] = True
    full = distance_weighted_tversky(pred, truth)
    part = distance_weighted_tversky(pred, truth, restrict_to=region)
    assert 0.0 < part.dti <= full.dti


def test_gt_scorer_matches_the_general_path():
    """The fast repeated-scoring path must equal the general definition exactly."""
    from gems.metric import GtScorer

    rng = np.random.default_rng(21)
    for seed_shape in [(80, 70), (55, 91)]:
        h, w = seed_shape
        truth = rng.random((h, w)) < 0.02
        valid = rng.random((h, w)) < 0.9
        exclude = rng.random((h, w)) < 0.1
        pred = rng.random((h, w)).astype(np.float32)
        pred[~valid] = np.nan
        # region deliberately DIFFERS from valid_mask: the general path zeroes the
        # prediction outside `restrict_to`, so the fast path must do the same or it
        # charges the whole raster as false positives (defect of 2026-09-27).
        region = np.zeros_like(truth)
        region[truth.shape[0] // 4: 3 * truth.shape[0] // 4,
               truth.shape[1] // 4: 3 * truth.shape[1] // 4] = True
        general = distance_weighted_tversky(pred, truth, restrict_to=region,
                                            fp_exclude=exclude)
        fast = GtScorer(truth, valid_mask=valid, fp_exclude=exclude, region=region).score(pred)
        assert fast.tp == pytest.approx(general.tp, abs=1e-9)
        assert fast.fp == pytest.approx(general.fp, abs=1e-9)
        assert fast.fn == pytest.approx(general.fn, abs=1e-9)
        assert fast.dti == pytest.approx(general.dti, abs=1e-12)
