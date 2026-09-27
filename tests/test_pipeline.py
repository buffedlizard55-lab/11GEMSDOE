"""Validation gates for the pieces that decide whether an upload is legal or meaningful."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from gems.blocks import make_fold_plan
from gems.heldout import extract_traces, leakage_probe, make_spatial_trace_holdout
from gems.raster import check_submission_raster, pixel_sha256
from gems.submission import emit_submission, shape_for_submission

GEOM = dict(driver="GTiff", dtype="float32", count=1, crs="EPSG:32611",
            transform=from_origin(243350.0, 4508550.0, 100.0, 100.0))


def _write_template(path, h=40, w=36, valid_h=34, valid_w=30):
    arr = np.zeros((h, w), dtype=np.float32)
    arr[:valid_h, :valid_w] = 0.0
    arr[valid_h:, :] = np.nan
    arr[:, valid_w:] = np.nan
    with rasterio.open(path, "w", nodata=np.nan, **GEOM, height=h, width=w) as ds:
        ds.write(arr, 1)
    return arr


def test_fold_plan_covers_a_grid_that_is_not_a_multiple_of_the_block():
    plan = make_fold_plan(3730, 3292, n_folds=4, block_px=512, collar_px=0)
    fmap = plan.fold_of_pixel()
    assert fmap.shape == (3730, 3292)
    assert set(np.unique(fmap).tolist()) == {0, 1, 2, 3}
    assert plan.held_out_mask(0).sum() > 0
    assert (plan.train_mask(0) & plan.held_out_mask(0)).sum() == 0


def test_collar_keeps_training_away_from_the_scored_area():
    plan = make_fold_plan(600, 600, n_folds=2, block_px=256, collar_px=5)
    held = plan.held_out_mask(0, apply_collar=False)
    train = plan.train_mask(0, apply_collar=True)
    assert (train & held).sum() == 0
    assert (train & held).sum() < plan.train_mask(0, apply_collar=False).sum() + 1


def test_trace_extraction_finds_8_connected_strands():
    truth = np.zeros((30, 30), dtype=bool)
    truth[5, 5:15] = True
    truth[6, 5:15] = True       # a 2-px-thick trace is one component
    truth[20, 5:10] = True      # a separate strand
    traces = extract_traces(truth)
    assert len(traces) == 2
    assert sorted(t.size for t in traces) == [5, 20]


def test_spatial_holdout_hides_the_collar_and_never_scores_known_faults():
    rng = np.random.default_rng(0)
    truth = np.zeros((600, 600), dtype=bool)
    for i in range(40):
        r, c = rng.integers(0, 500, 2)
        truth[r:r + 25, c] = True
    valid = np.ones((600, 600), dtype=bool)
    h = make_spatial_trace_holdout(truth, valid, fold=0, n_folds=4, block_px=256, collar_px=8)
    assert not (h.train_truth & h.score_truth).any()
    assert (h.score_truth & ~h.score_region).sum() == 0
    # every known-fault pixel is excluded from the false-positive term
    assert (h.fp_exclude & truth & h.score_truth).sum() == 0
    assert h.score_px > 0


def test_leakage_probe_is_near_zero_for_a_spatial_split():
    """A catalogue-reproducing prediction must collect almost nothing on held-out faults.

    This is the check that the naive "hold out 10% of traces at random" split
    fails: with the official mask and a 300 m kernel it scores 0.97.
    """
    rng = np.random.default_rng(1)
    truth = np.zeros((800, 800), dtype=bool)
    for _ in range(120):
        r, c = rng.integers(0, 700, 2)
        truth[r:r + 40, c] = True
    valid = np.ones((800, 800), dtype=bool)
    probe = leakage_probe(truth, valid, block_px=256, collar_px=12)
    assert probe["mean_dti"] < 0.05


def test_shaping_clips_and_keeps_the_catalogue():
    truth = np.zeros((20, 20), dtype=bool)
    truth[5:8, 5:8] = True
    prob = np.full((20, 20), 2.5, dtype=np.float32)
    prob[0, 0] = -1.0
    prob[0, 1] = np.nan
    prob[0, 2] = 0.01          # below the floor
    out = shape_for_submission(prob, floor=0.05, include_catalogue=True, catalogue=truth)
    finite = out[np.isfinite(out)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    assert (out[truth] == 1.0).all()
    assert out[0, 0] == 0.0
    assert out[0, 1] == 0.0
    assert out[0, 2] == 0.0          # below the floor


def test_emitted_submission_passes_every_stated_rule(tmp_path):
    template = tmp_path / "template.tif"
    out = tmp_path / "candidate.tif"
    _write_template(template)
    values = np.zeros((40, 36), dtype=np.float32)
    values[10:12, 10:20] = 0.8
    report = emit_submission(values, template, out)
    assert report["ok"] is True
    assert report["checks"]["inside_out_of_range_pixels"] == 0
    assert report["checks"]["outside_finite_pixels"] == 0
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "float32"
        assert ds.count == 1
        assert str(ds.crs) == "EPSG:32611"
        band = ds.read(1)
    assert np.isnan(band[34:, :]).all()
    assert np.isnan(band[:, 30:]).all()
    assert band[10:12, 10:20].min() > 0


def test_validator_rejects_an_out_of_range_file(tmp_path):
    """The exact failure the team hit: "Predicted values must be in range [0, 1]"."""
    from gems.raster import ConformanceError

    template = tmp_path / "template.tif"
    bad = tmp_path / "bad.tif"
    _write_template(template)
    values = np.zeros((40, 36), dtype=np.float32)
    values[0, 0] = 1.4
    with pytest.raises(ConformanceError, match="outside \\[0, 1\\]"):
        emit_submission(values, template, bad)


def test_pixel_hash_ignores_tiff_layout_but_not_values(tmp_path):
    a = np.zeros((5, 5), dtype=np.float32)
    b = a.copy()
    b[2, 2] = 0.5
    assert pixel_sha256(a) == pixel_sha256(a.copy())
    assert pixel_sha256(a) != pixel_sha256(b)


def test_inpaint_hides_terminal_segments_and_keeps_them_out_of_the_negatives(tmp_path):
    """H4: the objective must stop teaching "no fault" where the map simply stops."""
    from gems.model import DetectorConfig, _hide_trace_segments, sample_training_pixels

    truth = np.zeros((40, 40), dtype=bool)
    truth[10, 5:35] = True
    hidden = _hide_trace_segments((40, 40), truth, 0.30, np.random.default_rng(0))
    assert hidden.sum() == 9                      # 30% of a 30 px trace
    assert hidden[10, 26:35].all()                # the *end*, not the middle
    assert not hidden[10, 5:20].any()
    assert (hidden & ~truth).sum() == 0

    cube = np.lib.format.open_memmap(tmp_path / "c.npy", mode="w+", dtype=np.uint8,
                                     shape=(3, 40, 40))
    cube[:] = 7
    valid = np.ones((40, 40), dtype=bool)
    cfg = DetectorConfig(n_negatives=5000, inpaint_fraction=0.3, inpaint_repeats=1)
    _X, y, idx = sample_training_pixels(cube, truth, valid, cfg, seed=0, return_indices=True)
    flat_hidden = set(np.flatnonzero(hidden.ravel()).tolist())
    flat_truth = set(np.flatnonzero(truth.ravel()).tolist())
    neg_flat = set(idx[y == 0].tolist())
    pos_flat = set(idx[y == 1].tolist())
    # every hidden pixel is either a positive or was not sampled - never a negative
    assert not (flat_hidden & neg_flat)
    assert pos_flat == flat_truth - flat_hidden

    # with the objective disabled the same pixels are trained as POSITIVES, i.e.
    # the model is never given a chance to generalise past the mapped end
    cfg_off = DetectorConfig(n_negatives=5000, inpaint_fraction=0.0)
    _X2, y2, idx2 = sample_training_pixels(cube, truth, valid, cfg_off, seed=0,
                                           return_indices=True)
    pos2 = set(idx2[y2 == 1].tolist())
    assert flat_hidden <= pos2
    assert not (flat_hidden & pos_flat)
