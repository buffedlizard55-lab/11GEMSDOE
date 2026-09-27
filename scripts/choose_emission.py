#!/usr/bin/env python3
"""Choose the emission shape on a leakage-free holdout, then report the paired verdict.

THE QUESTION
------------
The model decides *where* it thinks there is structure.  The shape of the raster
you actually submit decides how much of that survives the metric.  This project
has historically emitted a thin, thresholded, near-binary mask and moved on.  The
metric says that is a scored decision, not a formatting detail:

    DTI(0.2, 0.8) with a 300 m triangular kernel
      corridor width 1 px on a perfectly-placed trace  -> 0.8798
      corridor width 3 px                               -> 0.7045
      corridor width 5 px                               -> 0.5325
      (tests/test_metric.py::test_ceiling_of_a_dilated_truth_prediction)

Every pixel of extra width is charged at alpha = 0.2 and buys positional
tolerance up to 300 m.  Which side of that trade wins depends on how accurately
the model places structure - and that is exactly what a leakage-free spatial
holdout can measure and a leaderboard score cannot.

WHAT IT SWEEPS
--------------
For every arm and fold: probability floor x corridor width (a morphological
dilation of the thresholded mask).  Rasters are the ones written by
``scripts/run_holdout_experiment.py``; nothing is refitted here, so this script
is cheap and re-runnable.

    python scripts/choose_emission.py
    python scripts/choose_emission.py --arms control,structural,inpaint --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.heldout import make_spatial_trace_holdout  # noqa: E402
from gems.metric import GtScorer  # noqa: E402
from gems.raster import sha256_file  # noqa: E402

#: Probability floors and corridor widths, for the shape response curve.
FLOORS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
WIDTHS = (0, 1, 2, 3)

#: AREA-CALIBRATED sweep. A probability floor does NOT transfer from a fold model
#: to the final all-catalogue fit: the fold model is evaluated on geography it never
#: trained on and emits ~9% of it at floor 0.10, while the final fit emits ~54% of
#: the whole grid at the same floor, because in-sample probability is shifted. The
#: quantity that does transfer is the *emitted area fraction*, so the sweep that
#: decides the submission is expressed in area. Measured 2026-09-27.
AREAS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40)


def disk(r: int) -> np.ndarray:
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r + 1e-12


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--arms", default="control,structural,inpaint")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--collar-px", type=int, default=12)
    ap.add_argument("--include-catalogue", action="store_true", default=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    derived = data / "derived"
    with rasterio.open(data / "labels.tif") as ds:
        truth = ds.read(1) > 0
    with rasterio.open(data / "sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))

    holds = [make_spatial_trace_holdout(truth, valid, fold=f, n_folds=args.folds,
                                        block_px=args.block_px, collar_px=args.collar_px)
             for f in range(args.folds)]
    # one scorer per fold; the truth geometry and the distance transform are built once
    scorers = [GtScorer(h.score_truth, valid_mask=valid, fp_exclude=h.fp_exclude,
                        region=h.score_region) for h in holds]

    def area_threshold(p: np.ndarray, region: np.ndarray, frac: float) -> np.ndarray:
        """Emit the top ``frac`` of in-region pixels by probability."""
        vals = p[region]
        if vals.size == 0:
            return np.zeros(p.shape, dtype=bool)
        thr = np.quantile(vals, 1.0 - frac)
        return (p >= thr) & region

    results: dict[str, dict] = {}
    known = holds[0].train_truth | np.zeros_like(truth)   # placeholder, set per fold below
    for arm in [a.strip() for a in args.arms.split(",")]:
        rasters = [derived / f"prob_tracehold_f{f}_{arm}.npy" for f in range(args.folds)]
        if not all(p.exists() for p in rasters):
            log(f"arm '{arm}': fold rasters missing ({rasters[0].name} ...), skipped")
            continue
        probs = [np.load(p).astype(np.float32) for p in rasters]
        table: dict[tuple[float, int], list[float]] = {}
        area: dict[tuple[float, int], list[float]] = {}
        area_tbl: dict[float, list[float]] = {}
        area_frac_tbl: dict[float, list[float]] = {}
        for f_i, (p, sc, h) in enumerate(zip(probs, scorers, holds)):
            known = h.train_truth      # the KNOWN catalogue of this fold, never the held-out truth
            for floor in FLOORS:
                base = (p >= floor)
                for w in WIDTHS:
                    m = base if w == 0 else binary_dilation(base, disk(w))
                    if args.include_catalogue:
                        # Known faults are excluded from the penalty term, so including
                        # them is free, and the page asks for "all faults in the region".
                        #
                        # DEFECT FOUND AND FIXED 2026-09-27: this must be the TRAINING
                        # catalogue, not the full label raster. Emitting `truth` here
                        # emits the held-out faults themselves, which the model has
                        # never seen, and the sweep reports a meaningless DTI of 1.0000
                        # at every floor. In the contest the scorer masks the known
                        # faults and scores only the new ones, so the honest analogue
                        # is: emit the known catalogue (free), never the unknown truth.
                        m = m | known
                    dti = sc.score(m.astype(np.float32)).dti
                    table.setdefault((floor, w), []).append(dti)
                    region_px = int(h.score_region.sum())
                    area.setdefault((floor, w), []).append(
                        float(m[h.score_region].sum() / max(1, region_px)))
            # ---- area-calibrated sweep: the one that decides the submission -----
            for frac in AREAS:
                m = area_threshold(p, h.score_region, frac) | (
                    h.train_truth if args.include_catalogue else False)
                region_px = int(h.score_region.sum())
                dti = sc.score(m.astype(np.float32)).dti
                area_tbl.setdefault(frac, []).append(dti)
                area_frac_tbl.setdefault(frac, []).append(
                    float(m[h.score_region].sum() / max(1, region_px)))
            del p
        probs.clear()
        mean = {k: float(np.mean(v)) for k, v in table.items()}
        best_key = max(mean, key=lambda k: mean[k])
        best_area = max(area_tbl, key=lambda a: float(np.mean(area_tbl[a])))
        results[arm] = {
            "mean_by_shape": {f"floor{floor}_w{w}": v for (floor, w), v in sorted(mean.items())},
            "emitted_area_fraction_by_shape": {
                f"floor{floor}_w{w}": float(np.mean(v)) for (floor, w), v in sorted(area.items())},
            "area_calibrated": {
                f"area{a}": {"mean_dti": float(np.mean(v)), "folds": v,
                             "realised_area_fraction": float(np.mean(area_frac_tbl[a]))}
                for a, v in sorted(area_tbl.items())},
            "folds_by_shape": {f"floor{f}_w{w}": v for (f, w), v in
                               {k: v for k, v in sorted(table.items())}.items()},
            "best_shape": {"floor": best_key[0], "width_px": best_key[1],
                           "mean_dti": mean[best_key], "fold_dti": table[best_key],
                           "emitted_area_fraction_of_valid_grid":
                               float(np.mean(area[best_key]))},
            "std_at_best": float(np.std(table[best_key])),
        }
        log(f"arm '{arm}': best PROBABILITY floor={best_key[0]} w={best_key[1]} -> "
            f"{mean[best_key]:.4f} +/- {results[arm]['std_at_best']:.4f}")
        log(f"arm '{arm}': best AREA target={best_area:.2f} -> "
            f"{float(np.mean(area_tbl[best_area])):.4f}  "
            f"(realised {100 * float(np.mean(area_frac_tbl[best_area])):.2f}% of the scored region)")

    # ---- paired comparison on identical folds -----------------------------------
    verdict = {}
    arms = list(results)
    for arm in arms:
        base = results[arms[0]]["folds_by_shape"]
        cur = results[arm]["folds_by_shape"]
        common = sorted(set(base) & set(cur))
        diffs = [float(np.mean(cur[k])) - float(np.mean(base[k])) for k in common]
        verdict[arm] = {
            "reference_arm": arms[0],
            "n_shapes_compared": len(common),
            "mean_delta_over_all_shapes": float(np.mean(diffs)),
            "max_delta_over_all_shapes": float(np.max(diffs)),
            "delta_at_best_shape": results[arm]["best_shape"]["mean_dti"]
            - results[arms[0]]["best_shape"]["mean_dti"],
        }
        log(f"{arm:12s} vs {arms[0]}: mean delta over {len(common)} shapes "
            f"{verdict[arm]['mean_delta_over_all_shapes']:+.5f}, "
            f"at best shape {verdict[arm]['delta_at_best_shape']:+.5f}")

    out = {
        "metric": {"alpha": 0.2, "beta": 0.8, "r_pixels": 3.0,
                   "source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/"},
        "official_mask_source": "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
        "features_sha256": sha256_file(data / "training_features.tif"),
        "labels_sha256": sha256_file(data / "labels.tif"),
        "split": {
            "folds": args.folds, "block_px": args.block_px, "collar_px": args.collar_px,
            "scored_fault_px_per_fold": [h.score_px for h in holds],
        },
        "include_catalogue": args.include_catalogue,
        "floors": list(FLOORS), "widths": list(WIDTHS), "area_targets": list(AREAS),
        "results": results, "paired_verdict": verdict,
        "caveats": [
            "Four folds give very little statistical power. A difference smaller than the "
            "fold-to-fold standard deviation is not evidence.",
            "The held-out faults are mapped catalogue faults the model never trained on, not "
            "expert-identified new faults. See docs/HYPOTHESES.md for why the two are close "
            "but not identical.",
            "Width is measured on a binary mask. The metric's kernel forgives a 300 m "
            "misplacement, which this sweep models by dilation, not by systematic offset; a "
            "systematic offset error would need its own sweep.",
        ],
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        log(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
