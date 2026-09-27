#!/usr/bin/env python3
"""Paired hypothesis experiment: same data, same folds, same model, one change at a time.

WHAT THIS IS FOR
----------------
docs/HYPOTHESES.md ranks candidate ideas before any of them is implemented.  This
script is the gate: an arm only reaches a submission slot if it beats its paired
control on the *same* split, measured with the *official* metric, with the
official scoring mask applied, and with the trace-holdout that simulates faults
the model has never seen.

THE TWO EVALUATIONS, AND WHY BOTH
---------------------------------
1. ``trace``  - hold out whole fault *traces* (default 10%), score only those, and
   exclude the remaining catalogue from the false-positive term exactly as DrivenData
   staff described.  This is the closest available simulation of the contest.
2. ``block``  - 4 spatially blocked folds with a 3 px training collar, the
   protocol the project registered in 2026-09-27 for H1.  Weaker as a proxy for
   the real task, kept so results stay comparable with the recorded H1 number.

ARMS
----
  control    the 19 official bands + the 9-band topographic and 7-band radiometric
             auxiliary stacks, quantile-encoded.  Same feature budget as the
             recorded H1 control, so H1 remains the comparison point.
  structural control + the H2/H3 derived channels (fabric coherence, basement
             contact step, conductivity coherence).
  inpaint    control + the H2/H3 channels AND the trace-inpainting objective, which
             stops the model being taught "no fault here" at the unmapped
             continuation of a mapped fault.

USAGE
  python scripts/run_holdout_experiment.py --arms control,structural,inpaint
  python scripts/run_holdout_experiment.py --quick      # smoke test, not a result
  python scripts/run_holdout_experiment.py --json docs/evidence/<run>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.blocks import make_fold_plan  # noqa: E402
from gems.features import FeatureSpec, build_feature_cube  # noqa: E402
from gems.heldout import leakage_probe, make_spatial_trace_holdout  # noqa: E402
from gems.metric import distance_weighted_tversky  # noqa: E402
from gems.model import DetectorConfig, fit_detector, predict_grid, sample_training_pixels  # noqa: E402
from gems.raster import sha256_file  # noqa: E402

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs(data: Path):
    with rasterio.open(data / "training_features.tif") as ds:
        height, width = ds.height, ds.width
    with rasterio.open(data / "labels.tif") as ds:
        truth = ds.read(1) > 0
    with rasterio.open(data / "sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))
    return truth, valid, height, width


def best_threshold(components_by_threshold: dict[float, float]) -> tuple[float, float]:
    t = max(components_by_threshold.items(), key=lambda kv: kv[1])
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--arms", default="control,structural,inpaint")
    ap.add_argument("--holdout-fraction", type=float, default=0.10,
                    help="kept for the record; the spatial split uses --block-folds/--block-px")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-folds", type=int, default=4)
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--collar-px", type=int, default=3)
    ap.add_argument("--negatives", type=int, default=400_000)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--inpaint-seed-offset", type=int, default=1000)
    ap.add_argument("--inpaint-fraction", type=float, default=0.30)
    ap.add_argument("--inpaint-repeats", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="smoke test only; not a result")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.quick:
        args.holdout_fraction = 0.10
        args.negatives = 60_000
        args.max_iter = 40
        args.block_folds = 2
        args.inpaint_repeats = 1

    data = Path(args.data)
    truth, valid, height, width = load_inputs(data)
    log(f"grid {width}x{height}; catalogue {int(truth.sum()):,} fault px; "
        f"valid footprint {int(valid.sum()):,} px")

    aux = [p for p in (data / "external" / "topo_u8.tif",
                       data / "external" / "radiometric_u8.tif") if p.exists()]
    if not aux:
        log("auxiliary stacks missing - run: bash scripts/download_competition_data.sh --with-aux")

    derived = data / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    cubes: dict[str, Path] = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        spec = FeatureSpec(use_aux=bool(aux), structural=arm in ("structural", "inpaint"))
        cube = derived / f"features_{arm}.npy"
        prov = cube.with_suffix(".json")
        if args.skip_build and prov.exists():
            log(f"reusing {cube.name} (--skip-build)")
        else:
            log(f"building feature cube for arm '{arm}' ({len(aux)} aux stacks, "
                f"structural={spec.structural})")
            build_feature_cube(data / "training_features.tif", aux, spec, cube, log=log)
        cubes[arm] = cube

    from gems.features import channel_names
    ch_names = {arm: channel_names(data / "training_features.tif", aux,
                                   FeatureSpec(use_aux=bool(aux),
                                               structural=arm in ("structural", "inpaint")))
                for arm in cubes}

    results: dict[str, dict] = {}
    usable_all = valid.copy()

    # ---- leakage gate: a catalogue-reproducing model must score near zero --------
    leak = leakage_probe(truth, valid, block_px=args.block_px, collar_px=args.collar_px)
    log(f"leakage probe: a model that only reproduces the training catalogue scores "
        f"{leak['mean_dti']:.4f} on held-out faults (must be small to be meaningful)")
    if leak["mean_dti"] > 0.25:
        log("ABORT: the holdout leaks. No model score on it would mean anything.")
        return 3

    # ====================== spatially separated trace holdout ====================
    hold_folds = [make_spatial_trace_holdout(truth, valid, fold=f,
                                             n_folds=args.block_folds,
                                             block_px=args.block_px,
                                             collar_px=args.collar_px)
                  for f in range(args.block_folds)]
    log("spatial trace holdout: " + "; ".join(
        f"fold{f.fold} {f.n_score_traces} traces/{f.score_px:,}px" for f in hold_folds))
    log("official mask applied (known catalogue pixels excluded from the FP term)")

    for arm, cube_path in cubes.items():
        t0 = time.time()
        per_fold = []
        h1: list[float] = []
        sweep_agg: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
        for f in hold_folds:
            cube = np.load(cube_path, mmap_mode="r")
            cfg = DetectorConfig(n_negatives=args.negatives, max_iter=args.max_iter,
                                 inpaint_fraction=(args.inpaint_fraction if arm == "inpaint"
                                                   else 0.0),
                                 inpaint_repeats=args.inpaint_repeats)
            X, y = sample_training_pixels(cube, f.train_truth, valid, cfg, seed=args.seed + f.fold)
            clf = fit_detector(X, y, cfg)
            prob = predict_grid(clf, cube, valid)
            del X, y, clf, cube
            c = f.score(prob)
            per_fold.append(c)
            # The protocol registered for H1 on 2026-09-27, kept so this run is
            # directly comparable with the recorded control 0.149986 / H1 0.149795:
            # truth = held-out catalogue, predictions scored inside the held-out
            # blocks, and the false-positive weight taken against the held-out
            # truth over the WHOLE grid (so re-drawing the training catalogue is
            # charged).  The official mask is NOT applied there - it was not
            # published to this project when that number was recorded.
            h1.append(distance_weighted_tversky(prob, f.score_truth,
                                                restrict_to=f.score_region).dti)
            for t in THRESHOLDS:
                sweep_agg[t].append(f.score(np.where(prob >= t, 1.0, 0.0)).dti)
            np.save(derived / f"prob_tracehold_f{f.fold}_{arm}.npy", prob.astype(np.float32))
            del prob
        sweep = {t: {"dti": float(np.mean(v)), "folds": v} for t, v in sweep_agg.items()}
        best_thr, best = best_threshold({k: v["dti"] for k, v in sweep.items()})
        dti_prob = float(np.mean([c.dti for c in per_fold]))
        results[arm] = {
            "trace_holdout": {
                "dti_at_probability": dti_prob,
                "dti_at_probability_folds": [c.dti for c in per_fold],
                "std": float(np.std([c.dti for c in per_fold])),
                "components_at_probability_folds": [c.as_dict() for c in per_fold],
                "sweep": sweep,
                "best_threshold": best_thr,
                "best_threshold_dti": best,
                "dti_at_preregistered_0_20": sweep[0.20]["dti"],
                "h1_registered_block_dti": {
                    "folds": h1,
                    "mean": float(np.mean(h1)),
                    "note": "same protocol as the 2026-09-27 H1 record; no official mask",
                },
            },
            "config": cfg.as_dict(),
            "n_channels": len(ch_names[arm]),
            "fit_seconds": round(time.time() - t0, 1),
        }
        log(f"arm '{arm}': TH-DTI(prob) {dti_prob:.4f} +/- {results[arm]['trace_holdout']['std']:.4f}; "
            f"best {best:.4f} @thr {best_thr}; H1-protocol block DTI "
            f"{results[arm]['trace_holdout']['h1_registered_block_dti']['mean']:.4f} "
            f"({results[arm]['fit_seconds']}s)")

    # =========================== verdict ==========================================
    verdict = {}
    base = results[args.arms.split(",")[0].strip()]["trace_holdout"]["dti_at_probability"]
    for arm, r in results.items():
        d = r["trace_holdout"]["dti_at_probability"] - base
        base_h1 = results[args.arms.split(",")[0].strip()]["trace_holdout"][
            "h1_registered_block_dti"]["mean"]
        verdict[arm] = {
            "delta_th_dti_vs_first_arm": d,
            "delta_h1_block_dti_vs_first_arm":
                r["trace_holdout"]["h1_registered_block_dti"]["mean"] - base_h1,
            "h1_reference_2026_09_27": {
                "control": 0.149986, "h1_strain_structure_tensor": 0.149795,
                "source": "docs/evidence/h1_holdout_2026-09-27.json",
            },
        }

    out = {
        "run": {
            "data_dir": str(data),
            "features_sha256": sha256_file(data / "training_features.tif"),
            "labels_sha256": sha256_file(data / "labels.tif"),
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "block_folds": args.block_folds,
            "block_px": args.block_px,
            "collar_px": args.collar_px,
            "negatives": args.negatives,
            "max_iter": args.max_iter,
            "inpaint_fraction": args.inpaint_fraction,
            "inpaint_repeats": args.inpaint_repeats,
            "quick": args.quick,
            "arms": [a.strip() for a in args.arms.split(",")],
            "channels": ch_names,
        },
        "metric": {
            "alpha": 0.2, "beta": 0.8, "r_pixels": 3.0,
            "source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
            "official_mask_source": "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
        },
        "leakage_probe": leak,
        "trace_holdout_splits": [f.describe() for f in hold_folds],
        "results": results,
        "paired_verdict": verdict,
        "limitations": [
            "The trace holdout removes whole mapped traces; the contest's private faults "
            "were identified by experts as faults missing from the catalogue, which may be "
            "systematically more subtle than an average mapped trace.",
            "Held-out traces are scored with the official mask applied, but the exact mask "
            "geometry used by the platform (catalogue pixels only, or a buffered corridor) "
            "is not published; only the statement that known faults are excluded is confirmed.",
            "A win here is necessary, not sufficient, for a submission slot. The leaderboard "
            "is the only authority on the real target and it cannot be predicted from here.",
        ],
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        log(f"wrote {args.json}")

    log("paired verdict (TH-DTI, probability raster):")
    for arm, v in verdict.items():
        log(f"  {arm:12s} TH-DTI delta {v['delta_th_dti_vs_first_arm']:+.5f}   "
            f"H1-protocol delta {v['delta_h1_block_dti_vs_first_arm']:+.5f}")
    if args.quick:
        log("QUICK RUN - numbers above are a plumbing check, not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
