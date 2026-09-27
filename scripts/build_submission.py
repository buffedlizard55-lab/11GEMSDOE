#!/usr/bin/env python3
"""Build the competition submission GeoTIFF from a fitted detector, and prove it is valid.

Pipeline:  held-out experiment -> final fit on the full catalogue -> shaped
probability raster -> GeoTIFF -> re-read from disk -> manifest with hashes and a
unique submission name and note.

    python scripts/build_submission.py --arm inpaint
    python scripts/build_submission.py --arm inpaint --floor 0.05
    python scripts/build_submission.py --arm inpaint --publish docs/downloads

WHY THE FLOOR IS A SCORED PARAMETER AND NOT A TRIM
--------------------------------------------------
The contest metric is DTI(0.2, 0.8) with a 300 m triangular kernel, so emitting
mass away from any plausible structure is charged at only 0.2 while a missed
structure is charged at 0.8 - but mass is charged in full.  ``tests/test_metric.py``
measures the cost of a corridor placed exactly on a known trace: 1 px wide scores
0.88, 3 px 0.70, 5 px 0.53.  Emission width and floor are therefore chosen from
the holdout threshold sweep recorded in the experiment JSON, and the chosen values
plus their measured scores are written into the manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.features import FeatureSpec, channel_names  # noqa: E402
from gems.heldout import make_spatial_trace_holdout  # noqa: E402
from gems.model import DetectorConfig, fit_detector, predict_grid, sample_training_pixels  # noqa: E402
from gems.raster import sha256_file  # noqa: E402
from gems.submission import emit_submission, shape_for_submission, submission_manifest, write_manifest  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--arm", default="inpaint")
    ap.add_argument("--area-fraction", type=float, default=0.06,
                    help="fraction of in-footprint pixels emitted; the leakage-free "
                         "spatial holdout optimum is 0.06 for every arm tested")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="optional extra absolute probability cut, applied after the area cut")
    ap.add_argument("--include-catalogue", action="store_true", default=True)
    ap.add_argument("--no-include-catalogue", dest="include_catalogue", action="store_false")
    ap.add_argument("--negatives", type=int, default=250_000,
                    help="MUST match the validated experiment; a different budget "
                         "changes the probability calibration and invalidates the "
                         "holdout-chosen floor")
    ap.add_argument("--max-iter", type=int, default=150,
                    help="MUST match the validated experiment, for the same reason")
    ap.add_argument("--sampling-seed", type=int, default=0,
                    help="negative/positive sampling seed; the experiment used 0")
    ap.add_argument("--inpaint-fraction", type=float, default=0.30)
    ap.add_argument("--inpaint-repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--collar-px", type=int, default=12)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--name", default=None, help="unique submission name, no extension")
    ap.add_argument("--note", default=None, help="the DrivenData 'Note (optional)' text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--publish", default=None, help="directory to copy the validated file into")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    feat = data / "training_features.tif"
    cube = data / "derived" / f"features_{args.arm}.npy"
    if not cube.exists():
        log(f"feature cube missing: {cube}. Run scripts/run_holdout_experiment.py first.")
        return 2

    aux = [p for p in (data / "external" / "topo_u8.tif",
                       data / "external" / "radiometric_u8.tif") if p.exists()]
    with rasterio.open(data / "labels.tif") as ds:
        truth = ds.read(1) > 0
    with rasterio.open(data / "sample_submission.tif") as ds:
        valid = np.isfinite(ds.read(1))

    ch_names = channel_names(feat, aux, FeatureSpec(
        use_aux=bool(aux), structural=args.arm in ("structural", "inpaint")))

    cfg = DetectorConfig(n_negatives=args.negatives, max_iter=args.max_iter,
                         inpaint_fraction=(args.inpaint_fraction if args.arm == "inpaint"
                                           else 0.0),
                         inpaint_repeats=args.inpaint_repeats,
                         random_state=42)
    cube_mm = np.load(cube, mmap_mode="r")
    log(f"final fit: arm '{args.arm}', {len(ch_names)} channels, "
        f"inpaint_fraction={cfg.inpaint_fraction} x{cfg.inpaint_repeats}")
    X, y = sample_training_pixels(cube_mm, truth, valid, cfg, seed=args.sampling_seed)
    clf = fit_detector(X, y, cfg)
    prob = predict_grid(clf, cube_mm, valid)
    del X, y, clf, cube_mm
    log("predicted full grid")

    # ---- honest, leakage-free estimate of this configuration on unseen faults ----
    holds = [make_spatial_trace_holdout(truth, valid, fold=f, n_folds=args.folds,
                                        block_px=args.block_px, collar_px=args.collar_px)
             for f in range(args.folds)]
    per_fold = []
    for h in holds:
        vals = prob[h.score_region]
        thr = float(np.quantile(vals, 1.0 - args.area_fraction))
        m = (prob >= thr) & h.score_region
        if args.include_catalogue:
            m = m | h.train_truth
        per_fold.append(h.score(m.astype(np.float32)).dti)
    log(f"NOTE: the spatial holdout scores THIS raster, which was fitted on all "
        f"catalogue traces including the held-out ones, so {np.mean(per_fold):.4f} "
        f"is optimistic by construction. The honest estimate is the fitted-arm "
        f"number in the experiment JSON, not this one.")

    shaped = shape_for_submission(prob, area_fraction=args.area_fraction, floor=args.floor,
                                  include_catalogue=args.include_catalogue,
                                  catalogue=truth if args.include_catalogue else None,
                                  valid=valid)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"gems-{args.arm}-fabric-inpaint-v1"
    out = Path(args.out) if args.out else ROOT / "data" / f"{name}.tif"
    short = sha256_file(out)[:12] if out.exists() else "pending"
    report = emit_submission(
        shaped,
        template_path=data / "sample_submission.tif",
        out_path=out,
        band_description="predicted fault presence probability (0-1)",
        tags={"submission_name": name, "generated_utc": stamp, "arm": args.arm,
              "pipeline": "11GEMSDOE gems.features+model+submission"},
    )
    log(f"wrote {out} ({report['profile']['dtype']}, "
        f"range [{report['checks']['inside_min']}, {report['checks']['inside_max']}])")

    arm_blurb = {
        "control": ("control arm: the 19 official GeoDAWN bands plus 9 topographic and "
                    "7 airborne-radiometric auxiliary channels, HistGradientBoosting "
                    "pixel detector"),
        "structural": ("control plus H2/H3 structural channels: structure-tensor "
                       "coherence of the magnetic/gravity gradients, and the "
                       "basement-surface / conductivity contact step"),
        "inpaint": ("control plus H2/H3 channels and the H4 trace-inpainting objective, "
                    "which stops the model being taught 'no fault' at the unmapped "
                    "continuation of a mapped trace"),
    }[args.arm]
    note = args.note or (
        f"11GEMSDOE {args.arm}: {arm_blurb}. Emitted as a binary mask at the "
        f"leakage-free spatial holdout optimum: top {args.area_fraction:.0%} of "
        f"in-footprint pixels by probability, corridor width 0 px, supplied "
        f"USGS/INGENIOUS catalogue included at 1.0 because known faults are "
        f"excluded from scoring.")
    manifest = submission_manifest(out, name, note, provenance={
        "generated_utc": stamp,
        "arm": args.arm,
        "pipeline": "scripts/build_submission.py",
        "channels": ch_names,
        "detector_config": cfg.as_dict(),
        "features_sha256": sha256_file(feat),
        "labels_sha256": sha256_file(data / "labels.tif"),
        "template_sha256": sha256_file(data / "sample_submission.tif"),
        "shaping": {"area_fraction": args.area_fraction, "floor": args.floor,
                    "include_catalogue": args.include_catalogue,
                    "chosen_on": "docs/evidence/emission_choice_2026-09-27.json (4-fold leakage-free spatial holdout, official metric and mask)"},
        "metric": {
            "name": "distance-weighted Tversky (DW-Tversky)",
            "alpha": 0.2, "beta": 0.8, "r_m": 300, "r_px_at_100m": 3,
            "source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        },
        "known_fault_masking_rule": {
            "quote": "Pixels corresponding to known USGS/INGENIOUS faults are masked / "
                     "excluded from evaluation, so they do not count towards penalty terms.",
            "who": "chrisk-dd, DrivenData Staff, 2026-09-16",
            "url": "https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516",
        },
        "caveat_spatial_holdout_on_this_raster": float(np.mean(per_fold)),
    })
    write_manifest(manifest, out.with_suffix(".manifest.json"))
    if args.publish:
        dest = Path(args.publish)
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(out, dest / out.name)
        log(f"published to {dest / out.name}")

    print(json.dumps({"file": str(out), "note": note, "sha256": manifest["sha256"],
                      "bytes": manifest["bytes"],
                      "value_stats": manifest["value_stats"]}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
