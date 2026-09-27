#!/usr/bin/env python3
"""Score historical team rasters against the official metric, and diagnose 0.1563.

WHY THIS SCRIPT EXISTS
----------------------
Our sites kept reporting 0.1563.  The file audit (docs/evidence/
submission_identity_audit_2026-09-27.json) proved GEMSDOE1 and 5GEMSDOE published
the *same* TIFF, so one number is explained.  It did not explain the rest: 8GEMSDOE
also shows 0.1563 with a different raster, and the public leaderboard shows
unrelated participants at 0.1563 too.  Repeating a number is not the same as
repeating a file.

This script answers the question that *can* be answered without an upload receipt,
by measuring what each published prediction actually says, using the official
distance-weighted Tversky index (src/gems/metric.py, verified against the worked
example on the official problem page):

  dti_catalogue   DTI against the full supplied catalogue.  Measures how well a
                  file reproduces faults the team already knew about.
  th_dti          "trace-holdout DTI" - DTI against a seeded 10% of catalogue
                  TRACES that are treated as unknown, with false positives
                  weighted against those traces only.  This is a simulation of
                  the contest, which scores faults that are absent from the
                  supplied labels.
  off_catalogue   fraction of the predicted mass that lies further than the
                  metric's 300 m support from any catalogued fault - i.e. the
                  part of the file that can only ever pay off on genuinely new
                  structure.
  area_frac       fraction of the in-footprint grid with p > 0, and the p budget.

Nothing here touches the contest, uploads nothing, and needs no credentials.

USAGE
  python scripts/diagnose_scores.py                       # built-in candidate list
  python scripts/diagnose_scores.py --raster a.tif --raster b.tif
  python scripts/diagnose_scores.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.heldout import leakage_probe, make_spatial_trace_holdout  # noqa: E402
from gems.metric import distance_weighted_tversky  # noqa: E402
from gems.raster import pixel_sha256, sha256_file  # noqa: E402

#: Rasters published by the team's sibling projects, with the score our team
#: reported for each.  Score values are USER-REPORTED, not receipts.
CANDIDATES: list[tuple[str, str, float | None]] = [
    ("GEMSDOE1/5GEMSDOE  'ens12-adopted'", "data/historical/gemsdoe-ens12-adopted-7f00890a.tif", 0.1563),
    ("GEMSDOE3           'pindrop nodes'", "data/historical/pindrop-v4-nodes-f347b70daa.tif", 0.1193),
    ("GEMSDOE3           'pindrop catalogue-gap'", "data/historical/pindrop-v4-discovery-37f9d5b855.tif", 0.0830),
    ("GEMSDOE3           'pindrop dense ridge'", "data/historical/pindrop-v4-ridge-4e03fc9705.tif", 0.1152),
    ("GEMSDOE2           'dual-family union'", "data/historical/gemsdoe2-dual-family-union-f68e590f.tif", 0.1560),
    ("GEMSDOE2           'extension arm'", "data/historical/gemsdoe2-extension-arm-ad5ba911.tif", None),
    ("GEMSDOE2           'precision arm'", "data/historical/gemsdoe2-precision-arm-8bce5dfe.tif", None),
    ("official sample submission (all-zero)", "data/sample_submission.tif", None),
    ("the supplied labels themselves", "data/labels.tif", None),
]


def load(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        crs, tr, w, h = str(ds.crs), tuple(ds.transform)[:6], ds.width, ds.height
    return arr, {"crs": crs, "transform": list(tr), "width": w, "height": h}


def describe(arr: np.ndarray, tpl_valid: np.ndarray, truth: np.ndarray,
             dist_to_truth: np.ndarray) -> dict:
    inside = tpl_valid
    p = np.where(inside, np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0)
    np.clip(p, 0.0, 1.0, out=p)
    mass = float(p.sum())
    off = inside & (dist_to_truth > 3.0)
    return {
        "area_frac_p_gt_0": float((p[inside] > 0).mean()),
        "area_frac_p_ge_0_2": float((p[inside] >= 0.2).mean()),
        "total_p_mass": mass,
        "mean_p": float(p[inside].mean()),
        "p_max": float(p[inside].max()) if inside.any() else None,
        "off_catalogue_mass_frac": float(p[off].sum() / mass) if mass > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--raster", action="append", default=[],
                    help="extra raster to diagnose (repeatable)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--collar-px", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    labels_path = data / "labels.tif"
    tpl_path = data / "sample_submission.tif"
    if not labels_path.exists() or not tpl_path.exists():
        print("data/labels.tif and data/sample_submission.tif are required; run "
              "scripts/download_competition_data.sh first", file=sys.stderr)
        return 2

    with rasterio.open(labels_path) as ds:
        truth = ds.read(1) > 0
    with rasterio.open(tpl_path) as ds:
        tpl = ds.read(1)
        tpl_valid = np.isfinite(tpl)
        if not tpl_valid.any():
            tpl_valid = np.ones(truth.shape, dtype=bool)
        tpl_grid = {"crs": str(ds.crs), "transform": list(ds.transform)[:6],
                    "width": ds.width, "height": ds.height}

    dist_to_truth = distance_transform_edt(~truth)
    leak = leakage_probe(truth, tpl_valid, block_px=args.block_px, collar_px=args.collar_px)
    if not args.quiet:
        print(f"leakage probe: a prediction that only reproduces the training catalogue "
              f"scores {leak['mean_dti']:.4f} on the held-out faults\n")
    holds = [make_spatial_trace_holdout(truth, tpl_valid, fold=f, n_folds=args.folds,
                                        block_px=args.block_px, collar_px=args.collar_px)
             for f in range(args.folds)]

    rows: list[dict] = []
    todo = list(CANDIDATES) + [(Path(r).name, r, None) for r in args.raster]
    for name, rel, reported in todo:
        path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        if not path.exists():
            if not args.quiet:
                print(f"  (skip, not present: {rel})")
            continue
        arr, grid = load(path)
        if (grid["width"], grid["height"]) != (tpl_grid["width"], tpl_grid["height"]):
            print(f"  (skip, grid mismatch: {rel})", file=sys.stderr)
            continue
        cat = distance_weighted_tversky(arr, truth)
        th_dti = float(np.mean([h.score(arr).dti for h in holds]))
        th_sd = float(np.std([h.score(arr).dti for h in holds]))
        row = {
            "name": name,
            "path": rel,
            "sha256": sha256_file(path),
            "pixel_sha256": pixel_sha256(np.where(tpl_valid, np.nan_to_num(arr, nan=0.0), 0.0)
                                         .astype("float32")),
            "reported_score": reported,
            "grid_matches_official": grid["crs"] == tpl_grid["crs"]
            and grid["transform"] == tpl_grid["transform"],
            "dti_catalogue": cat.dti,
            "catalogue_components": cat.as_dict(),
            "th_dti": th_dti,
            "th_dti_std": th_sd,
            "th_components_by_fold": [h.score(arr).as_dict() for h in holds],
            **describe(arr, tpl_valid, truth, dist_to_truth),
        }
        rows.append(row)

    out = {
        "metric": {
            "alpha": 0.2, "beta": 0.8, "r_pixels": 3.0,
            "source": "https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/",
        },
        "leakage_probe": leak,
        "trace_holdout": [h.describe() for h in holds],
        "catalogue": {
            "fault_pixels": int(truth.sum()),
            "grid_pixels": int(truth.size),
            "in_footprint_pixels": int(tpl_valid.sum()),
        },
        "rows": rows,
        "caveats": [
            "reported_score values are transcribed from the team's own submission "
            "notes; they are not DrivenData receipts and cannot be re-derived here.",
            "th_dti is a simulation: the held-out traces are mapped faults in the same "
            "style as the training traces, whereas the contest's private faults were "
            "identified by experts specifically because they are missing from the "
            "catalogue. A th_dti gain is necessary, not sufficient.",
            "A published raster was downloaded read-only from the team's public sibling "
            "repositories; the URL is recorded per file in docs/SOURCES.md (S14, S16).",
        ],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if not args.quiet:
        hdr = (f"{'candidate':38s} {'reported':>8s} {'DTI(cat)':>9s} {'TH-DTI':>7s} "
               f"{'area>0':>7s} {'offCat%':>7s} {'p_mass':>12s}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            rep = f"{r['reported_score']:.4f}" if r["reported_score"] is not None else "-"
            print(f"{r['name']:38s} {rep:>8s} {r['dti_catalogue']:9.4f} {r['th_dti']:7.4f} "
                  f"{100 * r['area_frac_p_gt_0']:6.2f}% "
                  f"{100 * r['off_catalogue_mass_frac']:6.1f}% {r['total_p_mass']:12.0f}")
        print(f"\nspatial trace holdout: {args.folds} folds x {args.block_px} px "
              f"blocks, {args.collar_px} px collar; scored faults "
              f"{sum(h.score_px for h in holds):,} px; official mask applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
