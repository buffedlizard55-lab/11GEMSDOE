#!/usr/bin/env python3
"""Verify the placed competition rasters and print an auditable inventory.

Run after ``scripts/download_competition_data.sh``.  It never modifies data; it
reads, checks, prints and (optionally) writes ``data/inventory.json``.

Every assertion is a restatement of a rule written on an official page:

  CRS / resolution / grid
      https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
      "This GeoTIFF is in a projected coordinate system for UTM zone 11N
       (EPSG 32611) at 100m resolution."
      "Your submission is at the same resolution as the training data (100m) ...
       same bounds as the training data."

  band semantics
      same page, "Provided features": surface conductivity and depth to
      conductive base, detrended elevation and its slope, dilatation / shear
      strain rate and the strain second invariant, isostatic gravity anomaly and
      its slope, magnetics (RTP anomaly, total magnetic intensity, vertical and
      horizontal gradient, top-of-crustal source depth) and earthquake density.
      The script prints the ``band_name`` tag actually stored in the file so the
      prose and the raster can be compared line by line.

  prediction range
      same page, "Submission format": "values between 0 and 1"; and the
      submission form's "Predicted values must be in range [0, 1]".

Exit status is non-zero if any pinned hash or structural rule fails, so this can
be used as a CI gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.raster import read_grid, sha256_file  # noqa: E402

PINS = ROOT / "config" / "data_pins.json"

#: Rules the official problem page states in prose.  Checked against the file's
#: own band_name tags, so a silent re-projection or renaming cannot pass.
PROSE_LAYERS = {
    "magnetic": ["mag_anom", "rtp", "tmi", "tmi_vg", "tmi_hg", "tc"],
    "gravity": ["iso_grav_anom", "iso_grav_anom_slope", "iso_grav_anom_vg", "iso_grav_anom_hg"],
    "geodetic_strain": ["geod_2ndinv", "geod_shearrate", "geod_dilaterate"],
    "topographic": ["det_elev", "det_elev_slope"],
    "subsurface": ["cond_surf", "depth_to_base_surf"],
    "seismic": ["ieq_n100a15", "deq_n100a15"],
}

EXPECTED_CRS = "EPSG:32611"
EXPECTED_RES = 100.0
EXPECTED_BANDS = 19


def band_names(ds: rasterio.DatasetReader) -> list[str]:
    return [ds.tags(i).get("band_name") or (ds.descriptions[i - 1] or f"band{i}")
            for i in range(1, ds.count + 1)]


def check(path: Path, expected_sha: str, expected_bytes: int) -> tuple[dict, list[str]]:
    errs: list[str] = []
    if not path.exists():
        return {}, [f"{path.name}: missing (run scripts/download_competition_data.sh)"]
    got_sha = sha256_file(path)
    got_bytes = path.stat().st_size
    if got_sha != expected_sha:
        errs.append(f"{path.name}: sha256 {got_sha} != pinned {expected_sha}")
    if got_bytes != expected_bytes:
        errs.append(f"{path.name}: {got_bytes} bytes != pinned {expected_bytes}")
    with rasterio.open(path) as ds:
        info = {
            "path": str(path),
            "sha256": got_sha,
            "bytes": got_bytes,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs),
            "transform": list(ds.transform)[:6],
            "res": [abs(ds.transform.a), abs(ds.transform.e)],
            "width": ds.width,
            "height": ds.height,
            "nodata": ds.nodata,
            "bounds": list(ds.bounds),
        }
    return info, errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"), help="directory holding the rasters")
    ap.add_argument("--json", default=None, help="also write the inventory JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    pins = json.loads(PINS.read_text())
    errors: list[str] = []
    inventory: dict = {"data_dir": str(data), "files": {}, "prose_layer_coverage": {}}

    for entry in pins["files"]:
        path = data / entry["canonical_name"]
        info, errs = check(path, entry["sha256"], entry["bytes"])
        errors += errs
        inventory["files"][entry["canonical_name"]] = info
        if not args.quiet:
            if info:
                print(f"{entry['canonical_name']}: {info['bytes']:,} bytes  sha256 {info['sha256'][:16]}...")
            else:
                print(f"{entry['canonical_name']}: MISSING")
        if info:
            if info["crs"] != EXPECTED_CRS:
                errors.append(f"{entry['canonical_name']}: CRS {info['crs']} != {EXPECTED_CRS}")

    # ---- structural checks on the feature stack --------------------------------
    feat_path = data / "training_features.tif"
    if feat_path.exists():
        with rasterio.open(feat_path) as ds:
            names = band_names(ds)
            res = (abs(ds.transform.a), abs(ds.transform.e))
            inventory["feature_stack"] = {
                "band_names": names,
                "band_descriptions": list(ds.descriptions),
                "n_bands": ds.count,
                "res": list(res),
            }
            if ds.count != EXPECTED_BANDS:
                errors.append(f"training_features.tif: {ds.count} bands, expected {EXPECTED_BANDS}")
            if res != (EXPECTED_RES, EXPECTED_RES):
                errors.append(f"training_features.tif: resolution {res} != 100 m")
            present = set(names)
            for category, expected in PROSE_LAYERS.items():
                missing = [n for n in expected if n not in present]
                inventory["prose_layer_coverage"][category] = {
                    "expected": expected, "missing": missing}
                if missing:
                    errors.append(
                        f"training_features.tif: prose layer category {category} is missing "
                        f"{missing} (band_name tags read from the file: {sorted(present)})")
            if not args.quiet:
                print("\nband_name tags read from training_features.tif (file is the authority):")
                for i, n in enumerate(names, 1):
                    print(f"  {i:2d}  {n}")

    # ---- label semantics -------------------------------------------------------
    lab_path = data / "labels.tif"
    if lab_path.exists():
        with rasterio.open(lab_path) as ds:
            lab = ds.read(1)
            vals, counts = np.unique(lab, return_counts=True)
            inventory["labels"] = {
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata,
                "value_counts": {str(v): int(c) for v, c in zip(vals, counts)},
                "n_positive": int((lab > 0).sum()),
                "positive_fraction": float((lab > 0).mean()),
            }
            if not args.quiet:
                print(f"\nlabels.tif: {lab.shape[0]}x{lab.shape[1]}, values {vals.tolist()}, "
                      f"positive px {int((lab > 0).sum()):,} "
                      f"({100 * float((lab > 0).mean()):.3f}% of the grid)")
            if int((lab > 0).sum()) == 0:
                errors.append("labels.tif contains no positive pixels")

    # ---- template conformity ---------------------------------------------------
    tpl_path = data / "sample_submission.tif"
    sub_path = data / "submission.tif"
    if tpl_path.exists():
        tpl = read_grid(tpl_path)
        inventory["template"] = tpl.as_dict()
        with rasterio.open(tpl_path) as ds:
            band = ds.read(1)
            finite = np.isfinite(band)
            inventory["template"]["finite_pixels"] = int(finite.sum())
            inventory["template"]["min"] = float(band[finite].min())
            inventory["template"]["max"] = float(band[finite].max())
            if not args.quiet:
                print(f"sample_submission.tif: {tpl.width}x{tpl.height} {tpl.crs} "
                      f"{tpl.dtype}, {int(finite.sum()):,} finite px, "
                      f"range [{float(band[finite].min()):g}, {float(band[finite].max()):g}]")
        if sub_path.exists():
            with rasterio.open(sub_path) as ds:
                b = ds.read(1)
                f = np.isfinite(b)
                rng_ok = bool(f.any() and b[f].min() >= 0.0 and b[f].max() <= 1.0)
                inventory["candidate_submission"] = {
                    "sha256": sha256_file(sub_path),
                    "dtype": ds.dtypes[0],
                    "crs": str(ds.crs),
                    "grid_matches_template": tuple(ds.transform)[:6] == tpl.transform
                    and (ds.width, ds.height) == (tpl.width, tpl.height),
                    "min": float(b[f].min()) if f.any() else None,
                    "max": float(b[f].max()) if f.any() else None,
                    "in_range_0_1": rng_ok,
                }
                if not rng_ok:
                    errors.append("data/submission.tif has values outside [0, 1] - "
                                  "DrivenData rejects these with "
                                  '"Predicted values must be in range [0, 1]"')
                if not args.quiet:
                    print(f"submission.tif: {inventory['candidate_submission']['sha256'][:16]}... "
                          f"range [{inventory['candidate_submission']['min']}, "
                          f"{inventory['candidate_submission']['max']}]")

    inventory["errors"] = errors
    if args.json:
        Path(args.json).write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    elif not args.quiet:
        print("\n(inventory JSON not written; pass --json PATH to persist it)")

    if errors:
        print("\nFAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("\nPASS: data/ matches the official grid, hashes and layer inventory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
