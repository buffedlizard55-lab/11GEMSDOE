"""Emit and validate a contest submission raster.

Every rule below is a quotation from the official pages; the validator is written
so that a candidate is either provably acceptable or refused, never "probably fine".

Official submission format
    https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
    "Your submission is in the same projected coordinate reference system as the
     training data (projected coordinate system for UTM zone 11N, EPSG 32611)"
    "Your submission is at the same resolution as the training data (100m)"
    "Your submission has the same bounds as the training data, and data outside
     the bounds is null or nan."
    "Your submission contains a single layer with datatype of 32-bit float
     (float32) with values between 0 and 1 indicating the confidence or
     probability of fault presence, with higher values indicating higher
     probability."

Official upload form
    DrivenData competition 306: "You can submit a single-band GeoTIFF (.tif)
    file, or a .zip file containing a single GeoTIFF, with your predictions. It
    must match the submission format's CRS, shape, and geotransform." and rejects
    a file with "Predicted values must be in range [0, 1]".

Why the [0, 1] rejection happens, and what this module does about it
--------------------------------------------------------------------
The most common causes, in order of how often they bite:
  1. a nodata sentinel written as a real number (e.g. -9999) instead of NaN. The
     platform reads every pixel of the band, so the sentinel is an out-of-range
     *value*;
  2. un-clipped model output - logits, counts, or an un-normalised score;
  3. a float64 or integer band promoted at write time;
  4. a multi-band stack, or a file whose transform differs by a rounding step.
``emit_submission`` closes all four by construction, and ``check_submission_raster``
re-reads the file from disk and fails loudly if any of them survived.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from .raster import check_submission_raster, pixel_sha256, sha256_file

__all__ = ["emit_submission", "submission_manifest", "shape_for_submission"]


def shape_for_submission(prob: np.ndarray, area_fraction: float | None = None,
                         floor: float = 0.0, ceiling: float = 1.0,
                         include_catalogue: bool = True,
                         catalogue: np.ndarray | None = None,
                         valid: np.ndarray | None = None) -> np.ndarray:
    """Turn a probability field into the raster that will actually be submitted.

    * every value is clipped into [0, 1] - the submission-format rule;
    * ``area_fraction`` emits the top fraction of in-footprint pixels by probability
      and zeroes the rest. This is the *scored* shaping decision, and it is
      expressed as an area rather than a probability threshold because a threshold
      does not transfer: a model fitted on held-out folds emits ~9% of unseen
      geography at p >= 0.10, while the same pipeline fitted on the whole catalogue
      emits ~54% at p >= 0.10, because in-sample probabilities are shifted. The
      area is what carries over. Measured response of the official metric on the
      leakage-free spatial holdout (control arm, mean of 4 folds):
      2% -> 0.1120, 4% -> 0.1292, **6% -> 0.1324**, 8% -> 0.1306, 10% -> 0.1304,
      15% -> 0.1246, 20% -> 0.1165, 40% -> 0.0943. Full table in
      ``docs/evidence/emission_choice_2026-09-27.json``;
    * ``floor``, if given, is an additional absolute probability cut;
    * ``include_catalogue`` sets the supplied USGS/INGENIOUS faults to 1.0. The
      organiser confirmed those pixels "are masked / excluded from evaluation, so
      they do not count towards penalty terms"
      (https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516),
      so including them is free, and the problem page asks for a prediction for
      "all faults in the region".
    """
    out = np.asarray(prob, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=ceiling, neginf=0.0)
    out = np.clip(out, 0.0, ceiling)
    if area_fraction is not None:
        pool = (out > 0)
        if valid is not None:
            pool &= np.asarray(valid).astype(bool)
        if float(area_fraction) <= 0:
            keep = np.zeros_like(pool)
        else:
            vals = out[pool]
            thr = float(np.quantile(vals, 1.0 - float(area_fraction))) if vals.size else 1.0
            keep = pool & (out >= thr)
        out = np.where(keep, 1.0, 0.0).astype(np.float32)
    if floor > 0:
        out[out < floor] = 0.0
    if include_catalogue and catalogue is not None:
        out[np.asarray(catalogue).astype(bool)] = 1.0
    return out


def emit_submission(values: np.ndarray, template_path: str | Path, out_path: str | Path,
                    band_description: str = "fault presence probability",
                    tags: dict | None = None) -> dict:
    """Write the GeoTIFF, then prove from disk that it satisfies every stated rule."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(template_path) as tpl:
        tpl_profile = tpl.profile.copy()
        height, width = tpl.height, tpl.width
        tpl_valid = np.isfinite(tpl.read(1))
        if not tpl_valid.any():
            tpl_valid = np.ones((height, width), dtype=bool)
        tpl_crs, tpl_transform = tpl.crs, tpl.transform

    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (height, width):
        raise ValueError(f"prediction shape {arr.shape} != template {(height, width)}")

    # Outside the data bounds the official page requires null/NaN - not a sentinel.
    arr = np.where(tpl_valid, arr, np.nan).astype(np.float32)

    profile = dict(tpl_profile)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                   width=width, height=height, crs=tpl_crs, transform=tpl_transform)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    # The contest requires the CRS, shape and geotransform to match; it says nothing
    # about the storage layout. DEFLATE with the floating-point predictor shrinks a
    # mostly-binary float32 band by roughly an order of magnitude versus the
    # template's LZW, which matters when the file is published as a download.
    profile.update(compress="deflate", predictor=3, zlevel=9)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        dst.set_band_description(1, band_description)
        dst.update_tags(**(tags or {}))

    report = check_submission_raster(out_path, template_path)
    report["band_description"] = band_description
    report["out_of_range_before_write"] = int(
        ((np.nan_to_num(values, nan=0.0) < 0) | (np.nan_to_num(values, nan=0.0) > 1)).sum())
    return report


def submission_manifest(path: str | Path, name: str, note: str,
                        provenance: dict) -> dict:
    """The record the DrivenData 'Note (optional)' field and our ledger both need.

    ``name`` must be unique across the team's submissions: the form asks for "A
    short comment to help you or your team tell submissions apart later e.g.
    clustering with k=25".  This manifest is the machine-readable form of that
    note, and every hash in it is measured, not asserted.
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        band = ds.read(1)
        finite = np.isfinite(band)
        lo = float(band[finite].min())
        hi = float(band[finite].max())
        mean = float(band[finite].mean())
        area = float((band[finite] > 0).mean())
        profile = {
            "count": ds.count, "dtype": ds.dtypes[0], "crs": str(ds.crs),
            "transform": list(ds.transform)[:6], "width": ds.width, "height": ds.height,
            "nodata": ds.nodata,
        }
    return {
        "file": str(path),
        "file_name": path.name,
        "submission_name": name,
        "drivendata_note": note,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "pixel_sha256": pixel_sha256(band),
        "profile": profile,
        "value_stats": {
            "min": lo, "max": hi, "mean": mean,
            "area_fraction_gt_0": area,
        },
        "provenance": provenance,
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
