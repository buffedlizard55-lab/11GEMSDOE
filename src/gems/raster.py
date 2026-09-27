"""GeoTIFF helpers: grid identity, conformance checks, byte/pixel hashing.

All rules implemented here come from two official pages:

* Submission format - https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
      "Your submission is in the same projected coordinate reference system as
       the training data (projected coordinate system for UTM zone 11N, EPSG 32611)"
      "Your submission is at the same resolution as the training data (100m)"
      "Your submission has the same bounds as the training data, and data
       outside the bounds is null or nan."
      "Your submission contains a single layer with datatype of 32-bit float
       (float32) with values between 0 and 1 indicating the confidence or
       probability of fault presence, with higher values indicating higher
       probability."

* Upload form - DrivenData competition 306 submission page, which validates
  "Predicted values must be in range [0, 1]" and accepts "a single-band GeoTIFF
  (.tif) file, or a .zip file containing a single GeoTIFF" that "must match the
  submission format's CRS, shape, and geotransform".

Nothing in this module writes a file.  Writing lives in ``submission.py`` so that
there is exactly one code path that can produce a contest artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

__all__ = [
    "Grid",
    "sha256_file",
    "pixel_sha256",
    "read_grid",
    "check_submission_raster",
    "ConformanceError",
]


class ConformanceError(RuntimeError):
    """Raised when a candidate submission violates a stated submission-format rule."""


@dataclass(frozen=True)
class Grid:
    """The identity of a competition raster: CRS, transform, shape, nodata."""

    crs: str
    transform: tuple[float, ...]
    width: int
    height: int
    count: int
    dtype: str

    def as_dict(self) -> dict:
        return {
            "crs": self.crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtype": self.dtype,
        }


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Whole-file SHA-256, streamed so a 400 MB raster costs 1 MB of RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    """Hash of the *values* on a grid, independent of TIFF layout.

    Two files can be byte-different and pixel-identical (compression, tags,
    strip layout).  For "did we submit the same prediction twice?" the pixel
    hash is the honest question; the byte hash is the file-identity question.
    Both are recorded.
    """
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def read_grid(path: str | Path) -> Grid:
    with rasterio.open(path) as src:
        return Grid(
            crs=str(src.crs),
            transform=tuple(src.transform)[:6],
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
        )


def check_submission_raster(path: str | Path, template: str | Path,
                            max_report: int = 5) -> dict:
    """Validate a candidate submission against the official rules and ``template``.

    Returns a report dict; raises ``ConformanceError`` on the first hard failure.
    The template is the official ``sample_submission.tif`` whose grid the contest
    page tells competitors to copy.
    """
    path, template = Path(path), Path(template)
    report: dict = {"file": str(path), "checks": {}, "errors": [], "warnings": []}

    with rasterio.open(path) as sub, rasterio.open(template) as tpl:
        report["profile"] = {
            "count": sub.count,
            "dtype": sub.dtypes[0],
            "crs": str(sub.crs),
            "transform": list(sub.transform)[:6],
            "width": sub.width,
            "height": sub.height,
            "nodata": sub.nodata,
            "bounds": list(sub.bounds),
        }
        report["template_profile"] = {
            "crs": str(tpl.crs),
            "transform": list(tpl.transform)[:6],
            "width": tpl.width,
            "height": tpl.height,
            "nodata": tpl.nodata,
        }

        if sub.count != 1:
            report["errors"].append(f"must contain a single layer, found {sub.count}")
        if sub.dtypes[0] != "float32":
            report["errors"].append(f"datatype must be float32, found {sub.dtypes[0]}")
        if str(sub.crs) != str(tpl.crs):
            report["errors"].append(
                f"CRS {sub.crs} does not match the template CRS {tpl.crs} (EPSG:32611)")
        if tuple(sub.transform)[:6] != tuple(tpl.transform)[:6]:
            report["errors"].append("geotransform does not match the template geotransform")
        if (sub.width, sub.height) != (tpl.width, tpl.height):
            report["errors"].append(
                f"shape {sub.width}x{sub.height} does not match template {tpl.width}x{tpl.height}")

        band = sub.read(1)
        # Valid = inside the template footprint. Outside it must be null/NaN.
        tpl_band = tpl.read(1)
        tpl_valid = np.isfinite(tpl_band)
        if not tpl_valid.any():
            tpl_valid = np.ones_like(band, dtype=bool)
        report["checks"]["template_valid_pixels"] = int(tpl_valid.sum())

        inside = band[tpl_valid]
        inside_finite = inside[np.isfinite(inside)]
        if inside_finite.size == 0:
            report["errors"].append("no finite values inside the template footprint")
            inside_finite = np.array([0.0], dtype=np.float32)

        lo, hi = float(inside_finite.min()), float(inside_finite.max())
        report["checks"]["inside_min"] = lo
        report["checks"]["inside_max"] = hi
        out_of_range = int(((inside_finite < 0.0) | (inside_finite > 1.0)).sum())
        report["checks"]["inside_out_of_range_pixels"] = out_of_range
        if out_of_range:
            report["errors"].append(
                f"{out_of_range} pixel(s) inside the footprint are outside [0, 1] "
                f"(min {lo!r}, max {hi!r}); DrivenData rejects these with "
                f'"Predicted values must be in range [0, 1]"')

        outside = band[~tpl_valid]
        outside_finite = outside[np.isfinite(outside)]
        report["checks"]["outside_finite_pixels"] = int(outside_finite.size)
        if outside_finite.size:
            report["warnings"].append(
                f"{outside_finite.size} finite value(s) outside the data bounds; the "
                f"official page requires those to be null or NaN")

        report["checks"]["nan_pixels"] = int(np.isnan(band).sum())
        report["checks"]["nodata_set"] = sub.nodata is not None

    report["ok"] = not report["errors"]
    if report["errors"]:
        raise ConformanceError(
            f"{path.name} is not submission-ready: " + "; ".join(report["errors"]))
    return report


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
