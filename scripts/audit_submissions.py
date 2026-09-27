#!/usr/bin/env python3
"""Audit local GEMS GeoTIFF submissions for duplicates and core format constraints.

Requires Python 3.10+. Raster metadata/value/pixel checks require optional Rasterio
(`pip install rasterio`). SHA-256 byte comparison works without Rasterio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def collect_files(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for item in paths:
        if item.is_dir():
            found.update(p.resolve() for p in item.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
        elif item.is_file() and item.suffix.lower() in {".tif", ".tiff"}:
            found.add(item.resolve())
        else:
            raise FileNotFoundError(f"Not a TIFF file or directory: {item}")
    return sorted(found, key=lambda p: str(p).casefold())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raster_details(path: Path, template_path: Path | None = None) -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError("Rasterio and NumPy are required for GeoTIFF validation; install rasterio to enable this check.") from exc

    issues: list[str] = []
    pixel_hash = hashlib.sha256()
    try:
        with rasterio.open(path) as src:
            grid = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": src.crs.to_string() if src.crs else None,
                "transform": tuple(src.transform)[:6],
                "dtype": src.dtypes[0] if src.count else None,
                "nodata": src.nodata,
            }
            if src.count != 1:
                issues.append(f"expected exactly one band, found {src.count}")
            if src.count < 1:
                return {"grid": grid, "issues": issues + ["raster has no bands"], "pixel_sha256": None}
            if src.dtypes[0] != "float32":
                issues.append(f"expected float32, found {src.dtypes[0]}")
            if src.crs is None:
                issues.append("CRS is missing")
            elif src.crs.to_epsg() != 32611:
                issues.append(f"expected EPSG:32611, found {src.crs.to_string()}")
            # A rotated affine grid is allowed only if both pixel-axis lengths are 100 m.
            x_resolution = math.hypot(src.transform.a, src.transform.d)
            y_resolution = math.hypot(src.transform.b, src.transform.e)
            if not (math.isclose(x_resolution, 100.0, abs_tol=1e-6) and math.isclose(y_resolution, 100.0, abs_tol=1e-6)):
                issues.append(f"expected 100 m pixel resolution, found {x_resolution:g} m by {y_resolution:g} m")

            finite_min = math.inf
            finite_max = -math.inf
            valid_count = 0
            invalid_count = 0
            outside_valid_count = 0
            # Fixed-width row strips make the semantic pixel hash independent of TIFF tiling.
            for row0 in range(0, src.height, 256):
                window = Window(0, row0, src.width, min(256, src.height - row0))
                band = src.read(1, window=window, masked=False)
                mask = src.read_masks(1, window=window) > 0
                finite = np.isfinite(band)
                valid = mask & finite
                invalid_count += int((mask & ~finite).sum())
                valid_count += int(valid.sum())
                if valid.any():
                    lo = float(band[valid].min())
                    hi = float(band[valid].max())
                    finite_min = min(finite_min, lo)
                    finite_max = max(finite_max, hi)
                    outside_valid_count += int(((band[valid] < 0) | (band[valid] > 1)).sum())
                # Canonical content digest: independent of TIFF compression, block size, and byte order.
                canonical = np.asarray(band, dtype="<f4").copy()
                canonical[~valid] = 0
                pixel_hash.update(np.ascontiguousarray(canonical).tobytes())
                pixel_hash.update(np.ascontiguousarray(valid.astype("uint8")).tobytes())

            if valid_count == 0:
                issues.append("raster contains no finite, unmasked prediction pixels")
            if invalid_count:
                issues.append(f"{invalid_count} unmasked pixels are NaN or infinite")
            if outside_valid_count:
                issues.append(f"{outside_valid_count} finite, unmasked pixels fall outside [0, 1]")
            grid_header = json.dumps({k: grid[k] for k in ("width", "height", "count", "crs", "transform", "dtype")}, sort_keys=True, default=str).encode()
            pixel_hash.update(grid_header)
            stats = {
                "finite_valid_pixels": valid_count,
                "nonfinite_unmasked_pixels": invalid_count,
                "outside_range_pixels": outside_valid_count,
                "minimum": None if finite_min == math.inf else finite_min,
                "maximum": None if finite_max == -math.inf else finite_max,
            }

        if template_path is not None:
            with rasterio.open(template_path) as template, rasterio.open(path) as src:
                same_shape = (src.width, src.height) == (template.width, template.height)
                same_crs = src.crs == template.crs
                same_transform = src.transform == template.transform
                if not same_shape:
                    issues.append("width/height do not match the template")
                if not same_crs:
                    issues.append("CRS does not match the template")
                if not same_transform:
                    issues.append("geotransform does not match the template")
                if same_shape:
                    # Template's mask defines locations outside the intended footprint.
                    for _, window in template.block_windows(1):
                        template_valid = template.read_masks(1, window=window) > 0
                        candidate_valid = src.read_masks(1, window=window) > 0
                        if np.any(candidate_valid & ~template_valid):
                            issues.append("prediction has unmasked values outside the template's valid footprint")
                            break
                        if np.any(template_valid & ~candidate_valid):
                            issues.append("prediction is missing finite, unmasked values inside the template's valid footprint")
                            break
        return {"grid": grid, "stats": stats, "pixel_sha256": pixel_hash.hexdigest(), "issues": issues}
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {"issues": [f"could not read raster: {type(exc).__name__}: {exc}"], "pixel_sha256": None}


def audit(paths: list[Path], template: Path | None = None) -> dict[str, Any]:
    files = collect_files(paths)
    if not files:
        raise ValueError("No .tif/.tiff files found")
    if template is not None and not template.is_file():
        raise FileNotFoundError(f"Template not found: {template}")

    rows: list[dict[str, Any]] = []
    for path in files:
        row: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        try:
            row.update(raster_details(path, template))
        except RuntimeError as exc:
            row["raster_validation"] = "not_run"
            row["issues"] = [str(exc)]
        rows.append(row)

    by_file: dict[str, list[str]] = {}
    by_pixels: dict[str, list[str]] = {}
    for row in rows:
        by_file.setdefault(row["sha256"], []).append(row["path"])
        if row.get("pixel_sha256"):
            by_pixels.setdefault(row["pixel_sha256"], []).append(row["path"])
    exact_groups = [items for items in by_file.values() if len(items) > 1]
    pixel_groups = [items for items in by_pixels.values() if len(items) > 1]
    failed = [row["path"] for row in rows if row.get("issues")]
    return {
        "files": rows,
        "exact_byte_duplicates": exact_groups,
        "same_grid_and_pixel_duplicates": pixel_groups,
        "validation_failures": failed,
        "interpretation": "A matching displayed competition score does not prove duplicate predictions. Compare file SHA-256 and same-grid pixel hashes; retain the submission IDs and provenance as well.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GeoTIFF files or directories to scan")
    parser.add_argument("--template", type=Path, help="Optional official sample/template GeoTIFF for exact grid and footprint comparison")
    parser.add_argument("--json", type=Path, help="Optional path to write the full JSON audit report")
    args = parser.parse_args(argv)
    try:
        report = audit(args.paths, args.template)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["validation_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
