#!/usr/bin/env python3
"""Build a strain-gradient structure-tensor feature stack for the H1 hypothesis.

Inputs are the three official GEMS layers tagged geod_2ndinv, geod_shearrate, and
geod_dilaterate. For each, robustly scale the valid data and derive gradients. Sum local
outer products of those gradients at two spatial supports (sigma 2.5 and 7.5 input pixels,
approximately 0.5 km and 1.5 km at the official 100 m grid), then write the coherence,
orientation (cos(2 theta), sin(2 theta)), and log edge-energy fields for each scale.

This creates features only; it does not train a model or establish fault presence. Rasterio,
NumPy, and SciPy are required. All outputs follow the input raster's grid and nodata footprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

NODATA = np.float32(-3.4028234663852886e38)
REQUIRED = ("geod_2ndinv", "geod_shearrate", "geod_dilaterate")
SCALES = (("500m", 2.5), ("1500m", 7.5))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_bands(src: rasterio.DatasetReader) -> dict[str, int]:
    found: dict[str, int] = {}
    for idx in range(1, src.count + 1):
        name = src.tags(idx).get("band_name", "").strip()
        if name in REQUIRED:
            if name in found:
                raise ValueError(f"duplicate feature band tag: {name}")
            found[name] = idx
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise ValueError(f"required band_name metadata missing: {', '.join(missing)}")
    return found


def build_fields(values: dict[str, np.ndarray], valid: np.ndarray) -> tuple[list[np.ndarray], list[str], dict[str, float]]:
    """Return eight structure-tensor bands plus robust scaling cut points."""
    grads: list[tuple[np.ndarray, np.ndarray]] = []
    cutpoints: dict[str, float] = {}
    for name in REQUIRED:
        raw = np.asarray(values[name], dtype=np.float32)
        sample = raw[valid & np.isfinite(raw)]
        if sample.size < 100:
            raise ValueError(f"not enough valid values for {name}: {sample.size}")
        lo, hi = np.percentile(sample, [1, 99]).astype(float)
        cutpoints[f"{name}_p01"] = lo
        cutpoints[f"{name}_p99"] = hi
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError(f"degenerate robust range for {name}: {lo}, {hi}")
        scaled = np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        scaled[~valid] = 0.0
        gy, gx = np.gradient(scaled)
        gx[~valid] = 0.0
        gy[~valid] = 0.0
        grads.append((gx, gy))

    fields: list[np.ndarray] = []
    names: list[str] = []
    for label, sigma in SCALES:
        jxx = np.zeros(valid.shape, dtype=np.float32)
        jxy = np.zeros(valid.shape, dtype=np.float32)
        jyy = np.zeros(valid.shape, dtype=np.float32)
        for gx, gy in grads:
            # Gaussian support smooths outer-product terms, not the original strain grids.
            jxx += gaussian_filter(gx * gx, sigma=sigma, mode="nearest")
            jxy += gaussian_filter(gx * gy, sigma=sigma, mode="nearest")
            jyy += gaussian_filter(gy * gy, sigma=sigma, mode="nearest")
        trace = np.maximum(jxx + jyy, 0.0)
        anisotropy = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy ** 2, 0.0))
        denom = np.maximum(anisotropy, np.float32(1e-12))
        coherence = np.clip(anisotropy / np.maximum(trace, np.float32(1e-12)), 0.0, 1.0)
        cos2 = np.clip((jxx - jyy) / denom, -1.0, 1.0)
        sin2 = np.clip((2.0 * jxy) / denom, -1.0, 1.0)
        energy = np.log1p(trace).astype(np.float32)
        for suffix, field in (("coherence", coherence), ("orientation_cos2", cos2),
                              ("orientation_sin2", sin2), ("log_edge_energy", energy)):
            field = np.asarray(field, dtype=np.float32)
            field[~valid] = NODATA
            fields.append(field)
            names.append(f"strain_{suffix}_{label}")
    return fields, names, cutpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="official multiband numerical_features GeoTIFF")
    parser.add_argument("output", type=Path, help="output multiband float32 auxiliary GeoTIFF")
    parser.add_argument("--report", type=Path, help="optional JSON provenance report")
    args = parser.parse_args()

    with rasterio.open(args.features) as src:
        band_indexes = find_bands(src)
        profile = src.profile.copy()
        arrays = {name: src.read(index) for name, index in band_indexes.items()}
        # The sample competition raster uses the survey footprint. Require all three source
        # bands to have finite non-sentinel values; do not infer validity from a single channel.
        valid = np.ones((src.height, src.width), dtype=bool)
        for name, index in band_indexes.items():
            band = arrays[name]
            valid &= (src.read_masks(index) > 0) & np.isfinite(band) & (band > np.float32(-1e30))
        grid = {
            "width": src.width,
            "height": src.height,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform)[:6],
            "count": src.count,
            "source_bands": band_indexes,
            "valid_pixels": int(valid.sum()),
        }
        if src.crs is None or src.crs.to_epsg() != 32611:
            raise ValueError(f"expected source EPSG:32611, found {src.crs}")
        if not (math.isclose(math.hypot(src.transform.a, src.transform.d), 100.0, abs_tol=1e-6)
                and math.isclose(math.hypot(src.transform.b, src.transform.e), 100.0, abs_tol=1e-6)):
            raise ValueError("expected 100 m source pixels")

    fields, names, cutpoints = build_fields(arrays, valid)
    profile.update(driver="GTiff", count=len(fields), dtype="float32", nodata=float(NODATA),
                   compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output, "w", **profile) as dst:
        for i, (name, field) in enumerate(zip(names, fields), start=1):
            dst.write(field, i)
            dst.set_band_description(i, name)
            dst.update_tags(i, band_name=name, hypothesis="H1 strain-gradient orientation coherence")
    report = {
        "hypothesis_id": "H1",
        "method": "multi-scale structure tensor over robustly scaled gradients of geod_2ndinv, geod_shearrate, geod_dilaterate",
        "scales_sigma_pixels": {name: sigma for name, sigma in SCALES},
        "source_path": str(args.features),
        "source_sha256": sha256_file(args.features),
        "source_grid": grid,
        "robust_percentile_cutpoints": cutpoints,
        "output_path": str(args.output),
        "output_sha256": sha256_file(args.output),
        "output_bands": names,
        "status": "feature construction only; no holdout score or fault discovery implied",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
