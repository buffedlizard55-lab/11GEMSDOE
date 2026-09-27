"""Feature-stack construction from the official 19 bands plus derived structural channels.

WHAT IS IN THE OFFICIAL STACK (read from the file's own ``band_name`` tags, and
cross-checked against the prose list on the official problem page)
----------------------------------------------------------------------
The official page, "Provided features", lists surface conductivity and depth to
conductive base, detrended elevation and its slope, dilatation rate / shear strain
rate / strain second invariant, isostatic gravity anomaly and its slope, magnetics
(reduced-to-pole anomaly, total magnetic intensity, vertical and horizontal
gradient, tilt angle, top-of-crustal source depth) and earthquake density.
https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/
``scripts/prepare_data.py`` fails if a named layer is missing from the raster, so
prose and file cannot silently drift apart.

HOW THE STACK IS BUILT
----------------------
* Every channel is percentile-clipped to its p1..p99 range over the valid
  footprint and stored as ``uint8``.  That is 12.3 MB per channel instead of
  49 MB, so a 43-channel cube fits in memory on a 2 vCPU / 3 GB machine, and the
  gradient-boosted model does not care about the monotone rescaling.
* The official stack's nodata is -3.4e38 (float32 minimum).  It is converted to
  NaN, then every filtered channel is forced back to the low quantile outside the
  *eroded* footprint, so a filter never smears -3.4e38 arithmetic through the map.
* Filtering happens one source band at a time and is written straight into a
  memory-mapped cube on disk, so peak RAM stays near one 49 MB band.

DERIVED CHANNELS (the new hypotheses - see docs/HYPOTHESES.md)
-------------------------------------------------------------
H2  magnetic / gravity fabric coherence.  A fault is not "a place where the
    magnetic field is steep"; it is a *linear, persistent discontinuity* in a
    smooth regional field.  The structure tensor of the horizontal gradient
    measures whether the gradient direction is locally consistent - a coherent
    zone of edge pixels is fabric, an isolated high-gradient blob is a contact.
H3  basement / conductive contact.  ``depth_to_base_surf`` is the structural
    datum of the cover; a range-bounding fault steps it.  The target is the step
    and its coherence, not the depth value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import (gaussian_filter, maximum_filter, minimum_filter,
                           uniform_filter)

__all__ = ["FeatureSpec", "DERIVED_CHANNELS", "build_feature_cube", "channel_names"]

#: Sentinel used by the official stack for "no data".  Comparing against -1e30
#: catches the float32 minimum without needing an exact bit pattern.
NODATA_SENTINEL = -1e30

#: Erosion applied to the valid footprint before filtering, in pixels, so that a
#: Gaussian/structure-tensor kernel never mixes real values with filled nodata.
EDGE_EROSION_PX = 10


@dataclass(frozen=True)
class FeatureSpec:
    """Which channels to build and why."""

    use_aux: bool = True          # add the 9-band topographic and 7-band radiometric stacks
    structural: bool = True       # add the H2/H3 derived channels
    n_quantiles: int = 256

    def as_dict(self) -> dict:
        return {"use_aux": self.use_aux, "structural": self.structural,
                "n_quantiles": self.n_quantiles}


#: (name, source bands, description) for every derived channel.
DERIVED_CHANNELS: list[tuple[str, tuple[str, ...], str]] = [
    ("rtp_grad_coh_s3", ("rtp",),
     "H2 structure-tensor coherence of the horizontal gradient of the RTP anomaly, sigma=3 px (300 m)"),
    ("rtp_grad_coh_s8", ("rtp",),
     "H2 same at sigma=8 px (800 m): a fabric that survives a kilometre is structural, not a contact"),
    ("grav_grad_coh_s3", ("iso_grav_anom",),
     "H2 structure-tensor coherence of the horizontal gradient of the isostatic gravity anomaly, sigma=3 px"),
    ("grav_grad_coh_s8", ("iso_grav_anom",),
     "H2 same at sigma=8 px"),
    ("tmi_hg_coh_s3", ("tmi_hg",),
     "H2 coherence of the supplied total-magnetic-intensity horizontal-gradient band, sigma=3 px"),
    ("dtb_lap_s3", ("depth_to_base_surf",),
     "H3 magnitude of the Laplacian of the basement surface: a linear, persistent step in depth, sigma=3 px"),
    ("dtb_grad_coh_s3", ("depth_to_base_surf",),
     "H3 structure-tensor coherence of the basement-surface gradient, sigma=3 px"),
    ("dtb_range5", ("depth_to_base_surf",),
     "H3 local relief of the basement surface in a 5x5 px (500 m) window: a throw proxy"),
    ("cond_grad_coh_s3", ("cond_surf",),
     "H3 structure-tensor coherence of the surface-conductivity gradient, sigma=3 px"),
]


def channel_names(feature_path: Path, aux_paths: list[Path], spec: FeatureSpec) -> list[str]:
    with rasterio.open(feature_path) as ds:
        base = [ds.tags(i).get("band_name") or f"band{i}" for i in range(1, ds.count + 1)]
    if spec.use_aux:
        for p in aux_paths:
            with rasterio.open(p) as ds:
                base += [f"{p.stem}:{ds.tags(i).get('band_name') or f'b{i}'}"
                         for i in range(1, ds.count + 1)]
    if spec.structural:
        base += [n for n, _, _ in DERIVED_CHANNELS]
    return base


# ---------------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------------
def _fill_nodata(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace nodata with the valid-footprint median so filters see real numbers."""
    out = np.where(valid, a, np.nan).astype(np.float32)
    if not valid.any():
        return np.zeros_like(out)
    med = np.nanmedian(out[valid]) if valid.any() else 0.0
    out[~np.isfinite(out)] = med
    return out


def _structure_tensor_coherence(field: np.ndarray, sigma: float) -> np.ndarray:
    """``(sqrt((Jxx-Jyy)^2 + 4Jxy^2)) / (Jxx+Jyy)`` with J the smoothed gradient tensor.

    0 where the gradient direction is isotropic (a blob or a smooth field), 1 where
    it is a single consistent orientation (a line - a fabric, a fault, a fracture).
    Central differences, then Gaussian smoothing of each tensor component, which is
    the standard coherence measure (Förstner/Uhelnick).
    """
    gx = np.gradient(field, axis=1)
    gy = np.gradient(field, axis=0)
    jxx = gaussian_filter(gx * gx, sigma, mode="nearest")
    jyy = gaussian_filter(gy * gy, sigma, mode="nearest")
    jxy = gaussian_filter(gx * gy, sigma, mode="nearest")
    num = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    den = jxx + jyy
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _quantiles(a: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    """Map ``a`` onto ``0..n-1`` using its p1..p99 range over the valid footprint."""
    v = a[valid]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(v, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(v)), float(np.max(v)) + 1e-9
    q = np.rint((np.clip(a, lo, hi) - lo) / (hi - lo) * (n - 1))
    return np.clip(q, 0, n - 1).astype(np.uint8)


# ---------------------------------------------------------------------------------
# cube builder
# ---------------------------------------------------------------------------------
def build_feature_cube(feature_path: Path, aux_paths: list[Path], spec: FeatureSpec,
                       out_path: Path, chunk_rows: int = 512,
                       log=print) -> dict:
    """Build the uint8 feature cube and write provenance JSON next to it."""
    from scipy.ndimage import binary_erosion

    with rasterio.open(feature_path) as ds:
        height, width = ds.height, ds.width
        base_names = [ds.tags(i).get("band_name") or f"band{i}" for i in range(1, ds.count + 1)]
        first = ds.read(1)
        valid = first > NODATA_SENTINEL
        valid &= np.isfinite(first)
        del first
        transform = tuple(ds.transform)[:6]
        crs = str(ds.crs)

    aux_names: list[str] = []
    if spec.use_aux:
        for p in aux_paths:
            with rasterio.open(p) as ad:
                aux_names += [f"{p.stem}:{ad.tags(i).get('band_name') or f'b{i}'}"
                              for i in range(1, ad.count + 1)]
    struct_names = [n for n, _, _ in DERIVED_CHANNELS] if spec.structural else []
    names = base_names + aux_names + struct_names

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cube = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                                     shape=(len(names), height, width))

    # ---- erosion so filters never mix filled nodata into the edge ---------------
    from scipy.ndimage import binary_dilation
    interior = binary_erosion(binary_dilation(valid, iterations=EDGE_EROSION_PX * 3),
                              iterations=EDGE_EROSION_PX)
    usable = valid & interior

    # ---- pass 1: the supplied bands, read and requantised in row chunks ---------
    col = 0
    with rasterio.open(feature_path) as ds:
        for b in range(1, ds.count + 1):
            arr = ds.read(b)
            keep = valid & np.isfinite(arr)
            cube[col] = _quantiles(np.where(keep, arr, np.nan_to_num(arr, nan=0.0)),
                                   usable, spec.n_quantiles)
            cube[col][~usable] = 0
            col += 1
            log(f"  base {col:2d}/{len(names)}  {base_names[b - 1]}")
    del arr, keep

    if spec.use_aux:
        for p in aux_paths:
            with rasterio.open(p) as ad:
                for b in range(1, ad.count + 1):
                    arr = ad.read(b)
                    keep = valid & (arr > 0)          # aux stacks use 0 as nodata
                    cube[col] = _quantiles(np.where(keep, arr, 0.0), usable, spec.n_quantiles)
                    cube[col][~usable] = 0
                    col += 1
                    log(f"  aux  {col:2d}/{len(names)}  {aux_names[col - len(base_names) - 1]}")
            del arr, keep

    # ---- pass 2: derived structural channels, one source band at a time ---------
    if spec.structural:
        with rasterio.open(feature_path) as ds:
            by_name = {ds.tags(i).get("band_name") or f"band{i}": i for i in range(1, ds.count + 1)}
        cache: dict[str, np.ndarray] = {}

        def get(band: str) -> np.ndarray:
            if band not in cache:
                with rasterio.open(feature_path) as ds:
                    raw = ds.read(by_name[band]).astype(np.float32)
                cache.clear()          # one source band in RAM at a time
                cache[band] = _fill_nodata(raw, valid)
                del raw
            return cache[band]

        for name, sources, _desc in DERIVED_CHANNELS:
            f = get(sources[0])
            if name.endswith("coh_s3"):
                sig = 3.0
            elif name.endswith("coh_s8"):
                sig = 8.0
            else:
                sig = 3.0
            if name.endswith("lap_s3"):
                out = np.abs(gaussian_filter(np.gradient(np.gradient(f, axis=0), axis=0), sig,
                                             mode="nearest")
                             + gaussian_filter(np.gradient(np.gradient(f, axis=1), axis=1), sig,
                                               mode="nearest"))
            elif name.endswith("range5"):
                out = (maximum_filter(f, size=5, mode="nearest")
                       - minimum_filter(f, size=5, mode="nearest"))
            else:
                out = _structure_tensor_coherence(f, sig)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            cube[col] = _quantiles(out, usable, spec.n_quantiles)
            cube[col][~usable] = 0
            col += 1
            log(f"  new  {col:2d}/{len(names)}  {name}")
        cache.clear()

    cube.flush()
    del cube

    prov = {
        "feature_path": str(feature_path),
        "feature_sha256": None,
        "aux_paths": [str(p) for p in aux_paths] if spec.use_aux else [],
        "spec": spec.as_dict(),
        "shape": [len(names), height, width],
        "channels": names,
        "derived_channels": [
            {"name": n, "sources": list(s), "description": d}
            for n, s, d in (DERIVED_CHANNELS if spec.structural else [])
        ],
        "crs": crs,
        "transform": list(transform),
        "valid_pixels": int(valid.sum()),
        "usable_pixels": int(usable.sum()),
        "edge_erosion_px": EDGE_EROSION_PX,
        "nodata_sentinel": NODATA_SENTINEL,
        "dtype": "uint8",
    }
    prov_path = out_path.with_suffix(".json")
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    return prov
