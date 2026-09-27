"""GEMS Prize research pipeline.

Every module in this package is written against a cited official source.  The
provenance for each rule is recorded in ``docs/SOURCES.md`` and repeated in the
module docstring that implements it.

Modules
-------
``metric``      distance-weighted Tversky index (the contest metric)
``raster``      GeoTIFF read/write/conformance helpers
``blocks``      spatially blocked cross-validation
``features``    feature-stack construction from the official 19 bands, plus the
                derived geophysical channels for hypotheses H2 and H3
``model``       gradient-boosted pixel detector (CPU, no GPU needed)
``submission``  submission raster emission and validation
"""

__all__ = [
    "metric",
    "raster",
    "blocks",
    "features",

    "model",
    "submission",
]
