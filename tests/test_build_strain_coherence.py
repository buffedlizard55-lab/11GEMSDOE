import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "build_strain_coherence", Path(__file__).parents[1] / "scripts" / "build_strain_coherence.py"
)
build_strain_coherence = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader

try:
    import numpy as np
    import rasterio  # noqa: F401
    from scipy.ndimage import gaussian_filter  # noqa: F401
except ImportError:
    np = None

if np is not None:
    SPEC.loader.exec_module(build_strain_coherence)


@unittest.skipUnless(np is not None, "requires optional Rasterio, NumPy, and SciPy")
class StrainCoherenceTests(unittest.TestCase):
    def test_outputs_eight_finite_features_on_valid_mask(self):
        y, x = np.mgrid[:96, :112].astype(np.float32)
        values = {
            "geod_2ndinv": 0.8 * x + 0.3 * y,
            "geod_shearrate": 0.2 * x - 0.9 * y,
            "geod_dilaterate": 0.5 * x + 0.7 * y,
        }
        valid = np.ones(x.shape, dtype=bool)
        valid[:4, :] = False
        fields, names, cutpoints = build_strain_coherence.build_fields(values, valid)

        self.assertEqual(len(fields), 8)
        self.assertEqual(len(set(names)), 8)
        self.assertEqual(len(cutpoints), 6)
        for field in fields:
            self.assertEqual(field.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(field[valid])))
            self.assertTrue(np.all(field[~valid] == build_strain_coherence.NODATA))
        for index in (0, 4):
            self.assertTrue(np.all((fields[index][valid] >= 0) & (fields[index][valid] <= 1)))
        for index in (1, 2, 5, 6):
            self.assertTrue(np.all((fields[index][valid] >= -1) & (fields[index][valid] <= 1)))
        for index in (3, 7):
            self.assertTrue(np.all(fields[index][valid] >= 0))

    def test_degenerate_layer_range_is_rejected(self):
        valid = np.ones((32, 32), dtype=bool)
        values = {name: np.ones(valid.shape, dtype=np.float32) for name in build_strain_coherence.REQUIRED}
        with self.assertRaisesRegex(ValueError, "degenerate robust range"):
            build_strain_coherence.build_fields(values, valid)


if __name__ == "__main__":
    unittest.main()
