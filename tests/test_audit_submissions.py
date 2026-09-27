import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit_submissions", Path(__file__).parents[1] / "scripts" / "audit_submissions.py"
)
audit_submissions = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(audit_submissions)

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    np = None
    rasterio = None
    from_origin = None


class AuditSubmissionTests(unittest.TestCase):
    def test_identical_bytes_reported_as_exact_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.tif"
            second = root / "second.tif"
            first.write_bytes(b"not a geotiff, but identical bytes")
            second.write_bytes(first.read_bytes())
            result = audit_submissions.audit([first, second])
            self.assertEqual(result["exact_byte_duplicates"], [[str(first.resolve()), str(second.resolve())]])

    def test_different_files_are_not_called_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.tif"
            second = root / "second.tif"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            result = audit_submissions.audit([first, second])
            self.assertEqual(result["exact_byte_duplicates"], [])
            self.assertEqual(result["same_grid_and_pixel_duplicates"], [])

    def test_recursive_directory_collection_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "nested" / "candidate.TIFF").write_bytes(b"a")
            (root / "notes.txt").write_text("ignored")
            self.assertEqual(len(audit_submissions.collect_files([root])), 1)

    def test_missing_path_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_submissions.collect_files([Path(tmp) / "missing.tif"])

    @unittest.skipUnless(rasterio is not None, "requires optional Rasterio and NumPy")
    def test_pixel_hash_ignores_tiff_storage_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiled = root / "tiled.tif"
            striped = root / "striped.tif"
            data = np.arange(512 * 512, dtype=np.float32).reshape(512, 512) / (512 * 512)
            profile = {
                "driver": "GTiff", "width": 512, "height": 512, "count": 1,
                "dtype": "float32", "crs": "EPSG:32611", "transform": from_origin(0, 51200, 100, 100),
                "nodata": float("nan"),
            }
            with rasterio.open(tiled, "w", **profile, tiled=True, blockxsize=256, blockysize=256) as ds:
                ds.write(data, 1)
            with rasterio.open(striped, "w", **profile, tiled=False) as ds:
                ds.write(data, 1)
            first = audit_submissions.raster_details(tiled)
            second = audit_submissions.raster_details(striped)
            self.assertEqual(first["issues"], [])
            self.assertEqual(second["issues"], [])
            self.assertEqual(first["pixel_sha256"], second["pixel_sha256"])
            self.assertNotEqual(audit_submissions.file_sha256(tiled), audit_submissions.file_sha256(striped))

    @unittest.skipUnless(rasterio is not None, "requires optional Rasterio and NumPy")
    def test_prediction_range_violation_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tif"
            profile = {
                "driver": "GTiff", "width": 4, "height": 4, "count": 1,
                "dtype": "float32", "crs": "EPSG:32611", "transform": from_origin(0, 400, 100, 100),
                "nodata": float("nan"),
            }
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(np.full((4, 4), 1.1, dtype=np.float32), 1)
            result = audit_submissions.raster_details(path)
            self.assertTrue(any("outside [0, 1]" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
