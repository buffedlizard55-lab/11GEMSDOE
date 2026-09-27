#!/usr/bin/env python3
"""Validate a candidate submission file against every official rule, from disk.

Use this before every upload.  It exists because a team member already lost an
upload to DrivenData's "Predicted values must be in range [0, 1]" rejection, and
because a rejection costs a weekly slot.

    python scripts/validate_submission.py data/submission.tif
    python scripts/validate_submission.py data/submission.tif --json report.json

Exit code 0 means "every check below passed".  Non-zero means do not upload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gems.raster import ConformanceError, check_submission_raster, sha256_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="candidate .tif")
    ap.add_argument("--template", default=str(ROOT / "data" / "sample_submission.tif"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    path, template = Path(args.path), Path(args.template)
    for p in (path, template):
        if not p.exists():
            print(f"FATAL: {p} not found", file=sys.stderr)
            return 2

    try:
        report = check_submission_raster(path, template)
    except ConformanceError as exc:
        print("NOT SUBMISSION-READY", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        if args.json:
            Path(args.json).write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    c = report["checks"]
    p = report["profile"]
    print(f"PASS  {path.name}")
    print(f"  sha256            {sha256_file(path)}")
    print(f"  bytes             {path.stat().st_size:,}")
    print(f"  bands / dtype     {p['count']} / {p['dtype']}")
    print(f"  crs / transform   {p['crs']} / {p['transform']}")
    print(f"  shape             {p['width']} x {p['height']}  (template {report['template_profile']['width']}"
          f" x {report['template_profile']['height']})")
    print(f"  value range       [{c['inside_min']}, {c['inside_max']}] over "
          f"{c['template_valid_pixels']:,} in-footprint px")
    print(f"  out of [0,1]      {c['inside_out_of_range_pixels']}")
    print(f"  outside bounds    {c['outside_finite_pixels']} finite px (must be 0; null/NaN outside)")
    print(f"  nodata            {p['nodata']}")
    for w in report["warnings"]:
        print(f"  WARNING: {w}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
