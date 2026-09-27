# 11GEMSDOE — GEMS Prize research and submission project

> **Session startup: read this charter before proposing changes.** Keep work evidence-led, reproducible, and focused on verified fault discoveries—not merely on producing another differently named GeoTIFF.

## Project charter (the original brief, made actionable)

Our goal is to develop a scientifically defensible, distinct, and competitive approach for the DOE Geologic Enhanced Mapping System (GEMS) Prize. Study the contest and its data, preserve an auditable research record with trusted links, and use reproducible experiments to improve fault predictions. The competition requires one single-band float32 GeoTIFF in EPSG:32611 at 100 m resolution, with the training grid's extent and confidence values in [0, 1]. Do not label a file submission-ready until the template, bounds, CRS, resolution, datatype, range, and nodata rules have been checked.

Use the scores shared by the team and the live public leaderboard as context—not as proof that two files are identical or that a method generalizes. The reported repeated score of 0.1563 is suspicious enough to audit, but four-decimal score equality alone cannot establish file reuse: scores are rounded, and different prediction rasters can produce the same displayed score. Record unique model/version names and concise submission notes. Compare exact file hashes and, when possible, georeferenced pixel arrays. Keep a submission ledger.

Before implementing a new model idea:

1. Read `docs/REVIEW.md`, `docs/HYPOTHESES.md`, and `docs/SOURCES.md`.
2. Describe at least three candidate geological hypotheses, their target signature, relevant feature layers, novelty relative to prior experiments, expected value, and cost.
3. Rank candidates explicitly, then evaluate the top candidate on a **spatially blocked holdout** using the official distance-weighted Tversky metric. Report folds, masks, seeds, scores, and uncertainty. A leaderboard score is not a substitute for this holdout.
4. Do not spend a contest submission on an idea that has not beaten the current holdout best under a fair paired comparison. Do not call an unrun or unmeasured idea viable.
5. If an experiment needs external data, identify the exact free, official source and verify that it is accessible and licensed for contest use before promoting the idea. No unofficial source, unsupported claim, invented result, or silent change to a submission is acceptable.

The official data tab redirects to DrivenData login in this environment, but the public sibling project 5GEMSDOE exposes checksum-pinned competition rasters and the project repositories publish the previous candidate files. This review assembled the data privately under excluded `.cache/`, verified its checksums to the bridge manifest, and compared published submission bytes. The primary GEMSDOE1 and 5GEMSDOE files are the exact same TIFF (SHA-256 `7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`), consistent with their repeated 0.1563 score. The separately published 8GEMSDOE raster also reports 0.1563 but has different pixels, illustrating why score equality alone is inconclusive. H1 strain-gradient orientation features were also tested against a paired four-fold spatial holdout: DTI 0.149795 versus control 0.149986 at the fixed threshold, so H1 did not beat the holdout best and is not approved for submission. See `docs/REVIEW.md` and `docs/evidence/h1_holdout_2026-09-27.json` for evidence and limitations. The portal upload receipt remains unavailable, and the public bridge's official-source lineage is repository-recorded rather than directly re-downloaded here. No H1 submission was made.

## Core values

- **Maximize P(Win)**: make evidence-based choices that balance potential gain, uncertainty, time, and submission budget. Prefer one well-controlled discriminating experiment over several cosmetic variants.
- **Own the Outcome:** record negative results, close data/validation/format gaps, and be accountable from official source through reproducible file and final scoring. Flag blockers instead of disguising them.

## Quick navigation

- [Project dashboard and public leaderboard feed](docs/index.html)
- [Executive summary and submission instructions](docs/executive_summary.html)
- [Current leaderboard JSON snapshot](docs/leaderboard.json) · [six-hour Pages refresh workflow](.github/workflows/pages.yml)
- [Repository review and current blockers](docs/REVIEW.md)
- [Ranked hypothesis register](docs/HYPOTHESES.md)
- [Official sources and verification notes](docs/SOURCES.md)
- [H1 spatial holdout result](docs/evidence/h1_holdout_2026-09-27.json)
- [Historical submission identity audit](docs/evidence/submission_identity_audit_2026-09-27.json)
- [Experiment ledger template](docs/experiment_ledger_template.csv) · [submission ledger template](docs/submission_ledger_template.csv)
- [Submission duplicate/format audit tool](scripts/audit_submissions.py)

## Official contest anchors

- [Competition hub](https://www.drivendata.org/competitions/306/competition-doe-gems/)
- [Problem description, task, data layers, metric, and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [About GeoDAWN and data context](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
- [Competition data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (login required in this environment)
- [Public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/)
- [Official rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf)
- [Organizer reference solution](https://github.com/drivendataorg/gems-prize-reference-solution)
- [USGS GeoDAWN survey information](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and)
- [USGS ScienceBase GeoDAWN study-area item](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7)

## Development rule

Re-read this charter at the start of every project session. Update the evidence and hypothesis register as facts change. Keep large/private competition data and generated rasters out of Git unless there is an explicit, legal, and documented reason to publish them. Never claim a result is current without recording when and where it was checked.
