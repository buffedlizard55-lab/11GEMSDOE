# 11GEMSDOE — GEMS Prize research and submission project

> **Session startup: read this charter before proposing anything.** It is the standing brief for this repository. Keep work evidence-led, reproducible, and aimed at faults that are *missing* from the catalogue — not at producing another differently named GeoTIFF.

---

## The operating brief (read this first, every session)

You are working on the U.S. Department of Energy **Geologic Enhanced Mapping System (GEMS) Prize Challenge**: predict where geological faults are in the GeoDAWN region of the north-western Great Basin, in order to find new geothermal resources.

```markdown
## Mission
Top the public leaderboard of the DOE GEMS Prize Challenge with a scientifically
defensible fault prediction — and leave behind a research record someone else
could audit line by line, with official links, that is useful beyond this contest.

## Non-negotiables
1. No hallucinations. Every factual claim is traced to an official or otherwise
   named source, quoted, and linked. If a source was not opened, the claim is not
   made. If a source cannot be obtained, say so and name what is needed.
2. No manual input required. The pipeline must run end to end unattended. If a
   step needs a human, it is a blocker to be named, not a question to be asked.
3. Flag irregularities; never smooth them over. A duplicate file, a
   mis-transcribed score, an unsourced number and a stale snapshot are all
   recorded in the open, with the evidence and the caveat.
4. The work must be genuinely distinct. A new feature transform applied to the
   same model and the same objective is not a new idea. Ask what mechanism
   changes, and say so explicitly.
5. A submission slot is a scarce resource. Never spend one on an idea that has
   not beaten the current holdout best under a fair, paired comparison.
6. Maximize P(Win). One well-controlled discriminating experiment beats ten
   cosmetic variants.
7. Own the outcome end to end: official source → reproducible experiment →
   valid GeoTIFF → unique name and note → recorded score → published evidence.

## Standing questions
- Why did the team's submissions keep returning 0.1563, and are the files
  actually identical? Answer with hashes and re-scored rasters, never with a
  guess.
- Which geological signal is present in the data, is absent from the supplied
  catalogue, and is therefore worth predicting?
- What is the smallest experiment that can distinguish the hypotheses that
  matter? Run that one.

## Definition of done for an idea
Named layers · named physical signature · why it targets a catalogue gap and not
a catalogue hit · how it differs from everything already tried · expected DTI
gain · implementation cost · a measured, paired, leakage-free holdout result with
uncertainty — positive or negative.
```

## Project charter (the original brief, made actionable)

The competition requires **one single-band float32 GeoTIFF in EPSG:32611 at 100 m resolution, with the training grid's extent and confidence values in [0, 1]**, and null/NaN outside the data bounds. Do not call a file submission-ready until the template, bounds, CRS, resolution, datatype, range and nodata rules have been checked *by re-reading the written file from disk*.

Use the scores shared by the team and the live public leaderboard as context — not as proof that two files are identical, and not as a validation baseline. Scores are rounded to four decimals, and the public leaderboard currently shows **several unrelated competitors at 0.1563**, so an equal displayed score proves nothing on its own. Compare SHA-256 and georeferenced pixel arrays. Keep a submission ledger.

Before implementing a new model idea:

1. Read `docs/REVIEW.md`, `docs/HYPOTHESES.md` and `docs/SOURCES.md`.
2. Describe at least three candidate geological hypotheses, each naming its layers, target signature, catalogue-gap rationale, novelty, expected value and cost.
3. Rank them explicitly.
4. Evaluate the top candidate against its **paired control**, on the same folds, with the official distance-weighted Tversky metric and the official scoring mask.
5. Do not spend a submission slot on an idea that has not beaten the holdout best. Do not call an unrun idea viable.
6. If it needs external data, name the exact free official source, check the licence permits contest use, and check it is actually obtainable, *before* proposing it.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

bash scripts/download_competition_data.sh --with-aux   # official rasters, SHA-256 pinned
.venv/bin/python scripts/prepare_data.py               # grid / band / label audit
.venv/bin/python -m pytest -q                          # 38 tests, incl. the official metric example

.venv/bin/python scripts/diagnose_scores.py            # re-score every historical file
.venv/bin/python scripts/run_holdout_experiment.py \
    --arms control,structural,inpaint --json docs/evidence/holdout_h2_h3_h4_2026-09-27.json
.venv/bin/python scripts/choose_emission.py --json docs/evidence/emission_choice_2026-09-27.json
.venv/bin/python scripts/build_submission.py --arm structural --area-fraction 0.06
.venv/bin/python scripts/validate_submission.py data/gems-structural-area06-v1.tif
```

The data blocker that stopped the previous session is **closed**: all three official rasters plus both auxiliary stacks are placed in `data/`, hash-verified against `config/data_pins.json`. No DrivenData credentials were needed, because `scripts/download_competition_data.sh` falls back to a checksum-pinned transport bridge; that fallback is documented as the weakest link in the lineage and is the first thing to repair.

## Core values

- **Maximize P(Win):** weigh trade-offs, risk and submission budget on every decision. Prefer one well-controlled discriminating experiment over several cosmetic variants. Put the project's chances of winning ahead of the comfort of any individual slice of the work.
- **Own the Outcome:** problems are ours from official source to final score. When we have the means to fix something, we do it without waiting to be asked. Failure and success are both signals. Blockers get named, not disguised.

## What we know now (all measured — details and links in [Review](docs/REVIEW.md))

- **GEMSDOE1 and 5GEMSDOE submitted the same bytes.** SHA-256 `7f00890a…`, 570,890 bytes, identical pixels. That is why both read 0.1563.
- **8GEMSDOE is a different file at the same score** — different SHA-256, different pixel hash. So 0.1563 is not a fingerprint of one prediction. The public leaderboard shows unrelated competitors at 0.1563 too.
- **The official sample submission is the catalogue, not an empty raster.** It is bit-for-bit `(labels.tif > 0)`. The problem page describes it as "predicts total fault absence"; it does not. Copying the template would score zero, because the private set is by definition the faults the catalogue does not contain.
- **Known faults are free, in both rounds.** "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms." — chrisk-dd, DrivenData Staff, 2026-09-16.
- **"New fault" includes newly mapped geometry of an existing system.** — chrisk-dd, 2026-09-22. The unmapped continuation of a mapped fault is a legitimate, in-scope target. That is the geological basis of the top-ranked hypothesis H4.
- **A random-trace holdout leaks.** Because known faults are free and the kernel reaches 300 m, a model can score on a held-out fault just by drawing its mapped neighbour. A naive split returned 0.97 for a model that had never seen those traces. `leakage_probe` now measures this and the experiment aborts if the split is leaky.
- **H1 (strain structure-tensor coherence) was tested and rejected**: 0.149795 against a control of 0.149986 on paired four-fold spatial blocks. It is kept in the register because negative results are evidence.

## Navigation

- [Project dashboard and public leaderboard feed](docs/index.html) — **the submission download is at the top of this page**
- [Executive summary and step-by-step submission guide](docs/executive_summary.html)
- [Ranked hypothesis register](docs/HYPOTHESES.md)
- [Evidence and source register](docs/SOURCES.md)
- [Repository review, findings and open blockers](docs/REVIEW.md)
- [H1 result](docs/evidence/h1_holdout_2026-09-27.json) · [control + structural arms](docs/evidence/holdout_control_structural_2026-09-27.json) · [inpaint arm](docs/evidence/holdout_inpaint_2026-09-27.json) · [emission-shape sweep](docs/evidence/emission_choice_2026-09-27.json)
- [Historical submission identity audit](docs/evidence/submission_identity_audit_2026-09-27.json) · [historical re-scoring](docs/historical/diagnosis.json)
- [Experiment ledger](docs/experiment_ledger.csv) · [submission ledger](docs/submission_ledger.csv) · [blank templates](docs/experiment_ledger_template.csv)
- [Downloadable submission GeoTIFF + manifest](docs/downloads/gems-structural-area06-v1.manifest.json)
- Submission audit tool `scripts/audit_submissions.py` · validator `scripts/validate_submission.py` · emission sweep `scripts/choose_emission.py` · diagnosis `scripts/diagnose_scores.py`

## Official contest anchors

- [Competition hub](https://www.drivendata.org/competitions/306/competition-doe-gems/)
- [Problem description, metric, and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/)
- [About GeoDAWN](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/)
- [Public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) · [data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (login required here)
- [Organiser: known faults are excluded from scoring](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)
- [Organiser: what counts as a "new fault"](https://community.drivendata.org/t/where-do-you-draw-the-line/11536)
- [Organiser: how the test faults were identified](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527)
- [Official rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) · [organizer reference solution](https://github.com/drivendataorg/gems-prize-reference-solution)
- [USGS GeoDAWN release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) · [ScienceBase item](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7)

## Development rule

Re-read the operating brief at the start of every session. Update the evidence and hypothesis registers as facts change. Keep large or competition-derived data and generated rasters out of Git unless there is an explicit, legal and documented reason to publish them. **Never claim a result is current without recording when and where it was checked.**
