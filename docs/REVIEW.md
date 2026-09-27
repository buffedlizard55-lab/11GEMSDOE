# Repository review and session findings

**Session date:** 2026-09-27 (UTC)
**Branch:** `arena/01a0e06c-11gemsdoe`, from `e709769`
**Prior state:** the data blocker was open — no official raster could be placed, so no experiment could run and no valid file could be produced.

**Outcome:** the blocker is closed, the official metric is implemented and verified against the official worked example, five hypotheses were measured on a leakage-free holdout, two are rejected, one defect-free submission file exists and validates, and the emission geometry turned out to matter more than every model change combined.

---

## 1. The data blocker is closed

`scripts/download_competition_data.sh` places all three official rasters plus both auxiliary stacks and verifies each against `config/data_pins.json`; `scripts/prepare_data.py` then checks the grid, the CRS, the band inventory and the label statistics and exits non-zero on any mismatch.

| File | Bytes | SHA-256 | Verified |
|---|---:|---|---|
| `training_features.tif` | 418,912,844 | `4371c82e3b8339b8…` | matches the pin |
| `labels.tif` | 425,830 | `7ba308ccdc4418b3…` | matches the pin |
| `sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd…` | matches the pin |
| `external/topo_u8.tif` | 32,523,329 | `a6398d9950965dec…` | matches the pin |
| `external/radiometric_u8.tif` | 25,475,158 | `6cb051f70f941fd7…` | matches the pin |

**Stated limitation, unchanged and not hidden.** The official DrivenData data tab requires an enrolled account and redirected to login. The script tries the official Dropbox mirrors first; in this sandbox they are unreachable (TLS handshake fails for every non-GitHub host), so it falls back to a public, checksum-pinned transport bridge published by a sibling repository. The *bytes* are verified; the *origin* is a Git mirror rather than a fresh pull from DrivenData. **Repairing this is the first task of the next session** and needs nothing but a login.

Band names are read from the file's own `band_name` tags and checked against the prose list on the official page, so the two cannot drift apart silently. The 19 bands are: `mag_anom, rtp, tmi_hg, geod_2ndinv, iso_grav_anom_slope, tc, geod_shearrate, geod_dilaterate, tmi_vg, deq_n100a15, iso_grav_anom_vg, det_elev, iso_grav_anom, tmi, depth_to_base_surf, ieq_n100a15, cond_surf, iso_grav_anom_hg, det_elev_slope`. The label raster has 60,988 positive pixels — 1.18% of the 5,167,373-pixel in-footprint grid.

## 2. The metric is now written, and verified against the official example

The organizer's reference repository contains **no** implementation of the contest metric — it uses `segmentation_models_pytorch.losses.TverskyLoss(alpha=0.2, beta=0.8)` as a *training loss* and does not implement the distance-weighted scoring. So `src/gems/metric.py` is a transcription of the three published equations, and it is checked three ways:

1. **Against the page's own worked example.** `tests/test_metric.py::test_official_worked_example_is_reproduced_component_by_component` rebuilds a configuration with the page's exact component values and asserts the implementation returns TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 and DTI = 0.6026…, which the page rounds to 0.60.
2. **Against a literal loop-for-loop reimplementation** of the same three equations, on five random arrays.
3. **Against the structural identity** `TP_w + FN_w = |G|`, which follows from the two sums for any prediction.

`tests/test_pipeline.py` additionally pins the submission-format rules and the 0–1 rejection, and `tests/test_metric.py::test_gt_scorer_matches_the_general_path` pins the fast scorer against the general definition with the scored region deliberately different from the valid mask (see the defect in §6).

## 3. Why 0.1563 kept coming back — two different answers, both measured

### 3.1 For GEMSDOE1 and 5GEMSDOE: the same file, confirmed again

The three published copies (in 5GEMSDOE, GEMSDOE and 7GEMSDOE) are byte-identical: 570,890 bytes, SHA-256 `7f00890a…`, identical pixel arrays. Unchanged from the 2026-09-27 audit. That is the entire explanation for that pair.

### 3.2 For everything else: we were optimising a proxy the contest removes

This is the substantive finding of the session.

The organiser stated on 2026-09-16: *"Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms."* Every detector family this team has built was selected against the **mapped catalogue**. On the real metric that is worth exactly nothing. The contest is, by design, a test of whether you can find structure the cartographers missed.

Re-scoring every published file with our own verified metric on a **leakage-free** holdout (contiguous 512 px blocks, 1.2 km collar removed from both sides, official mask applied, a catalogue-reproducing prediction scores 0.0000 so the split carries no shortcut):

| File | reported | DTI vs catalogue | DTI on unseen faults | emitted area | mass >300 m from any known fault |
|---|---:|---:|---:|---:|---:|
| GEMSDOE1/5GEMSDOE `ens12-adopted` | 0.1563 | 0.2298 | 0.2304 | 3.35% | 78.4% |
| GEMSDOE2 `dual-family-union` | 0.1560 | 0.2382 | 0.2389 | 3.55% | 76.8% |
| GEMSDOE2 `extension-arm` | — | 0.4463 | 0.4470 | 5.65% | 48.3% |
| GEMSDOE2 `precision-arm` | — | 0.1387 | 0.1407 | 0.42% | 48.1% |
| GEMSDOE3 `pindrop nodes` | 0.1193 | 0.2068 | 0.2059 | 3.00% | 91.2% |
| GEMSDOE3 `pindrop catalogue-gap` | 0.0830 | 0.1778 | 0.1767 | 3.00% | 91.5% |
| GEMSDOE3 `pindrop dense ridge` | 0.1152 | 0.1957 | 0.1948 | 3.00% | 80.5% |

**The honest caveat that makes this table work as evidence:** these files were fitted on the *entire* catalogue, including the traces being held out, so their holdout numbers are optimistic in exactly the way that matters — they can recognise the neighbourhood of a held-out fault. They score 0.18–0.45 on this proxy and 0.083–0.156 on the contest. **That gap is the finding**: the proxy is not a proxy, it is the shortcut. Our own models, which never saw the held-out blocks, score 0.13 on the same instrument.

8GEMSDOE still shows 0.1563 from a raster with different bytes and different pixels, and the public leaderboard shows several unrelated competitors at 0.1563. Four-decimal score equality is not file identity, and 0.1563 is not a fingerprint.

## 4. Hypotheses measured, and what happened to them

Full detail in [Hypotheses](HYPOTHESES.md); numbers in [`evidence/emission_choice_2026-09-27.json`](evidence/emission_choice_2026-09-27.json) and [`evidence/holdout_control_structural_2026-09-27.json`](evidence/holdout_control_structural_2026-09-27.json).

| Arm | What changed | Holdout DTI at 6% area (4 folds) | Paired delta vs control |
|---|---|---:|---:|
| control | 19 official bands + 9 topographic + 7 radiometric | 0.1324 | — |
| **structural** | + 9 H2/H3 derived channels | **0.1366** | +0.0042, t = 1.29, wins 9/10 area targets |
| inpaint | control + H4 trace-inpainting objective | 0.1370 | +0.0046, t = 0.88, wins 8/10 area targets |

**H1 (strain structure tensor) remains rejected** from the earlier session: 0.149795 against 0.149986.

**None of the model arms is significant on four folds.** A paired t of 1.3 on n = 4 is not evidence, and the site says so. H2/H3 and H4 are indistinguishable from each other (0.1366 vs 0.1370). The structural arm is used for the downloadable file because it wins more area targets and has the better paired t — that is a tie-break, not a claim.

### The result that actually moved the number

Sweeping the **emission geometry** on the same folds, with the same predictions and no refitting:

| emitted area | 2% | 4% | **6%** | 8% | 10% | 15% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DTI (control) | 0.1120 | 0.1292 | **0.1324** | 0.1306 | 0.1304 | 0.1246 | 0.1165 | 0.1042 | 0.0943 |

**0.0943 → 0.1324, a 40% relative gain from a shaping parameter** — an order of magnitude larger than every model variant tested. The metric charges every pixel of emitted area at α = 0.2 and forgives positional error out to 300 m; those two facts alone fix the optimum near 6%. Widening the corridor beyond that is strictly harmful on this holdout: at the same floor, 0 px = 0.1320, 1 px = 0.1193, 2 px = 0.1120, 3 px = 0.1058.

A corollary that cost us a rebuild: **the optimum must be expressed as an area, not as a probability threshold.** A fold model emits ~9% of unseen geography at p ≥ 0.10; the same pipeline fitted on the whole catalogue emits 54% at p ≥ 0.10, because in-sample probabilities are shifted. Thresholding the final model at 0.10 produced a 10.8 MB file covering half the grid. The shipped file uses an area target instead.

## 5. Two validation defects we caught in our own code

Both produced a result that looked excellent and was meaningless. Both are now covered by tests, and the leakage gate is a hard abort in the experiment runner.

1. **The emission sweep was emitting the answer key.** The sweep added the full label raster to the prediction to emulate "include the known catalogue, which is free". Inside a scored block that raster *is* the held-out truth. Every arm scored exactly 1.0000 at high floors and the inpainting arm appeared to beat the control by **+0.19** — a fake, and the biggest fake this project has produced. Fixed to add the fold's *training* catalogue only.
2. **The fast scorer charged the whole map as false positive.** `GtScorer` did not zero predictions outside the scored region, so three quarters of every prediction was billed. Every number read about ten times too low, and the sweep's apparent optimum was nonsense. Fixed, with a parity test that deliberately uses a region different from the valid mask.

## 6. A methodological result worth keeping: a random-trace holdout leaks

The first version of the trace holdout removed a random 10% of *traces* and scored the rest of the grid with the official mask applied. It returned a DTI of **0.97** for a model that had never seen those traces. The mechanism is worth writing down:

- the kernel reaches 300 m, so a prediction on a *training* fault earns true-positive credit for a held-out trace that runs within 300 m of it, and
- the official mask makes predictions on known faults free, so that credit costs nothing.

Held-out traces are rarely more than 300 m from a mapped neighbour. The fix is spatial — hold out contiguous blocks, hide every training trace within a collar of them, score only inside the held-out blocks — and to check the fix by measuring what a catalogue-reproducing prediction earns. `leakage_probe` now does that and the runner aborts above 0.25. On the shipped split it returns **0.0000**.

## 7. The downloadable file

`docs/downloads/gems-structural-area06-v1.tif` — 705,023 bytes, SHA-256 `6292a7916c83d371fc650b0f0c6a090702b9890eaa6e6546d50f104eea2fadc3`, single band, float32, EPSG:32611, 100 m, 3292 × 3730, values in [0, 1], NaN outside the bounds, 6.6% of the in-footprint grid emitted. `scripts/validate_submission.py` re-derives every one of those from the file on disk and exits non-zero on any failure; the report is published beside it.

**Strategic trade-off, named rather than hidden.** This is an open competition with a public leaderboard. Publishing a validated artefact hands the identical file to every other entrant. We publish it because a validated, reproducible artefact is what this repository exists to produce and the provenance is the point, but that is a judgement, not a neutral act, and a team that disagrees can withhold the file without changing anything else.

## 8. What the numbers do not establish

- **A holdout score is not a leaderboard score.** 0.1366 is measured on held-out *mapped* faults. The private set is made of faults an expert judged to be missing from the map, which may be systematically subtler. The gap between the 0.23–0.45 proxy numbers of our historical files and their 0.083–0.156 contest scores is direct evidence of how far a proxy can mislead.
- **Four folds cannot resolve 0.004.** The honest statement is "H2/H3 and H4 are neutral-to-slightly-positive on this instrument".
- **The exact geometry of the official known-fault mask is not published.** We implement the literal reading — the catalogue pixels themselves. If the platform masks a buffered corridor instead, every false-positive number here is slightly pessimistic.
- **Nothing here is a discovery.** A flagged pixel is a candidate structure until an expert looks at it.
- **The recorded H1 figure (0.149986) is not reproduced by this implementation.** The 2026-09-27 record used a scoring patch whose exact formulation is described only in prose. This session's re-implementation of a comparable block protocol returns ~0.05 on a stricter split. The difference is not explained and is **flagged as unresolved** rather than reconciled by assertion. Possible contributors: the stricter 1.2 km collar, the official mask, and the H1 record's use of global rather than held-out-only label geometry for false-positive weighting.

## 9. Next session, in priority order

1. **Re-anchor the data lineage** on the official DrivenData account. One command, needs a login, and it is the weakest link in everything above.
2. **Reconcile the 0.149986 discrepancy** by re-running the 2026-09-27 protocol verbatim from the sibling repository, with its own scoring patch, on identical folds. Until that is explained, the H1 record and this session's numbers live in different measurement systems and should not be compared in any write-up.
3. **Scale the holdout** — more folds, more seeds, and a block-bootstrap. The current instrument has ~0.018 fold dispersion, which cannot resolve the effect sizes on offer.
4. **Test H7 (cross-catalogue disagreement)** properly: check the licence of any candidate compilation against the competition's external-data clause, and enumerate its coverage of the GeoDAWN footprint, before writing any code.
5. **Run the organizer's U-Net reference** on a GPU runner as a *different detector family* and blend it with the boosted-tree field. The team's own notes claim the CNN and the booster produce nearly disjoint supports, which makes a blend the obvious untried move — but that claim is currently unverified here and must be measured before it is relied on.
6. **Never auto-upload.** Submission slots are scarce, receipts are the team's only proof, and the public round is not the final round.

## 10. Irregularities kept on the record

- The official `example_submission.tif` is **not** an empty raster. The problem page calls it "a sample submission that predicts total fault absence". Measured, it is the supplied catalogue rasterised to float32 — 1.0 on all 60,988 catalogue fault pixels, 0 elsewhere in the footprint, NaN outside, bit-for-bit equal to `(labels.tif > 0)`. A competitor who submitted the template unchanged would have predicted exactly the faults the private set excludes, and would have scored zero. **Do not copy the template.**
- The reported team scores remain user-reported transcriptions. Only `extradr19` at 0.1563 can be corroborated against the public leaderboard, and no upload receipt is accessible from this environment.
- The data bridge is a team-published mirror. Verified bytes, unverified origin.
- The 0.149986 figure is not reproduced (§8).
- A sibling project's notes cite "Hermant et al. 2025, cited by the organisers" for a ~400 m catalogue-vs-lidar offset claim. That paper was not opened, so it is used nowhere in this repository.
- The public leaderboard is a dated snapshot and changes. Re-fetch before quoting a rank.
