# Candidate geological hypotheses — preregistration, then measured

**Register opened 2026-09-27 (H1–H5), rewritten the same day after the data blocker cleared and the official scoring rules were found, then updated with measured results.**

## Measured results, 2026-09-27

Leakage-free spatial holdout: 4 contiguous 512 px blocks, 1.2 km collar removed from both sides, official metric with the official mask. A catalogue-reproducing prediction scores **0.0000** on this split, so there is no shortcut. All arms at their common optimum of 6% emitted area:

| arm | mechanism changed | DTI (mean of 4 folds) | paired delta vs control | wins across 10 area targets |
|---|---|---:|---:|---:|
| control | — | 0.1324 | — | — |
| **structural (H2+H3)** | nine derived geophysical channels | **0.1366** | +0.0042, t = 1.29 | 9 / 10 |
| inpaint (H4) | the training objective | 0.1370 | +0.0046, t = 0.88 | 8 / 10 |
| H1 (previous session) | strain structure tensor | 0.149795 vs 0.149986 control | −0.0002 | rejected |

**Read that table honestly.** Four folds give ~0.018 fold dispersion. A paired t of 1.3 is not significant. H2/H3 and H4 are statistically indistinguishable from the control and from each other. The structural arm ships only as a tie-break (more area targets won, better paired t), not as a demonstrated improvement.

**And the result that is large:** the *emission geometry*, which is not a model change at all.

| emitted area | 2% | 4% | **6%** | 8% | 10% | 15% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DTI (control) | 0.1120 | 0.1292 | **0.1324** | 0.1306 | 0.1304 | 0.1246 | 0.1165 | 0.1042 | 0.0943 |

A 40% relative gain from a shaping parameter — roughly an order of magnitude more than every model variant put together. Metric reasoning: the false-positive term is charged at α = 0.2 per unit of emitted mass, while the kernel forgives 300 m of positional error, so there is a finite optimal footprint and it is small. Corridor widening is strictly harmful here: 0 px 0.1320, 1 px 0.1193, 2 px 0.1120, 3 px 0.1058.

Two facts changed the register on 2026-09-27, both verified against official sources and quoted in full in [Sources](SOURCES.md):

1. **Known faults are free.** "Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms. Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults." — chrisk-dd, DrivenData Staff, 2026-09-16 ([thread](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516)).
2. **"New fault" is a generous label.** "For the purposes of this competition, 'new fault' means 'any fault pixel not already captured by USGS/INGENIOUS' and can include newly mapped geometry of an existing fault system." — chrisk-dd, 2026-09-22 ([thread](https://community.drivendata.org/t/where-do-you-draw-the-line/11536)).

Together these say the target is **not** "a brand-new fault zone in virgin ground". It is **any fault pixel a cartographer did not already draw** — which explicitly includes the continuation past a mapped end and the strand inside a mapped zone. That reframes every candidate below, and it is the single most useful thing found this session.

---

## What the official data actually contains

Read from the file's own `band_name` tags by `scripts/prepare_data.py`, which fails if the prose list and the raster disagree:

| category | bands |
|---|---|
| magnetic | `mag_anom`, `rtp`, `tmi`, `tmi_vg`, `tmi_hg`, `tc` (tilt angle / total curvature) |
| gravity | `iso_grav_anom`, `iso_grav_anom_slope`, `iso_grav_anom_vg`, `iso_grav_anom_hg` |
| geodetic strain | `geod_2ndinv`, `geod_shearrate`, `geod_dilaterate` |
| topographic | `det_elev`, `det_elev_slope` |
| subsurface | `cond_surf`, `depth_to_base_surf` |
| seismic | `ieq_n100a15`, `deq_n100a15` |

19 bands, EPSG:32611, 100 m, 3292 × 3730 px, 5,167,373 in-footprint pixels, 60,988 catalogue fault pixels (1.18% of the footprint). **The official stack carries no radiometrics and no 10 m topography** — the GeoDAWN airborne gamma and the 3DEP lidar are the two evidence families the contest's own feature stack does not contain, which is why the auxiliary channels matter.

## What the team has already tried (do not re-propose)

| family | evidence it was tried | where |
|---|---|---|
| CNN ensemble, 11 folds, ResNet34 U-Net/UNet++/DeepLabV3+ | the 0.1563 file, SHA-256 `7f00890a…` | `docs/evidence/submission_identity_audit_2026-09-27.json` |
| pixel gradient boosting ("Pindrop") in three variants — nodes, catalogue-gap, dense ridge | 0.1193 / 0.0830 / 0.1152 reported | `docs/historical/diagnosis.json` |
| dual-family union / precision arm / extension arm | 0.1560 reported for the union | `docs/historical/diagnosis.json` |
| DEM scarp channel from public 3DEP (9 bands, 10 m → 100 m) | used as an auxiliary | `data/external/topo_u8.tif` |
| radiometric ratio channel (7 bands) | used as an auxiliary | `data/external/radiometric_u8.tif` |
| H1 strain structure-tensor coherence | measured, **rejected** (0.149795 vs control 0.149986) | [evidence](evidence/h1_holdout_2026-09-27.json) |
| proxy catalogues, hidden-prior fitting, emission-width decision rules | family retired after the label-provenance check | sibling project records |

---

## Ranked candidates

Ranking is by expected improvement in the **distance-weighted Tversky index on faults the model has never seen**, divided by implementation cost. "Distinct" below means the *mechanism* differs, not that the feature list was permuted.

### Rank 1 — H4 · Trace-inpainting objective (the unmapped-continuation target)

* **Layers used:** the 19 official bands unchanged. The intervention is to the **objective**, not the features.
* **Physical signature targeted:** a fault system whose cartographic trace stops. A normal fault does not stop at a map edge; it steps, splays, or transfers to an en-echelon strand. The pixels just beyond a mapped endpoint, or in a gap along a mapped trace, are exactly the pixels that "newly mapped geometry of an existing fault system" describes — the definition the organiser gave on 2026-09-22.
* **Why it should catch a fault missing from the catalogue rather than one already in it:** a standard binary classifier is explicitly trained "no fault here" on every unlabelled pixel, including the unmapped continuation. H4 removes that contradiction: a random contiguous **terminal** fraction of each training trace is hidden from the model *and* excluded from the negative sample, so the model is never penalised for completing a trace. The model is then asked, at inference, for exactly the quantity the contest scores.
* **How it differs from anything in this repo:** no prior experiment changes the training objective. Everything prior changes features, thresholds, or post-processing. This is a different failure mode, not a different tuning.
* **Expected lift:** high. **Cost:** low — one label-manipulation function and one extra sampling pass, no new data.
* **Measured:** 0.1370 vs 0.1324 control, +0.0046, paired t = 0.88, wins 8 of 10 area targets. **Not significant.** Mechanism is implemented and tested (`_hide_trace_segments`, `tests/test_pipeline.py::test_inpaint_hides_terminal_segments_and_keeps_them_out_of_the_negatives`); it is parked, not rejected, because the effect could not be resolved on four folds and the hypothesis is cheap to re-test at scale.

### Rank 2 — H2 · Magnetic / gravity **fabric coherence** (not edge magnitude)

* **Layers used:** `rtp`, `iso_grav_anom`, `tmi_hg`.
* **Signature:** the structure tensor of the *horizontal gradient* — how consistently oriented the steepest-descent direction is over a 300 m and an 800 m window. Coherence ≈ 1 means a linear fabric; a high-gradient blob has low coherence. A lithologic contact is a step; a fault is a *persistent, oriented, laterally offset* discontinuity, and orientation persistence is the part that separates them.
* **Why a fault missing from the catalogue:** the catalogue records where a fault was *mapped*. A fault whose only surface expression is a long, aeromagnetically visible lineament can be entirely absent from the map while its magnetic signature is unmistakable. The two scales (300 m / 800 m) also let a subtle local break inherit confidence from the regional fabric.
* **Different from prior work:** the earlier "magnetic edge" framing selected high-gradient magnitude; the team's `context_detector` EDGE_BANDS list is magnitude-based. Orientation *coherence* is a different statistic and is invariant to the contrast of the contact, which magnitude is not.
* **Expected lift:** medium-high. **Cost:** low — 9 derived channels, no new data.
* **Measured:** bundled with H3 in the `structural` arm: 0.1366 vs 0.1324, +0.0042, paired t = 1.29, wins 9 of 10 area targets. **Not significant**, but the most consistent of the three arms and the one used for the shipped file.

### Rank 3 — H3 · Basement and conductive contact **step**

* **Layers used:** `depth_to_base_surf`, `cond_surf`.
* **Signature:** the magnitude of the Laplacian of the basement surface (a linear, persistent *step* in depth), the local 500 m relief of that surface (a throw proxy), and the structure-tensor coherence of its gradient. A range-bounding normal fault offsets the structural datum even where the surface has no scarp.
* **Why a fault missing from the catalogue:** blind and buried structures have little or no topographic expression — a well-known problem in the Basin and Range, and the GeoDAWN stack supplies exactly the layer that sees through cover. The official `depth_to_base_surf` band is in the contest's own feature file and was **not** used by any prior team detector in this family.
* **Different from prior work:** neither H1 (strain) nor the team's edge/corridor work touches the basement datum.
* **Expected lift:** medium. **Cost:** low.
* **Measured:** bundled with H2 in the `structural` arm, as above.

### Rank 4 — H6 · Emission geometry as an explicit, scored decision

* **Layers used:** none new.
* **Signature:** the metric's own geometry. A 1 px corridor placed exactly on a known trace scores 0.8798; 3 px scores 0.7045; 5 px scores 0.5325 (measured, `tests/test_metric.py::test_ceiling_of_a_dilated_truth_prediction`). Because the kernel reaches 300 m, a corridor is insurance against positional error; because the false-positive term is linear in area, the same corridor is a cost when the model is right.
* **Why it matters here:** every historical file this repo published emits a thin, thresholded, near-binary mask. The measured optimum on a leakage-free holdout is the only defensible width, and it is not "1 px".
* **Different from prior work:** the sibling project's "emission width decision" was chosen on a proxy catalogue; this is chosen on the spatial trace holdout with the official mask.
* **Expected lift:** medium. **Cost:** very low.
* **Measured:** this is the largest measured effect in the whole project. 0.0943 at 40% emitted area → 0.1324 at 6%, a 40% relative gain; widening beyond 0 px is strictly negative. **This is what the shipped file is optimised for**, and the optimum had never been swept before, in any sibling project, at any time.

### Rank 5 — H7 · Cross-catalogue disagreement layer (deferred, external data)

* **Layers used:** the official stack plus an independent public fault compilation.
* **Signature:** where two independently compiled catalogues disagree — a strand present in one and absent in the other — is a candidate for a *mapping* gap rather than a physical absence.
* **Viability check required before implementing:** the competition page permits external data only if "participants possess a license that permits the data to be used in this challenge and shared with the sponsor for evaluation purposes". Any such compilation must be checked for that licence first, and its coverage of the GeoDAWN footprint must be enumerated, before this is promoted from *listed* to *runnable*. It is not runnable today from what is pinned in `config/data_pins.json`.
* **Expected lift:** unknown. **Cost:** medium-high, with a licensing gate. **Status: parked, not approved.**

---

## Validation protocol (unchanged in spirit, strengthened in 2026-09-27)

1. **Leakage gate first.** Before any model is scored, measure what a prediction that *only reproduces the training catalogue* earns on the held-out faults. It must be near zero or the split is meaningless. `leakage_probe` in `src/gems/heldout.py` does this and the experiment aborts above 0.25.
2. **Hold out space, not pixels.** Contiguous 512 px (51.2 km) blocks, four folds, with a 12 px (1.2 km) collar removed from *both* sides. The collar must exceed the metric's R = 3 px by a wide margin, because of fact (1) above: known faults are free, so without a collar a model can score on a held-out fault simply by drawing its mapped neighbour.
3. **Score with the official metric, with the official mask.** `src/gems/metric.py` is a transcription of the three published equations, checked against the page's own worked example (TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 → 0.60) in `tests/test_metric.py`, and against a literal loop-for-loop reimplementation in the same file. The known-catalogue mask is applied to the false-positive term, per the 2026-09-16 staff statement.
4. **One change at a time.** Every arm uses the same folds, the same model class, the same hyper-parameters, the same sampling budget. A difference is attributable to the thing that changed, or it is noise.
5. **Both numbers reported.** The H1-registered block protocol is recomputed on the same predicted rasters so the 2026-09-27 control (0.149986) and H1 (0.149795) stay comparable, alongside the new leakage-free number. They are different measurements and must not be mixed.
6. **Negative results are recorded.** H1 is in this register precisely because it failed.

## Known limits of every number here

* A spatial holdout measures generalisation to *held-out mapped faults*. The contest scores faults an expert judged to be missing from the map, which may be systematically more subtle than an average mapped trace. A holdout win is **necessary, not sufficient**.
* The exact geometry of the official "known fault" mask is not published — only the statement that known faults are excluded. This project implements the literal reading (the catalogue pixels themselves). If the platform masks a buffered corridor instead, every false-positive number here is slightly pessimistic.
* Nothing in this register is a discovery. A pixel flagged by a coherence or step transform is a *candidate* structure. Calling it a fault before an expert looks at it is exactly the error the competition's own prize process exists to catch.
