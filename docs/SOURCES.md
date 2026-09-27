# Evidence and source register

**Checked on:** 2026-09-27 UTC. Every link below was opened and read in this session; the "used here" column says what the page actually says, not what we wish it said. Official contest definitions, official government data pages, official organiser statements on the competition forum, and team-published records are kept separate on purpose.

## Official contest pages

| ID | Source | What it states, used here | Access / limits |
|---|---|---|---|
| S1 | [Competition hub](https://www.drivendata.org/competitions/306/competition-doe-gems/) | Host page. "A unique feature of this challenge is that there is spatial overlap between the training dataset (the USGS quaternary fault dataset) and the test dataset (newly identified faults within the GeoDAWN region). Participants will be evaluated based on their performance predicting faults contained in a newly labeled, private set of faults." $300,000 in prizes. | Public. Eligibility, team rules and dates must be re-read before any entry. |
| S2 | [Problem description](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) | The task is fault presence. The expert "new faults … will comprise the test dataset for the initial prize round". Feature list, EPSG:32611, 100 m, submission format (single band, float32, [0,1], same bounds, null/NaN outside), the distance-weighted Tversky index with the triangular kernel k(d) = max(1 − d/R, 0), R = 300 m = 3 px, α = 0.2, β = 0.8, and the worked example TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 → 0.60. | Read in two chunks. The organizer's reference repository does **not** contain the scoring code, so `src/gems/metric.py` is transcribed from this page and checked against this page's example. |
| S3 | [About / GeoDAWN](https://www.drivendata.org/competitions/306/competition-doe-gems/page/968/) | USGS/DOE GeoDAWN airborne magnetic and radiometric surveys with coordinated lidar over the north-western Great Basin. | Context, not evidence for any individual raster signature. |
| S4 | [Competition data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) | The route for the official rasters. | Checked 2026-09-27: unauthenticated requests are redirected to DrivenData login. **No file was downloaded from DrivenData in this workspace.** |
| S5 | [Official rules PDF](https://docs.nlr.gov/docs/fy26osti/96647.pdf) | U.S. DOE prize rules: structure, eligibility, entry and evaluation phases. | Read for the prize mechanics only. The live competition site is authoritative for dates. |
| S6 | [Organizer reference solution](https://github.com/drivendataorg/gems-prize-reference-solution) | Official organizer repository (clone `aebe92f7`, 2026-06-16). A `segmentation_models_pytorch` U-Net/ResNet18 with `TverskyLoss(alpha=0.2, beta=0.8, mode="binary")`, 128 px patches, 5 Monte-Carlo splits. **It contains no implementation of the contest's distance-weighted scoring metric** — the loss is a training loss, not the evaluator. | Confirmed by reading the notebook source. This is why the metric here is written from S2, not borrowed. |
| S13 | [Public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) | Read 2026-09-27: DARD 0.3049 (rank 1, 10 submissions), alexoktaba 0.2993, HardcoreTechGod 0.2854 … **extradr19 0.1563 (rank 24, 2 submissions)**. The column heading is "Best public DW-Tversky". | A dated snapshot. Public scores change and are not the private or final-round score. `scripts/fetch_leaderboard.py` refreshes `docs/leaderboard.json`; it only reads the public table. |

## Official organiser statements on the competition forum

These are statements by **chrisk-dd, badged "DrivenData Staff"**, on the GEMS category forum. They are not on the problem page, which is why they are registered separately and quoted verbatim.

| ID | Source | Verbatim statement | Why it changed the work |
|---|---|---|---|
| **S18** | [Scoring clarification: are known USGS/INGENIOUS faults masked when scoring?](https://community.drivendata.org/t/scoring-clarification-are-known-usgs-ingenious-faults-masked-when-scoring-and-are-they-in-the-final-round-label-set/11516) (2026-09-16) | "1. Pixels corresponding to known USGS/INGENIOUS faults are masked / excluded from evaluation, so they do not count towards penalty terms. 2. Re-evaluation will also mask/exclude the existing USGS/INGENIOUS faults. We'll consider changing the description, but for scoring purposes it should not matter whether these known faults are included with predictions or not." | Predicting the supplied catalogue is **free**, in both rounds. Implemented as `fp_exclude` in `gems.metric.distance_weighted_tversky`. It is also why a random-trace holdout leaks: see S19's note in [Hypotheses](HYPOTHESES.md) and `leakage_probe`. |
| **S19** | [Where do you draw the line?](https://community.drivendata.org/t/where-do-you-draw-the-line/11536) (2026-09-22) | "For the purposes of this competition, 'new fault' means 'any fault pixel not already captured by USGS/INGENIOUS' and can include newly mapped geometry of an existing fault system." | The target is explicitly not "virgin ground". Unmapped continuations of mapped systems are in scope. This is the geological basis of hypothesis H4. |
| **S20** | [How were the new test faults identified?](https://community.drivendata.org/t/how-were-the-new-test-faults-identified-data-sources-and-fault-types/11527) (2026-09-24) | "We're not sharing details about the data sources, fault types, or coverage behind the test faults beyond what's in the problem description. Note that the largest prize pool (Phase 2) will use a test set that is updated by expert review of all Phase 1 submissions, so your fault predictions have an impact on final evaluation even if they are not the most performant in Phase 1." | The organisers **decline** to say whether the labels came from lidar, DEM, magnetics or field mapping. Any claim in this project about which evidence the experts used would be a hallucination. Recorded so nobody repeats it. It also confirms the Phase 2 dependency on every submission. |
| S21 | [About the GEMS Prize Challenge category](https://community.drivendata.org/t/about-the-gems-prize-challenge-category/11499) | Category scope only. | No technical content. |

**Not verified, and therefore not used:** a sibling project's notes refer to "Hermant et al. 2025, cited by the organisers" for the claim that catalogue traces can sit ~400 m from lidar-mapped faults. That paper was not opened and is not cited anywhere in this repository. If it matters, read it first.

## Official government data

| ID | Source | What it states | Limits |
|---|---|---|---|
| S7 | [USGS GeoDAWN data release](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) | The GeoDAWN airborne magnetic/radiometric survey; DOI 10.5066/P93LGLVQ; work marked **CC0 1.0 Universal**. | CC0 is compatible with the competition's external-data clause (S2). Individual product coverage must still be checked before use. |
| S8 | [USGS ScienceBase item 657e1d85d34e23d3533209f7](https://www.sciencebase.gov/catalog/item/657e1d85d34e23d3533209f7) | The GeoDAWN study-area record; publication date 2024-03-01. | Catalogue record. |
| S9 | [DOE OSTI 2345165](https://www.osti.gov/biblio/2345165) | 2023 Geothermics paper on Play Fairway Analysis integration of geologic/geophysical evidence. | Methodological context only. |
| S10 | [NREL FY23OSTI 86139](https://docs.nrel.gov/docs/fy23osti/86139.pdf) | Geothermal Play Fairway Analysis best practices, 2023. | Terminology context. |
| S11 | [DOE Data Explorer record 1493758](https://www.osti.gov/dataexplorer/biblio/dataset/1493758) | 2017 Eastern Great Basin Play Fairway final report record. | **Not** asserted to be inside GeoDAWN coverage. |

## Team-published records (not official)

| ID | Source | How it is used | Limits |
|---|---|---|---|
| S12 | Sibling repositories under `buffedlizard55-lab` (GEMSDOE, 5GEMSDOE, 7GEMSDOE, 8GEMSDOE, GEMSDOE2, GEMSDOE3) | The published candidate rasters are downloaded read-only and re-scored with our own verified metric in `docs/historical/diagnosis.json`. | A public repository is not an authenticated DrivenData upload receipt. Reported scores are the team's own transcriptions. |
| S14 | [5GEMSDOE artifact](https://github.com/buffedlizard55-lab/5GEMSDOE/blob/main/data/evidence/leaderboard_anchor/gemsdoe-ens12-adopted-7f00890a.tif), [GEMSDOE artifact](https://github.com/buffedlizard55-lab/GEMSDOE/blob/main/data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif), [7GEMSDOE copy](https://github.com/buffedlizard55-lab/7GEMSDOE/blob/main/external/scored/gemsdoe1-ens12-7f00890a.tif) | All three are the same TIFF, SHA-256 `7f00890a…`, 570,890 bytes. Re-verified in this session. | Proves file reuse; does not prove which bytes were uploaded. |
| S15 | [5GEMSDOE bridge manifest](https://github.com/buffedlizard55-lab/5GEMSDOE/blob/6e8d28ba407b5d748de5b3e94635c41d95e35354/data/bridge/manifest.json) | The transport route used by `scripts/download_competition_data.sh` when the official Dropbox mirrors are unreachable. Pins are copied into `config/data_pins.json`. | **Not an official publisher endpoint and not a direct re-download from DrivenData.** This is the weakest link in the data lineage and is the first thing to repair when credentials exist. |
| S16 | [8GEMSDOE candidate](https://github.com/buffedlizard55-lab/8GEMSDOE/blob/main/docs/downloads/submission.tif) | Different bytes *and* different pixels from S14, at the same displayed 0.1563. | Score attribution is user-reported. |

## Data actually used in this repository

| File | Bytes | SHA-256 | Role |
|---|---|---|---|
| `data/training_features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` | The 19 official feature bands. |
| `data/labels.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` | The supplied USGS/INGENIOUS catalogue, 60,988 positive pixels. |
| `data/sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` | The official template, and the grid every submission must match. |
| `data/external/topo_u8.tif` | 32,523,329 | `a6398d9950965dec6aae6ccecdaa6ced48645d133eab222cbdd11def9bdabfa4` | 9-band topographic auxiliary, built by the team from public 3DEP. |
| `data/external/radiometric_u8.tif` | 25,475,158 | `6cb051f70f941fd78028fe66a9f71e87204fcd8d9a85903df0b94993bad1ec4d` | 7-band airborne-gamma auxiliary. |

All five are verified by `scripts/prepare_data.py`, which exits non-zero on any hash or structural mismatch. `data/` is git-ignored; nothing large or competition-derived is committed.

**Finding worth recording:** the official `example_submission.tif` is *not* "a raster that predicts total fault absence" as the problem page describes. Measured, it is the supplied catalogue rasterised to float32 — 1.0 on all 60,988 catalogue fault pixels, 0.0 elsewhere in the footprint, NaN outside, and bit-for-bit equal to `(labels.tif > 0)` as float32. A competitor who submitted the template unchanged would have predicted exactly the faults the private set is defined to exclude, and would have scored zero. Nobody should copy it.

## Citation discipline

* A candidate pixel is a hypothesis, not a discovery, until an expert looks at it.
* A mechanism that is plausible geology is not evidence that a particular raster signature detects a fault. The two are kept in separate columns throughout this repository.
* Band positions are read from the file, never inferred from prose order. `scripts/prepare_data.py` enforces this.
* Every recorded number has a date, a command that reproduces it, and the code hash that produced it.
* A statement attributed to a named person is quoted verbatim with a link. If the link was not opened, the statement is not used.
