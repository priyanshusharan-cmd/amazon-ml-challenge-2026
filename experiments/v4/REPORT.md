# v4 report — attacking the leaderboard gap (France)

## Starting point
* C4 (v2): local DEV 0.98322, sealed CONF 0.98321, **leaderboard 0.970441**. v3 (roles): local +0.001, leaderboard not measured.
* The gap is not explained by overfitting (DEV ≈ CONF). The question was which part of the test set is worse than local validation.

## Diagnosis (no test labels used)
1. **US/India test look like DEV plus more distractors**, which the model rejects confidently. The share of records in the uncertain band [0.3, 0.97) is similar to DEV: India 5.3% vs 4.1%, US 3.8% vs 3.4%.
2. **France has twice the uncertain band (7.9%).** France is 15% of test S1 entities and has no training data. If US/India test ≈ 0.982, a leaderboard score of 0.9704 implies France ≈ 0.90.
3. **Cross-country transfer proxy (labelled):** a model trained on US only scores **0.9295** on India DEV, versus 0.9810 when trained with Indian data. India-only → US: 0.9455 vs 0.9827. An unseen country costs 0.04–0.05, which is consistent with (2).

## What was tried on the proxy (train one country → evaluate the other country's DEV)
| variant | US→India best F0.5 | verdict |
|---|---|---|
| plain C3 recipe | 0.92951 (t .8) | reference |
| monotone constraints + stronger regularisation | 0.92560 | rejected |
| per-country quantile-normalised features | 0.91702 | rejected |
| drop scale-dependent features | 0.92336 | rejected |
| quantile + drop scale | 0.92587 | rejected |
| **self-training, 1 round** (10% of target non-DEV records pseudo-labelled at p ≥ .98 / ≤ .02) | 0.93441 (t .9); +0.0035…+0.0069 at every threshold | see both directions below |
| self-training, 2 rounds | 0.93108 | rejected (confirmation bias) |

Both directions, F0.5 on the unseen country's DEV:

| | t .7 | t .8 | t .9 |
|---|---|---|---|
| US→India base | 0.92751 | **0.92951** | 0.92749 |
| US→India self-trained | 0.93095 | 0.93300 | **0.93441** |
| India→US base | 0.94227 | 0.94477 | **0.94549** |
| India→US self-trained | 0.94107 | 0.94313 | 0.94525 |

Choosing one common threshold per method (average of both directions): base 0.93714 at t .8, **self-trained 0.93983 at t .9 (+0.0027)**.
Self-training clearly helps one direction and is neutral in the other (−0.0002 vs base's best), so the evidence is positive but not overwhelming.
An unseen country prefers a higher threshold than in-country (.8–.9 vs .7) in every setting tested.

## Changes applied to France only (US/India predictions are byte-identical to C4)
1. **French-locale normalisation** (`fr_locale.py`). Noise vocabulary was mined without labels from high-confidence test matches:
   * whole address components that are a French region or department map to one region code. S1 records carry "Hauts-de-France" while S2/S3 records carry "Nord": the analogue of US state names vs codes, which the base normaliser already handles;
   * bis/ter → b/t, CRS → cours, rond-point variants, apartment tokens;
   * legal forms `cie`/`compagnie`/`ei` are removed from the name core; "S.A" with no final dot is handled.
   France was then re-retrieved and re-featurised (split `v4test`). US/India artefacts are hard-linked, and their predictions are verified identical.
2. **Self-training for France** (`fr_pseudo.py`): 6.19M labelled US+India TRAIN rows plus 2.21M France pseudo rows (from C3's own confident predictions, 20% of France records). The model is used only for France records. The France uncertain band fell from 7.2% to 4.3%, which is DEV-like.
3. **France threshold**: 0.8 for the normalisation-only variant, 0.9 for the self-trained variant (the thresholds each method prefers across both proxy directions).

Manual reading of flipped French decisions (judgement, not a score): most new accepts share the same address and differ only in French legal/filler words (`Jules & Compagnie` vs `Jules & Cie`). Most new rejects have shifted house numbers or a different city.

## Candidate files (all: official validator PASS, matches ⊆ candidates, one S1 per record, 1,732,544 rows)
| path | US/India | France | status |
|---|---|---|---|
| `output/matching_results.tsv` | C4 | C4 | leaderboard **0.970441** |
| **`output/v4_c4fr_frp_t90/`** | **C4 (identical)** | **FR-locale + self-trained, t .9** | **recommended first upload** |
| `output/v4_c4fr_t80/` | C4 (identical) | FR-locale only, C4, t .8 | conservative alternative (no self-training) |
| `output/v4_c4fr_frp_t80/`, `output/v4_v3fr_frp_t80/` | C4 / v3 | self-trained, t .8 | superseded by t .9 |
| `output/v3_roles/`, `output/v3_roles_fr/` | v3 | v3 (± FR-locale) | v3 US/India part not leaderboard-tested |

US/India rows of every C4-based v4 file are byte-identical to the 0.970441 file, so a leaderboard change isolates the France effect.
Not applied: under test-like distractor density (x2 on DEV) C4's in-country optimum moves from t .7 to .8, but only by +0.0003. US/India were kept unchanged to preserve that isolation.

## Honest limits
* France accuracy is still **unmeasured**. The proxy (one labelled country → another) is the best available evidence but not France itself.
* Expected effect is modest per step. If France ≈ 0.90, each +0.01 on France is +0.0015 overall.
* The v3 US/India part and all France changes need the leaderboard to confirm. Upload `v4_c4fr_frp_t80` first: US/India are identical to the 0.970441 file, so any score change is due to France.
