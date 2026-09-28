# v4 progress log (autonomous session, 27 Sep 2026) — read first after any crash

Goal: improve LEADERBOARD macro F0.5 (C4 = 0.970441 on LB; local DEV 0.98322) without overfitting.
Preserved: output/matching_results.tsv = C4 (sha 5a7f2c06…), output/v3_roles = v3 (sha fa607c39…, not LB-scored).
Rules: heavy jobs strictly sequential; resumable; retry wrappers; ~32 GB disk free -> keep new artifacts small;
no sibling/population features; select only on DEV / proxy-transfer tests; never tune on CONF or leaderboard.

## Log
- [01:20] Diagnosis: US/India test ~ DEV + more distractors (confidently rejected). France has 2x the uncertain band (7.9% vs 3.4-4.1% DEV). If US/India test ~0.982, LB 0.9704 implies France ~0.90.
- [01:20] Unsupervised noise mining (high-confidence test matches): French S1 addresses carry REGION, records carry DEPARTMENT; bis/b; crs; legal cie/compagnie/ei; "S.A".
- [01:20] fr_locale.py: France-only normalization (whole-component region/department -> region code for all 13 regions + metropolitan departments; bis/ter; crs; cie/compagnie/ei; dotted abbreviations). US/India untouched.
- [01:15] v4test chain (France re-normalized, re-retrieved, re-featurized, C3+C4 re-scored) DONE, no failures. US/India predictions byte-identical to v2test (100.000%).
- [01:18] France: accept 61.3%->62.0%, uncertain 7.9%->7.2%; flips +19,529 accepted / -9,613 rejected. Manual read of 28 flips: ~11/14 accepts plausible (same address, French legal noise), ~10/14 rejects plausible (shifted house numbers). Judgement, not a label-based score.
- [01:20] C4+FR bundle: cache/runs/submission_C4_stack_on_C3_direct_x_more_data_v4test/ matching sha e04367f5..., 5,768,013 matches, validator PASS, candidates streamed-subset PASS.
- [01:42] TRANSFER PROXY (train one country -> evaluate other country's DEV): US->India base 0.92951 (t0.8) vs in-country C3 India 0.98100. Unseen-country loss ~0.05 — consistent with France ~0.90 on LB.
- [01:42] monotone+regularized (127 leaves, min_leaf 400, L2 10): US->India 0.92560 -> REJECTED (worse).
- [01:50] US->India errors: FN below threshold 47k (C3 11k), FP on unmatched 22.5k (C3 3.2k); errors broad (Latin-script India 10.9% vs 3.9%), not just Indic.
- [01:55] Next: per-country quantile normalization (unlabeled, transductive) and no-scale feature variants on the same proxy; v3+FR chain running first.
- [02:11-02:31] More transfer variants (US->India, best threshold): per-country quantile normalization 0.91702 (REJECTED), no scale features 0.92336 (REJECTED), both 0.92587 (REJECTED). Plain full-feature model transfers best (0.92951).
- [02:40] v3+FR: frozen v3 role model on v4test via v3fr.py (no v3 script edited; module loaded by path after a name-shadowing fix). output/v3_roles_fr/ matching sha 0e9951d5..., 5,727,529 matches, validator PASS, streamed subset PASS.
- [02:43] Identity check: US/India rows 100% identical between v3 vs v3+FR and C4 vs C4+FR; France rows differ ~9-10%.
- [02:45] Queued: India->US transfer (threshold behaviour for unseen country), then self-training (pseudo-label) proxy US+India-pseudo -> India DEV.
- [03:16] SELF-TRAINING proxy (US labelled + 10% India non-DEV pseudo-labels, hi .98 / lo .02): India DEV 0.93095/0.93300/0.93441 at t .7/.8/.9 vs source-only 0.92751/0.92951/0.92749 -> +0.0035..+0.0069 at every threshold. India->US base also prefers t .8-.9 (0.94227 @.7 -> 0.94477 @.8 -> 0.94549 @.9).
- [03:22] Pseudo 40% proxy: watchdog abort (15M rows too big). Kept 10% result.
- [03:34] FR self-training (fr_pseudo.py fr1): labelled 6.19M (US+India TRAIN 10%) + 2.21M France pseudo rows (166k positive; hi .98/lo .02 from C3 on v4test, 20% of France records). 750 trees. France uncertain band 7.2% -> 4.3% (DEV-like); accept@.8 62.3%.
- [03:40] Read 24 disagreements vs C4+FR@.8: self-trained right on ~8-9/12 of its extra accepts (same address, French legal noise) and ~7/12 of its extra rejects (different numbers / even different city). Judgement only.
- [03:45] ASSEMBLED (all validator PASS, streamed-subset PASS, US/India rows 100% identical to source model):
  output/v4_c4fr_t80 (C4 + FR-locale, France t=.8), output/v4_c4fr_frp_t80 (C4 US/India + self-trained France t=.8) <- RECOMMENDED,
  output/v4_v3fr_frp_t80 (v3 US/India t=.8 + self-trained France t=.8).
- [03:51] Self-training round 2 (US->India): 0.93108 best < round 1 0.93441 -> REJECTED (confirmation bias). One round only.
- [04:08] Self-training India->US: 0.94107/0.94313/0.94525 at t .7/.8/.9 vs base 0.94227/0.94477/0.94549 -> neutral-to-slightly-negative in this direction.
  Fair comparison (one common threshold per method, averaged over both directions): base best avg 0.93714 (t .8); self-trained best avg 0.93983 (t .9) -> +0.0027,
  and self-trained@.9 is within 0.0002 of base's best in the unfavourable direction. => France self-trained variant uses t=0.9.
- [04:25] Final: recommended output/v4_c4fr_frp_t90 (C4 US/India identical to LB 0.970441 file; France FR-locale + self-trained, t .9); conservative alt output/v4_c4fr_t80. REPORT.md written.
- [04:40] Packaged: business_entity_resolution/src/v4/ (v4 scripts, path-fixed), README v4 section, Documentation addendum, make_submission_v4.py; built team_submission_v4_c4fr_frp_t90.zip (588 MB, 44 files, testzip OK; not committed).
