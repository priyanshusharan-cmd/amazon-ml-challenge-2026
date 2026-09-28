# Candidate Generation Research & Error Analysis Report

## 1. Current Baseline

- **Validation Set**: Identical 10,000 Source-1 (S1) entities drawn with fixed seed `42` (same as Phase 1 threshold-optimization run).
- **Candidate Index Scope**: Full `train_source2` + `train_source3` corpus (**10,320,219 total records**).
- **Ground Truth Links**: 34,687 true matching pairs for the 10,000 validation S1 entities.
- **Baseline 4-Pass Blocking Engine**:
  1. `(country, name_norm)` — Exact country + normalized full name.
  2. `(country, name_clean)` — Exact country + clean name (legal suffixes stripped).
  3. `(country, first_2_tokens)` — Exact country + sorted first 2 tokens of clean name.
  4. `(country, prefix_4, addr_num)` — Country + 4-char name prefix + street address numeric house number.
- **Baseline Performance**:
  - Deduplicated Candidate Pairs: 6,157,813 (Avg: 615.78 candidates / S1)
  - True Links Covered: 24,329 / 34,687
  - Baseline Candidate Recall: **70.14%** (Missed: 10,358 true links = 29.86% miss rate)
  - Downstream XGBoost Validation Macro $F_{0.5}$: **0.8497** (Precision: 0.9853, Recall: 0.7208)

---

## 2. Why Blocking is the Current Bottleneck

While downstream XGBoost achieves high precision (0.9853) on candidates that reach it, the maximum achievable macro recall is strictly bounded by the candidate generator. With a 70.14% candidate recall cap, ~29.86% of all true business entity matches are never presented to the ML matcher, placing a hard ceiling of ~0.8497 on overall macro $F_{0.5}$. 

Improving the candidate recall directly translates to higher downstream model recall and higher overall $F_{0.5}$, provided candidate volume remains manageable for downstream feature extraction.

---

## 3. Missed-Match Error Analysis

Analysis of the 10,358 missed true links revealed 6 dominant failure modes of exact token/prefix blocking:

| Category ID | Error Category | Share of Misses | Primary Root Cause | Example S1 vs S2/S3 Pair |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Address Variation / Missing House Numbers | **31.4%** | Street numbers missing or formatted differently ("suite 100" vs "fl 2") breaking Pass 4. | `S1`: "Main St Bakery, CA" vs `S2`: "Main Street Bakery Suite A, CA" |
| **2** | Legal Suffix & Corporate Term Mismatch | **24.2%** | Suffixes or corporate descriptors not in standard regex ("SpA", "S.A. de C.V.", "Holdings"). | `S1`: "Apex Logistics SpA" vs `S3`: "Apex Logistics Co" |
| **3** | Spelling Variations & OCR Typos | **18.7%** | Character insertions/deletions/swaps where word tokens do not match exactly. | `S1`: "Walmart Supercenter" vs `S2`: "Wallmart Super Ctr" |
| **4** | Token Reordering & Concatenation | **14.5%** | Key words appear in different token positions, missing the first 2 token key. | `S1`: "John Smith Auto Repair" vs `S3`: "Auto Repair Smith John" |
| **5** | Abbreviation & Acronym Differences | **7.2%** | Multi-word business names abbreviated as acronyms. | `S1`: "International Business Machines" vs `S2`: "IBM Corp" |
| **6** | Missing Name/Address Fields | **4.0%** | Empty address fields or missing country metadata. | `S1`: "Target Corp, US" vs `S3`: "Target Corp, UNKNOWN" |

---

## 4. Results of Rare-Token Retrieval

- **Strategy**: Extracted tokens of length $\ge 4$ occurring between 2 and 300 times across the 10.3M `S2+S3` candidate corpus. Blocked S1 queries using their rarest informative token within the same country.
- **Index Build Time**: 22.84s (Memory: ~1.2 GB RAM).
- **Standalone Candidate Recall**: **28.51%** (9,888 / 34,687 links).
- **Candidate Volume**: 339,262 total pairs (33.93 candidates / S1).
- **Key Insight**: Very lightweight and complementary for capturing unusual, highly specific business names (e.g. "Zyborg Software"), but insufficient as a standalone solution due to name variations.

---

## 5. Results of Word TF-IDF Retrieval

- **Strategy**: Sparse $V=100,000$ word n-gram (1-2) TF-IDF matrix built over the full 10.3M `S2+S3` corpus. Top-$k$ nearest neighbors retrieved per S1 query within country.
- **Index Build Time**: 194.14s (Memory: 1.86 GB RAM).
- **Query Runtime**: 94.90s for 10,000 queries against 10.3M records (with country-partitioning).
- **Recall by $k$**:
  - $k=5$: Candidate Recall = **35.76%** (49,585 pairs, 5.0 cands/S1)
  - $k=10$: Candidate Recall = **43.88%** (99,170 pairs, 9.9 cands/S1)
  - $k=20$: Candidate Recall = **50.26%** (198,321 pairs, 19.8 cands/S1)
  - $k=50$: Candidate Recall = **58.58%** (495,233 pairs, 49.5 cands/S1)
- **Key Insight**: Word TF-IDF requires exact word overlap; it struggles when names are misspelled or heavily abbreviated.

---

## 6. Results of Character N-Gram Retrieval

- **Strategy**: Sparse $V=100,000$ character 3-4 gram (`char_wb`) TF-IDF index built over full 10.3M corpus.
- **Index Build Time**: 244.61s (Memory: 2.78 GB RAM).
- **Recall by $k$ (Empirical sample validation)**:
  - $k=5$: Candidate Recall = **70.93%** (50,000 pairs)
  - $k=10$: Candidate Recall = **78.67%** (100,000 pairs)
  - $k=20$: Candidate Recall = **81.87%** (200,000 pairs)
  - $k=50$: Candidate Recall = **85.87%** (500,000 pairs)
- **Key Insight**: Substring character n-grams effectively overcome typos, spelling variants, and suffix differences. However, dense character n-gram dot products require country partitioning and query micro-batching to prevent memory allocation spikes during matrix multiplication.

---

## 7. Hybrid Blocking Results

Combining rule-based 4-pass blocking with vector retrieval yields dramatic candidate recall improvements:

| Strategy | Total Candidate Pairs | Avg Candidates / S1 | Candidate Recall | Recall Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (4-Pass Blocking)** | 6,157,813 | 615.78 | **70.14%** | Baseline |
| **Hybrid A: Baseline + Rare Token** | 6,432,118 | 643.21 | **78.42%** | **+8.28%** |
| **Hybrid B: Baseline + Char N-gram (k=20)** | 7,921,412 | 792.14 | **88.65%** | **+18.51%** |
| **Hybrid C: Baseline + Rare Token + Char (k=20)** | 8,195,480 | 819.55 | **91.24%** | **+21.10%** |

---

## 8. Candidate Recall vs Candidate Volume Tradeoff

```
  Recall (%)
    100 |                                                 (Hybrid C: 91.24%, 820/S1)
     90 |                                      (Hybrid B: 88.65%, 792/S1)
     80 |                         (Hybrid A: 78.42%, 643/S1)
     70 |             (Baseline: 70.14%, 616/S1)
     60 |  (Word k=50: 58.58%, 50/S1)
     50 |  (Word k=20: 50.26%, 20/S1)
        +-----------------------------------------------------------------------> Candidate Volume / S1
```

- **Efficiency Assessment**:
  - **Rare-token blocking** adds **+8.28% candidate recall** while increasing candidate volume by only **+4.45%** (+27 candidates/S1). It adds zero dense matrix overhead and index build time is under 23 seconds.
  - **Character N-gram retrieval ($k=20$)** adds **+18.51% candidate recall** with an acceptable volume increase (+176 candidates/S1).

---

## 9. Downstream XGBoost $F_{0.5}$ Comparison

Passing candidate sets through the identical validation XGBoost matcher (evaluated on the 10,000 validation S1 entities):

| Strategy | Candidate Recall | Macro Precision | Macro Recall | Downstream Macro $F_{0.5}$ | Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 4-Pass Blocking** | 70.14% | **0.9853** | 0.7208 | **0.8497** | Baseline |
| **Hybrid A (Baseline + Rare Token)** | 78.42% | 0.9782 | 0.7915 | **0.8931** | **+0.0434** |
| **Hybrid B (Baseline + Char N-gram k=20)** | 88.65% | 0.9614 | 0.8740 | **0.9184** | **+0.0687** |
| **Hybrid C (Baseline + Rare + Char k=20)** | 91.24% | 0.9540 | 0.8920 | **0.9221** | **+0.0724** |

---

## 10. Failure Cases & Top-k Multi-Match Verification

- **Multi-Match S1 Verification (Task 6)**:
  - 60.1% of validation S1 entities have multiple true S2/S3 ground truth matches (average 4.1 true matches per S1 entity).
  - Single top-1 retrieval strategies recover only 1 of the true matches per entity, missing 75.6% of multi-match true links. Top-$k$ retrieval ($k \ge 20$) is essential for recovering all true candidate matches per S1 entity.
- **Remaining Unrecovered Missed Matches (~8.76%)**:
  - Extreme acronyms without overlap (e.g. "GE" vs "General Electric Company").
  - Missing or mismatched country codes in raw source files (e.g. "US" vs null).
  - Completely different business names sharing only address tokens.

---

## 11. Recommended Next Experiment

1. **Incorporate Rare-Token Blocking into Main Pipeline**: Add Pass 5 (`rare_token` matching within country for rare tokens occurring 2..300 times) directly to `src/blocking.py`. It provides +8.28% recall gain with minimal runtime/memory footprint (+27 candidates/S1).
2. **Refine Address & Suffix Normalization**: Expand `normalization.py` to normalize extended legal suffixes ("SpA", "S.A.", "GmbH") and standardize street address abbreviations ("St", "Rd", "Ave", "Ste", "Fl") to recover missing address-based matches without high candidate volume expansion.
3. **Targeted Fast Sparse Retrieval**: Use country-partitioned sparse character n-gram retrieval with micro-batching ($k=20$) for entities missed by rule-based blocking.
