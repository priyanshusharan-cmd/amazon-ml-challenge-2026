"""Forensics step 3: reports/false_positive_analysis.tsv, reports/false_negative_analysis.tsv,
and the 'twin' test (do distractors come with look-alike siblings carrying the same deviation?)."""
import polars as pl

from harness import EXP_DIR, ROOT, cache_path, mem

qt = pl.read_parquet(EXP_DIR / "qtable.parquet")
cls = pl.read_parquet(EXP_DIR / "val_classes.parquet")
b = qt.join(cls, on="q_idx")
fp = b.filter(pl.col("cls").str.starts_with("FP")).sort("p1", descending=True).head(15000)
fn = b.filter(pl.col("cls").str.starts_with("FN")).sample(15000, seed=1)
need = pl.concat([fp, fn])
qids = need.select("q_idx").unique()
sids = pl.concat([need.select("s1_idx"), need.select(pl.col("true_s1").alias("s1_idx"))]).drop_nulls().unique()

raw_q = (pl.concat([pl.scan_parquet(cache_path(f"train_{s}.parquet")) for s in ("source2", "source3")]).with_row_index("q_idx")
         .join(qids.lazy().with_columns(pl.col("q_idx").cast(pl.UInt32)), on="q_idx", how="semi").collect())
raw_s = (pl.scan_parquet(cache_path("train_source1.parquet")).with_row_index("s1_idx")
         .join(sids.lazy().with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="semi").collect())
print("raw loaded", mem(), flush=True)
fcols = ["q_idx", "s1_idx", "rank", "cos_name", "cos_addr", "nm_tset", "ad_tset", "num_conflict", "num_q_only", "q_addr_missing"]
pairs = need.select("q_idx", "s1_idx")
feats = []
for f in sorted(cache_path("train_feats").glob("*.parquet")):
    feats.append(pl.read_parquet(f, columns=fcols).join(pairs, on=["q_idx", "s1_idx"], how="semi"))
F = pl.concat(feats)
print("feats loaded", F.height, mem(), flush=True)


def build(df, true_cols):
    d = (df.join(F, on=["q_idx", "s1_idx"], how="left")
         .join(raw_q.select("q_idx", pl.col("entity_id").alias("candidate_id"), pl.col("business_name").alias("cand_name"),
                            pl.col("business_address").alias("cand_address")), on="q_idx")
         .join(raw_s.select("s1_idx", pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1_name"),
                            pl.col("business_address").alias("s1_address"), "country"), on="s1_idx", how="left")
         .with_columns((pl.col("p1") - pl.col("p2")).alias("rank_margin"),
                       (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("true_label")))
    if true_cols:
        d = d.join(raw_s.select(pl.col("s1_idx").alias("true_s1"), pl.col("entity_id").alias("true_s1_id"),
                                pl.col("business_name").alias("true_s1_name"), pl.col("business_address").alias("true_s1_address")),
                   on="true_s1", how="left")
    return d


FP = build(fp, False)
FP = FP.with_columns(  # reason clusters (rule-based tags from the features)
    pl.when(pl.col("q_addr_missing") == 1).then(pl.lit("no_address_name_only"))
      .when(pl.col("num_conflict") == 1).then(pl.lit("numeric_mismatch"))
      .when(pl.col("num_q_only") > 0).then(pl.lit("extra_or_changed_number"))
      .when(pl.col("nm_tset") < 90).then(pl.lit("name_word_changed"))
      .when(pl.col("ad_tset") < 90).then(pl.lit("address_token_changed"))
      .otherwise(pl.lit("near_identical")).alias("reason"))
FP.select("cls", "reason", "s1_id", "candidate_id", "true_label", "country", pl.col("p1").alias("model_score"), "rank", "rank_margin",
          pl.col("cos_name").alias("name_sim_tfidf"), pl.col("nm_tset").alias("name_sim_tokenset"),
          pl.col("cos_addr").alias("addr_sim_tfidf"), pl.col("ad_tset").alias("addr_sim_tokenset"),
          "s1_name", "cand_name", "s1_address", "cand_address").write_csv(ROOT / "reports" / "false_positive_analysis.tsv", separator="\t")
print("\nFP reasons (top-15000 by confidence):")
print(FP.group_by("cls", "reason").len().sort("len", descending=True))

FN = build(fn, True).with_columns(
    pl.when(pl.col("cls") == "FN_A_not_retrieved").then(pl.lit("A_candidate_generator_missed"))
      .otherwise(pl.lit("B_candidate_present_classifier_rejected")).alias("fn_reason"))
FN.select("fn_reason", "cls", "s1_id", "candidate_id", "true_s1_id", "country", pl.col("p1").alias("model_score_best"), "rank_margin",
          "q_addr_missing", "cand_name", "cand_address", pl.col("s1_name").alias("predicted_s1_name"),
          pl.col("s1_address").alias("predicted_s1_address"), "true_s1_name", "true_s1_address"
          ).write_csv(ROOT / "reports" / "false_negative_analysis.tsv", separator="\t")
allfn = b.filter(pl.col("cls").str.starts_with("FN"))
print("\nFN reason (all validation FNs):")
print(allfn.with_columns(pl.col("cls").str.slice(0, 4).alias("AB")).group_by("AB").len())
print("FN with empty candidate address, by class:")
print(FN.group_by("cls").agg(pl.len(), pl.col("q_addr_missing").mean().alias("share_no_address")))
print(mem())
