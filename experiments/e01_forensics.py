"""Error forensics on the validation fold for BASELINE_BEST (t=0.7)."""
import sys

import polars as pl

from harness import BASELINE, ROOT, Ctx, cache_path, fmt

sys.stdout.reconfigure(encoding="utf-8")
T = 0.7
ctx = Ctx()
best = pl.read_parquet(BASELINE / "train_best.parquet")
lab = ctx.lab
val_ids = ctx.val.select("s1_idx")
n_true = ctx.n_true

# queries relevant to validation: assigned to a val S1 or truly belonging to one
b = best.join(lab, on="q_idx", how="left")
retr = (pl.scan_parquet(str(cache_path("train_cands") / "*.parquet")).select("q_idx", "s1_idx")
        .join(lab.lazy(), left_on=["q_idx", "s1_idx"], right_on=["q_idx", "true_s1"], how="semi")
        .select("q_idx").collect().with_columns(pl.lit(True).alias("retrieved")))
b = b.join(retr, on="q_idx", how="left").with_columns(pl.col("retrieved").fill_null(False))
isval = pl.col("s1_idx").is_in(val_ids["s1_idx"].implode())
tval = pl.col("true_s1").is_in(val_ids["s1_idx"].implode())
b = b.filter(isval | tval.fill_null(False))
b = b.join(n_true.rename({"s1_idx": "_s", "n_true": "pred_s1_ntrue"}), left_on="s1_idx", right_on="_s", how="left") \
     .with_columns(pl.col("pred_s1_ntrue").fill_null(0))
acc = pl.col("p1") >= T
b = b.with_columns(
    pl.when(acc & (pl.col("s1_idx") == pl.col("true_s1"))).then(pl.lit("TP"))
      .when(acc & isval & (pl.col("pred_s1_ntrue") == 0)).then(pl.lit("FP_singleton"))
      .when(acc & isval & pl.col("true_s1").is_null()).then(pl.lit("FP_distractor"))
      .when(acc & isval & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FP_wrong_s1"))
      .when(tval.fill_null(False) & ~pl.col("retrieved")).then(pl.lit("FN_not_retrieved"))
      .when(tval.fill_null(False) & (pl.col("s1_idx") != pl.col("true_s1"))).then(pl.lit("FN_best_wrong"))
      .when(tval.fill_null(False) & ~acc).then(pl.lit("FN_below_threshold"))
      .otherwise(pl.lit("other")).alias("cls"))
print(b.group_by("cls").agg(pl.len().alias("n"), pl.col("p1").median().alias("p1_med")).sort("n", descending=True))
print("\nerrors by p1 bin:")
bins = b.filter(pl.col("cls") != "TP").with_columns(pl.col("p1").cut([0.05, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).alias("bin"))
print(bins.group_by("bin", "cls").len().pivot(on="cls", index="bin").sort("bin"))

# theoretical gain of fixing each class alone
base_pred = best.filter(pl.col("p1") >= T).select("q_idx", "s1_idx")
r0 = ctx.f05(base_pred)
print("\nbaseline", fmt(r0))
cls_q = b.select("q_idx", "cls")
for c in ["FP_singleton", "FP_distractor", "FP_wrong_s1"]:
    drop = cls_q.filter(pl.col("cls") == c)
    r = ctx.f05(base_pred.join(drop, on="q_idx", how="anti"))
    print(f"fix {c:22s} -> F={r['F']:.5f} (+{r['F']-r0['F']:.5f})")
for c in ["FN_below_threshold", "FN_best_wrong", "FN_not_retrieved"]:
    add = b.filter(pl.col("cls") == c).select("q_idx", pl.col("true_s1").alias("s1_idx"))
    r = ctx.f05(pl.concat([base_pred.join(add, on="q_idx", how="anti"), add.with_columns(pl.col("s1_idx").cast(pl.UInt32))]))
    print(f"fix {c:22s} -> F={r['F']:.5f} (+{r['F']-r0['F']:.5f})")

# ---- twin hypothesis: do distractor FPs have look-alike sibling records with the same deviation?
qn = pl.concat([pl.read_parquet(cache_path(f"train_{s}_norm.parquet"), columns=["name_core", "addr_nums", "addr_norm"])
                for s in ("source2", "source3")]).with_row_index("q_idx")
s1n = pl.read_parquet(cache_path("train_source1_norm.parquet"), columns=["name_core", "addr_nums"]).with_row_index("s1_idx")
allb = best.join(qn, on="q_idx").join(s1n.rename({"name_core": "s_name", "addr_nums": "s_nums"}), on="s1_idx")
twin = allb.group_by("s1_idx", "addr_nums").agg(pl.len().alias("n_same_nums"))
allb = allb.join(twin, on=["s1_idx", "addr_nums"])
t2 = b.join(allb.select("q_idx", "addr_nums", "s_nums", "n_same_nums"), on="q_idx")
t2 = t2.with_columns((pl.col("addr_nums") != pl.col("s_nums")).alias("nums_differ_from_s1"))
print("\nTwin statistic (other records with same top-1 S1 AND identical house numbers), when numbers differ from S1:")
print(t2.filter(pl.col("nums_differ_from_s1") & (pl.col("addr_nums") != "")).group_by("cls").agg(
    pl.len(), (pl.col("n_same_nums") > 1).mean().alias("share_with_twin"), pl.col("n_same_nums").mean().alias("mean_twins")).sort("cls"))

# ---- forensic reports
raw_q = pl.concat([pl.read_parquet(cache_path(f"train_{s}.parquet")) for s in ("source2", "source3")]).with_row_index("q_idx")
raw_s = pl.read_parquet(cache_path("train_source1.parquet")).with_row_index("s1_idx")
feats_needed = ["q_idx", "s1_idx", "cos_name", "cos_addr", "rank", "m_cos_name", "m_cos_addr", "nm_tset", "ad_tset"]
fp = b.filter(pl.col("cls").str.starts_with("FP")).sort("p1", descending=True)
fn = b.filter(pl.col("cls").str.starts_with("FN"))
pairs = pl.concat([fp.select("q_idx", "s1_idx"), fn.select("q_idx", "s1_idx")])
F = (pl.scan_parquet(str(cache_path("train_feats") / "*.parquet")).select(feats_needed)
     .join(pairs.lazy(), on=["q_idx", "s1_idx"], how="semi").collect())


def report(df, path, extra_true=False):
    d = (df.join(F, on=["q_idx", "s1_idx"], how="left")
         .join(raw_q.select("q_idx", pl.col("entity_id").alias("candidate_id"), pl.col("business_name").alias("cand_name"),
                            pl.col("business_address").alias("cand_address")), on="q_idx")
         .join(raw_s.select("s1_idx", pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1_name"),
                            pl.col("business_address").alias("s1_address"), "country"), on="s1_idx"))
    if extra_true:
        d = d.join(raw_s.select(pl.col("s1_idx").alias("true_s1"), pl.col("entity_id").alias("true_s1_id"),
                                pl.col("business_name").alias("true_s1_name"), pl.col("business_address").alias("true_s1_address")),
                   on="true_s1", how="left")
    cols = ["cls", "s1_id", "candidate_id", "country", "p1", "p2", "rank", "cos_name", "cos_addr", "nm_tset", "ad_tset",
            "m_cos_name", "m_cos_addr", "s1_name", "cand_name", "s1_address", "cand_address"]
    if extra_true:
        cols += ["true_s1_id", "true_s1_name", "true_s1_address"]
    d = d.with_columns((pl.col("p1") - pl.col("p2")).alias("rank_margin"))
    d.select(cols[:6] + ["rank_margin"] + cols[6:]).write_csv(path, separator="\t")
    print("wrote", path, d.height)


report(fp.head(20000), ROOT / "reports" / "false_positive_analysis.tsv")
report(fn.sample(min(20000, fn.height), seed=1).with_columns(
    pl.when(pl.col("cls") == "FN_not_retrieved").then(pl.lit("A_candidate_missed")).otherwise(pl.lit("B_classifier_rejected")).alias("fn_reason")),
    ROOT / "reports" / "false_negative_analysis.tsv", extra_true=True)
print(fn.with_columns(pl.when(pl.col("cls") == "FN_not_retrieved").then(pl.lit("A_candidate_missed"))
                      .otherwise(pl.lit("B_classifier_rejected")).alias("r")).group_by("r").len())
