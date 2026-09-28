"""E10: number-conflict guard. For records whose house numbers conflict with the S1 (num_conflict=1),
stage 2/3 may not override stage 1: accepted only if stage-1 p1 >= T1. Variants:
 G1: all conflict records;  G2: conflict records whose conflicting numbers are shared by a sibling (cluster)."""
import polars as pl
from harness import EXP_DIR, Ctx
b2 = 0.25
def expf(P, floor=0.5):
    d = P.filter(pl.col("prob") >= 0.02).sort(["s1_idx", "prob"], descending=[False, True]).with_columns(
        pl.col("prob").cum_sum().over("s1_idx").alias("S"), pl.int_range(1, pl.len() + 1).over("s1_idx").alias("m"),
        pl.col("prob").sum().over("s1_idx").alias("T"), (1 - pl.col("prob")).clip(1e-9).log().sum().over("s1_idx").exp().alias("p_none"))
    d = d.with_columns(((1 + b2) * pl.col("S") / (b2 * pl.col("T") + pl.col("m"))).alias("ef"))
    bm = d.group_by("s1_idx").agg(pl.col("ef").max().alias("efmax"), pl.col("m").get(pl.col("ef").arg_max()).alias("mstar"), pl.col("p_none").first())
    return d.join(bm, on="s1_idx").filter((pl.col("efmax") > pl.col("p_none")) & (pl.col("m") <= pl.col("mstar")) & (pl.col("prob") >= floor))
def load(split, predfile):
    P = pl.read_parquet(predfile).select("q_idx", "s1_idx", "prob")
    x = pl.concat([pl.read_parquet(f, columns=["q_idx", "p1", "num_conflict", "sup_nums_all"]) for f in sorted((EXP_DIR / f"s2_{split}").glob("*.parquet"))])
    return P.join(x, on="q_idx")
def variants(P):
    out = {"G0 none": P}
    for T1 in (0.05, 0.1, 0.2):
        g1 = pl.col("num_conflict") == 1
        g2 = g1 & (pl.col("sup_nums_all") >= 1)
        for tag, g in (("G1 all-conflict", g1), ("G2 conflict-cluster", g2)):
            out[f"{tag} T1={T1}"] = P.with_columns(pl.when(g & (pl.col("p1") < T1)).then(0.0).otherwise(pl.col("prob")).alias("prob"))
    return out
ctx = Ctx()
drop = pl.read_parquet(EXP_DIR / "sim_dropped_s1.parquet")
ctx_sim = Ctx(); ctx_sim.lab = ctx_sim.lab.join(drop.rename({"s1_idx": "true_s1"}), on="true_s1", how="anti")
ctx_sim.val = ctx_sim.val.join(drop, on="s1_idx", how="anti"); ctx_sim.n_true = ctx_sim.n_true.join(drop, on="s1_idx", how="anti")
Ptr = load("train", EXP_DIR / "E07_crossfit_train_pred.parquet")
Psim = load("sim", EXP_DIR / "sim_E07_pred.parquet")
Pte = load("test", EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet")
print(f"{'variant':30s} {'val F':>8s} {'h0':>8s} {'h1':>8s} {'sim F':>8s} {'test acc':>9s}")
vt, vs, ve = variants(Ptr), variants(Psim), variants(Pte)
for k in vt:
    r = ctx.f05(expf(vt[k])); rs = ctx_sim.f05(expf(vs[k])); acc = expf(ve[k]).height / Pte.height
    print(f"{k:30s} {r['F']:.5f} {r['F_h0']:.5f} {r['F_h1']:.5f} {rs['F']:.5f} {acc:9.4f}", flush=True)
print("reference: BASELINE val 0.97826, sim 0.97574, test acceptance 0.5878")
