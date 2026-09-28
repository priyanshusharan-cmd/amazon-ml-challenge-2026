import polars as pl
from harness import EXP_DIR, cache_path
te = pl.read_parquet(EXP_DIR / "SUBMISSION_E09" / "test_stage3_pred.parquet").join(pl.read_parquet(EXP_DIR / "qtable_test.parquet", columns=["q_idx", "p1"]), on="q_idx")
res = te.filter((pl.col("p1") < 0.7) & (pl.col("prob") >= 0.65))
print("rescued on test:", res.height, " p1 quantiles:", [round(res["p1"].quantile(q), 3) for q in (0.1, 0.5, 0.9)])
fz = pl.read_parquet(EXP_DIR / "fz_test.parquet")
g = pl.concat([pl.read_parquet(f, columns=["q_idx", "g_all", "g_conf", "sup_nums_conf", "sup_name_conf", "nm_tset", "ad_tset", "cos_comb", "s1_rank1_deg", "q_addr_missing"])
               for f in sorted((EXP_DIR / "s2_test").glob("*.parquet"))])
r = res.join(g, on="q_idx").join(fz.select("q_idx", "fz_both_conf", "fz_addr_r_conf"), on="q_idx")
print(r.select(pl.all().exclude("q_idx", "s1_idx").median()))
sm = r.sample(14, seed=3)
rq = pl.concat([pl.scan_parquet(cache_path(f"test_{s}.parquet")) for s in ("source2", "source3")]).with_row_index("q_idx").join(sm.lazy().select("q_idx"), on="q_idx", how="semi").collect()
rs = pl.scan_parquet(cache_path("test_source1.parquet")).with_row_index("s1_idx").join(sm.lazy().select("s1_idx"), on="s1_idx", how="semi").collect()
for x in sm.iter_rows(named=True):
    a = rq.filter(pl.col("q_idx") == x["q_idx"]).row(0, named=True); b = rs.filter(pl.col("s1_idx") == x["s1_idx"]).row(0, named=True)
    print(f"\np1={x['p1']:.2f}->{x['prob']:.2f} [{b['country']}] g_conf={x['g_conf']:.0f} sup_nums={x['sup_nums_conf']:.0f} deg={x['s1_rank1_deg']:.0f}")
    print(f"   REC: {a['business_name']} | {a['business_address']}\n   S1 : {b['business_name']} | {b['business_address']}")
