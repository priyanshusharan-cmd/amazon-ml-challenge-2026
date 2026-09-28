"""Group (sibling) features: how a record relates to the other records whose best S1 is the same entity."""
import polars as pl

from harness import cache_path

GROUP_BASE = ["g_all", "g_conf", "g_psum", "sup_nums_all", "sup_nums_conf", "sup_num1_all", "sup_num1_conf",
              "sup_name_all", "sup_name_conf", "sup_both_all", "sup_both_conf", "sup_nums_frac", "sup_name_frac"]


def load_strings(split):
    return pl.concat([pl.read_parquet(cache_path(f"{split}_{s}_norm.parquet"), columns=["addr_nums", "name_core"])
                      for s in ("source2", "source3")]).with_row_index("q_idx").with_columns(
        pl.col("addr_nums").str.split(" ").list.first().alias("num1"),
        pl.col("name_core").str.replace_all(" ", "").alias("nkey")).drop("name_core")


def group_features(assign: pl.DataFrame, strings: pl.DataFrame, prefix="", conf=0.5) -> pl.DataFrame:
    """assign: (q_idx, s1_idx, prob) = each record's best S1 and its probability."""
    d = assign.join(strings, on="q_idx").with_columns((pl.col("prob") >= conf).cast(pl.Int32).alias("conf"))

    def support(keys, tag):
        grp = ["s1_idx"] + keys
        return [(pl.len().over(grp) - 1).alias(f"sup_{tag}_all"),
                (pl.col("conf").sum().over(grp) - pl.col("conf")).alias(f"sup_{tag}_conf")]

    d = d.with_columns(
        (pl.len().over("s1_idx") - 1).alias("g_all"),
        (pl.col("conf").sum().over("s1_idx") - pl.col("conf")).alias("g_conf"),
        (pl.col("prob").sum().over("s1_idx") - pl.col("prob")).alias("g_psum"),
        *support(["addr_nums"], "nums"), *support(["num1"], "num1"), *support(["nkey"], "name"),
        *support(["addr_nums", "nkey"], "both"))
    no_nums = pl.col("addr_nums") == ""
    d = d.with_columns([pl.when(no_nums).then(-1).otherwise(pl.col(c)).alias(c) for c in
                        ["sup_nums_all", "sup_nums_conf", "sup_num1_all", "sup_num1_conf", "sup_both_all", "sup_both_conf"]])
    d = d.with_columns((pl.col("sup_nums_conf") / pl.col("g_conf").clip(1)).alias("sup_nums_frac"),
                       (pl.col("sup_name_conf") / pl.col("g_conf").clip(1)).alias("sup_name_frac"))
    return d.select("q_idx", *[pl.col(c).cast(pl.Float32).alias(prefix + c) for c in GROUP_BASE])
