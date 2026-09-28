"""Stage-1 (BASELINE model) scores for the simulated split."""
import sys
import lightgbm as lgb
sys.path.insert(0, "../business_entity_resolution/src")
from harness import BASELINE, cache_path, mem
from train import score_split
m = lgb.Booster(model_file=str(BASELINE / "lgbm_model.txt"))
sc = score_split(m, "sim")
sc.write_parquet(cache_path("sim_scored.parquet"))
print("scored", sc.height, mem())
