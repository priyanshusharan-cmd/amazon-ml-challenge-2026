"""Guard against silently attaching another query's text during bounded reads."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/v3"))
from enrich_rows import QueryTexts
import numpy as np
import polars as pl


class TextGatherTests(unittest.TestCase):
    def test_reordered_duplicate_indices_across_sources_and_row_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for source, prefix, n in [("source2", "two", 5), ("source3", "three", 4)]:
                pl.DataFrame({"business_name": [f"{prefix}{i}" for i in range(n)],
                              "business_address": [f"{i} Main" for i in range(n)]}).write_parquet(
                                  root / f"test_{source}.parquet", row_group_size=2)
                pl.DataFrame({"name_core": [f"core_{prefix}{i}" for i in range(n)],
                              "addr_norm": [f"{i} main" for i in range(n)]}).write_parquet(
                                  root / f"test_{source}_norm.parquet", row_group_size=3)
            reader = QueryTexts("test", cache_mb=.001, cache_dir=root)
            result = reader.take(np.array([8, 1, 5, 4, 1, 0]))
            self.assertEqual(result["q_name"].to_list(), ["three3", "two1", "three0", "two4", "two1", "two0"])
            self.assertEqual(result["q_core"].to_list(), ["core_" + x for x in result["q_name"]])
            self.assertEqual(reader.take(np.array([], dtype=np.int64)).height, 0)
            with self.assertRaises(IndexError):
                reader.take(np.array([9]))
            reader.close()


if __name__ == "__main__":
    unittest.main()
