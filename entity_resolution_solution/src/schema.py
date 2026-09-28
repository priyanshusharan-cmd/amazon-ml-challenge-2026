"""
Intermediate Parquet Schema Definitions for Entity Resolution Pipeline.
Freezes data structure across Phase 1 sampling and full execution.
"""

import polars as pl

ENTITY_SCHEMA = {
    "entity_id": pl.Utf8,
    "source": pl.Utf8,          # 'S1', 'S2', 'S3'
    "country": pl.Utf8,
    "name_orig": pl.Utf8,
    "addr_orig": pl.Utf8,
    "name_norm": pl.Utf8,
    "addr_norm": pl.Utf8,
    "name_tokens": pl.Utf8,      # Space-separated clean name tokens
    "addr_tokens": pl.Utf8,      # Space-separated clean address tokens
    "name_prefix": pl.Utf8,      # First 4 chars of name_norm
    "rare_token": pl.Utf8,       # Most informative/rare token in business name
    "has_address": pl.Boolean,
}

GROUND_TRUTH_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
    "match_count": pl.Int32,
    "match_class": pl.Utf8,     # '0', '1', '2+'
}
