import os

with open("src/config.py", "r") as f:
    config_text = f.read()

if "LAYER4_FALLBACK_THRESHOLD" not in config_text:
    config_text = config_text.replace(
        "LAYER4_INTERNAL_TOP_K: int = 200",
        "LAYER4_INTERNAL_TOP_K: int = 500\n    LAYER4_FALLBACK_THRESHOLD: int = 50\n    LAYER4_FALLBACK_TOP_K: int = 10"
    )
    with open("src/config.py", "w") as f:
        f.write(config_text)


with open("src/blocking.py", "r") as f:
    blocking_text = f.read()

# Add to globals declaration
if "_GLOBAL_S1_NAMES" not in blocking_text:
    blocking_text = blocking_text.replace(
        "global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES",
        "global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES\n            global _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES"
    )
    blocking_text = blocking_text.replace(
        "_GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY",
        "_GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY\n            _GLOBAL_S1_NAMES = s1_names\n            _GLOBAL_TARGET_NAMES = target_names"
    )

# Add to worker variables
if "s1_names = _GLOBAL_S1_NAMES" not in blocking_text:
    blocking_text = blocking_text.replace(
        "min_sim = _GLOBAL_MIN_SIM",
        "min_sim = _GLOBAL_MIN_SIM\n    s1_names = _GLOBAL_S1_NAMES\n    target_names = _GLOBAL_TARGET_NAMES"
    )

# Inject fallback logic
fallback_logic = """        candidate_ids = reached_ids[scores[reached_ids] >= min_sim]

        if len(candidate_ids) < getattr(config, "LAYER4_FALLBACK_THRESHOLD", 0):
            from rapidfuzz import process, fuzz
            query_name = s1_names[s1_idx]
            if query_name and len(query_name) > 3:
                fallback_results = process.extract(
                    query_name, 
                    target_names, 
                    scorer=fuzz.token_set_ratio, 
                    limit=getattr(config, "LAYER4_FALLBACK_TOP_K", 10)
                )
                import numpy as np
                fallback_indices = np.array([match[2] for match in fallback_results if match[1] >= 60.0], dtype=np.int32)
                if len(fallback_indices) > 0:
                    for t_idx in fallback_indices:
                        if scores[t_idx] < min_sim:
                            scores[t_idx] = min_sim + 0.01
                    candidate_ids = np.unique(np.concatenate([candidate_ids, fallback_indices]))
                    reached_ids = np.unique(np.concatenate([reached_ids, fallback_indices]))

        if not len(candidate_ids):"""

blocking_text = blocking_text.replace(
"""        candidate_ids = reached_ids[scores[reached_ids] >= min_sim]
        if not len(candidate_ids):""",
    fallback_logic
)

with open("src/blocking.py", "w") as f:
    f.write(blocking_text)

print("Patch applied successfully.")
