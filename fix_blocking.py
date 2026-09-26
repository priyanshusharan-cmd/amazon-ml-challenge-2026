import re

with open("src/blocking.py", "r") as f:
    text = f.read()

# The section we want to replace starts with:
#             try:
#                 mp_ctx = mp.get_context("fork")
# ... and ends right after batch_outputs = pool.map(_layer4_worker_task, tasks)

pattern = re.compile(
    r"(\s*)try:\s*mp_ctx = mp\.get_context\(\"fork\"\).*?batch_outputs = pool\.map\(_layer4_worker_task, tasks\)",
    re.DOTALL
)

new_logic = """\g<1>import sys
\g<1>is_windows = sys.platform == "win32"
\g<1>num_workers = 1 if is_windows else max(1, self.num_workers)

\g<1>logger.info(
\g<1>    "[%s | Layer 4] Determining parallelization strategy (W=%d workers)",
\g<1>    country, num_workers
\g<1>)

\g<1>global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
\g<1>global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM
\g<1>global _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES

\g<1>_GLOBAL_TARGET_INDEX = X_target_index
\g<1>_GLOBAL_QUERY_INDEX = X_s1
\g<1>_GLOBAL_TRUE_TARGET_INDICES = true_target_indices if true_matches is not None else None
\g<1>_GLOBAL_N_TARGET = n_target
\g<1>_GLOBAL_INTERNAL_TOP_K = self.layer4_internal_top_k
\g<1>_GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY
\g<1>_GLOBAL_S1_NAMES = s1_names
\g<1>_GLOBAL_TARGET_NAMES = target_names

\g<1>if n_s1 <= num_workers * 4:
\g<1>    batch_size = max(1, (n_s1 + num_workers - 1) // num_workers)
\g<1>elif n_s1 <= 4000:
\g<1>    batch_size = max(25, n_s1 // (num_workers * 4))
\g<1>else:
\g<1>    batch_size = max(1000, n_s1 // (num_workers * 4))

\g<1>tasks = [(i, min(i + batch_size, n_s1)) for i in range(0, n_s1, batch_size)]

\g<1>if is_windows:
\g<1>    logger.info("[%s | Layer 4] Windows detected. Falling back to safe sequential execution (W=1).", country)
\g<1>    _init_layer4_worker_scratch(n_target, X_target_index.indices.dtype)
\g<1>    batch_outputs = [_layer4_worker_task(t) for t in tasks]
\g<1>else:
\g<1>    try:
\g<1>        mp_ctx = mp.get_context("fork")
\g<1>    except Exception as exc:
\g<1>        raise RuntimeError("macOS/Linux fork Copy-on-Write mode is required.") from exc
\g<1>    with mp_ctx.Pool(
\g<1>        processes=num_workers,
\g<1>        initializer=_init_layer4_worker_scratch,
\g<1>        initargs=(n_target, X_target_index.indices.dtype),
\g<1>    ) as pool:
\g<1>        batch_outputs = pool.map(_layer4_worker_task, tasks)"""

if "is_windows = sys.platform" not in text:
    new_text = pattern.sub(new_logic, text)
    if new_text == text:
        print("REGEX FAILED TO MATCH!")
    else:
        with open("src/blocking.py", "w") as f:
            f.write(new_text)
        print("Successfully applied Windows fallback.")
else:
    print("Fallback already present.")
