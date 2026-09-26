import sys

# 1. Update config.py
with open("src/config.py", "r") as f:
    config_text = f.read()

config_text = config_text.replace("LAYER4_INTERNAL_TOP_K: int = 500", "LAYER4_INTERNAL_TOP_K: int = 1000")
config_text = config_text.replace("TFIDF_MIN_SIMILARITY: float = 0.32", "TFIDF_MIN_SIMILARITY: float = 0.30")

with open("src/config.py", "w") as f:
    f.write(config_text)
print("Updated config.py")

# 2. Update blocking.py for Windows fallback
with open("src/blocking.py", "r") as f:
    blocking_text = f.read()

original_pool_logic = """
            try:
                mp_ctx = mp.get_context("fork")
            except Exception as exc:
                raise RuntimeError(
                    "macOS fork Copy-on-Write mode is required for Layer 4 parallelization. "
                    "Serialized fallback is refused."
                ) from exc

            num_workers = max(1, self.num_workers)
            logger.info(
                "[%s | Layer 4] Using macOS fork Copy-on-Write mode for Layer 4 parallelization "
                "(zero-copy inheritance, W=%d workers)",
                country, num_workers
            )

            global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
            global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM
            global _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES

            _GLOBAL_TARGET_INDEX = X_target_index
            _GLOBAL_QUERY_INDEX = X_s1
            _GLOBAL_TRUE_TARGET_INDICES = true_target_indices if true_matches is not None else None
            _GLOBAL_N_TARGET = n_target
            _GLOBAL_INTERNAL_TOP_K = self.layer4_internal_top_k
            _GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY
            _GLOBAL_S1_NAMES = s1_names
            _GLOBAL_TARGET_NAMES = target_names

            if n_s1 <= num_workers * 4:
                batch_size = max(1, (n_s1 + num_workers - 1) // num_workers)
            elif n_s1 <= 4000:
                batch_size = max(25, n_s1 // (num_workers * 4))
            else:
                batch_size = max(1000, n_s1 // (num_workers * 4))

            tasks = [(i, min(i + batch_size, n_s1)) for i in range(0, n_s1, batch_size)]

            try:
                with mp_ctx.Pool(
                    processes=num_workers,
                    initializer=_init_layer4_worker_scratch,
                    initargs=(n_target, X_target_index.indices.dtype),
                ) as pool:
                    batch_outputs = pool.map(_layer4_worker_task, tasks)"""

new_pool_logic = """
            global _GLOBAL_TARGET_INDEX, _GLOBAL_QUERY_INDEX, _GLOBAL_TRUE_TARGET_INDICES
            global _GLOBAL_N_TARGET, _GLOBAL_INTERNAL_TOP_K, _GLOBAL_MIN_SIM
            global _GLOBAL_S1_NAMES, _GLOBAL_TARGET_NAMES

            _GLOBAL_TARGET_INDEX = X_target_index
            _GLOBAL_QUERY_INDEX = X_s1
            _GLOBAL_TRUE_TARGET_INDICES = true_target_indices if true_matches is not None else None
            _GLOBAL_N_TARGET = n_target
            _GLOBAL_INTERNAL_TOP_K = self.layer4_internal_top_k
            _GLOBAL_MIN_SIM = config.TFIDF_MIN_SIMILARITY
            _GLOBAL_S1_NAMES = s1_names
            _GLOBAL_TARGET_NAMES = target_names

            import sys
            is_windows = sys.platform == "win32"
            num_workers = 1 if is_windows else max(1, self.num_workers)

            if n_s1 <= num_workers * 4:
                batch_size = max(1, (n_s1 + num_workers - 1) // num_workers)
            elif n_s1 <= 4000:
                batch_size = max(25, n_s1 // (num_workers * 4))
            else:
                batch_size = max(1000, n_s1 // (num_workers * 4))

            tasks = [(i, min(i + batch_size, n_s1)) for i in range(0, n_s1, batch_size)]

            if is_windows:
                logger.info(
                    "[%s | Layer 4] Windows detected. Falling back to safe sequential execution (W=1) "
                    "to prevent Memory OOM from spawn serialization.", country
                )
                _init_layer4_worker_scratch(n_target, X_target_index.indices.dtype)
                batch_outputs = [_layer4_worker_task(t) for t in tasks]
            else:
                try:
                    mp_ctx = mp.get_context("fork")
                except Exception as exc:
                    raise RuntimeError("macOS/Linux fork Copy-on-Write mode is required.") from exc
                logger.info(
                    "[%s | Layer 4] Using macOS/Linux fork Copy-on-Write mode for Layer 4 parallelization "
                    "(zero-copy inheritance, W=%d workers)",
                    country, num_workers
                )
                with mp_ctx.Pool(
                    processes=num_workers,
                    initializer=_init_layer4_worker_scratch,
                    initargs=(n_target, X_target_index.indices.dtype),
                ) as pool:
                    batch_outputs = pool.map(_layer4_worker_task, tasks)"""

blocking_text = blocking_text.replace(original_pool_logic, new_pool_logic)

with open("src/blocking.py", "w") as f:
    f.write(blocking_text)
print("Updated blocking.py")

