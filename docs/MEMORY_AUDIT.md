# Memory audit — 2026-09-27

Scope: current working tree, including pre-existing uncommitted edits in
`src/blocking.py` and `tests/test_correctness.py`. No pipeline implementation
was changed during this audit. Reviewed ingestion, blocking, feature extraction,
training, inference, benchmark, documentation, and correctness coverage.

## Verified scale and worker decision

Parquet metadata for India reports 40,021 validation queries, 843,167 training
queries, and 4,133,346 target records (2,017,799 S2 plus 2,115,547 S3).

Reproducing the supplied 3.51 GiB available-memory snapshot selects one worker:

- Fixed reserve: 3 GiB.
- Score/touched arrays: 49,600,152 bytes per worker (float64 + int32 per target).
- Fixed fork allowance: 256 MiB per worker.
- Result estimate: 64 queries × 1,000 candidates × 96 bytes.
- Total estimate: 324,179,608 bytes per worker.
- Sixteen workers require approximately 7.83 GiB available including the reserve.

The fixed allowance alone costs 4 GiB for 16 fork workers. It is a conservative
estimate, not a measurement of private worker memory. Conversely, the planner
omits important growth sources below. Changing this allowance alone cannot
establish safety. It also falls back to one worker even when that worker does
not fit its own budget.

## Findings

1. **Country-wide candidate retention is the principal scaling risk.** Layers
   2–4 populate nested Python dictionaries; Layer 5 runs only after all queries
   finish. At the configured internal top-K, validation can retain 40,021,000
   Layer 4 pairs and training 843,167,000, before additional Layer 2/3 pairs.
   These are ceilings, not observed pair counts. A representative hits dict,
   integer target ID, and floating score occupy 236 bytes on this interpreter,
   excluding the enclosing map's storage. Forty million such entries would
   occupy roughly 8.8 GiB before that additional overhead. Actual memory depends
   on candidate counts and object sharing.

2. **Small tasks do not bound outstanding results.** Both platform paths use
   `Pool.imap`. Inspection of the installed Python implementation confirms that
   completed results accumulate in a deque and an out-of-order dictionary.
   A slow early query or parent-side fuzzy fallback can grow these buffers.
   Switching to unordered iteration alone would not impose backpressure.

3. **Dead indexes remain live at the planning point.** Rare-token frequencies,
   rare-token postings, numeric postings, the target-ID lookup, vocabulary input
   list, and vectorizer remain referenced after their last useful operation.
   Several contain millions of Python objects. Release them before planning
   workers; measure the benefit, since freeing Python objects does not guarantee
   immediate release of all allocator pages to the OS.

4. **Fork shares pages only while they remain unmodified.** Large parent Python
   object graphs remain inherited. Parent candidate mutations and child garbage
   collection can erode sharing. Deferring fuzzy fallback to the parent avoids
   one source of child string traversal, but does not prove zero-copy operation.
   Spawn uses mapped sparse arrays but has a larger import/private-memory cost;
   switching Linux to spawn is not automatically a memory saving.

5. **Fallback is serial and potentially expensive.** For queries below 50 TF-IDF
   candidates, the parent scans target names using RapidFuzz. With 4.13 million
   targets this can dominate consumption and leave retrieval workers idle or
   results buffered. Sixteen processes do not imply sustained 16-core usage.

6. **Peak-memory reporting is misleading.** The blocker samples parent RSS at
   completion, not the time-series peak of the process tree. The benchmark prints
   the requested CPU count rather than the selected worker count. Summed RSS
   would also overcount fork-shared pages; use Linux PSS/USS plus system available
   memory, swap, and relevant container limits where accessible.

7. **Tests do not validate production-scale parallel safety.** All seven tests
   in `test_correctness.py` passed. The separate fork equivalence script has its
   own worker implementation, old batching, and different top-K/thresholds;
   it does not establish equivalence of the current production execution paths.

8. **Later stages also retain whole datasets.** Feature extraction loads all
   candidate rows, constructs whole-entity dictionaries and per-pair lists, and
   retains every output feature batch before concatenation. Its 50,000-row inner
   loop is not an end-to-end memory bound. Training and inference load complete
   feature tables and NumPy matrices. Country candidate outputs are also held
   until final concatenation.

## Recommended implementation sequence

1. Preserve the full-country TF-IDF fit, query feature ordering, float64 scoring,
   thresholds, internal top-K, Layer 2/3 posting caps, and stable ranking ties.
   Fitting a new vocabulary per query batch would change retrieval semantics.
2. Keep reusable target indexes, but generate Layer 2/3 candidates per query
   batch. Merge Layer 4 results, perform fallback and Layer 5 ranking immediately,
   update diagnostics, and write final top-50 rows to temporary Parquet parts.
   Release intermediate candidates after each batch. Preserve the public
   DataFrame API for small tests; use an incremental writer in the pipeline.
3. Limit submitted-but-not-consumed tasks explicitly, initially to one or two
   tasks per worker. Preserve query/output ordering with a bounded reorder window.
4. Release obsolete build objects before starting workers; prevent accidental
   collection of the inherited graph in fork children using a carefully tested
   GC lifecycle. Do not lower the safety reserve as a substitute for these changes.
5. Return actual worker-plan metadata and monitor parent/child memory throughout
   index construction, retrieval, fallback, ranking, and writing. Calibrate worker
   allowances using representative queries against the full target index.
6. Compare production-path results at 1 and 16 workers, including fallback,
   tied scores, empty names, diagnostics, and exact output ordering. Then run
   full India validation with memory monitoring before claiming 16-worker safety.
7. Stream feature extraction and inference separately; assess training capacity
   from actual final pair counts and feature storage.

## Windows comparison and limits

Git HEAD selected workers using CPU/query count only. The RAM selector producing
the supplied log is part of the pre-existing local diff and now applies to both
platforms, with an even larger fixed allowance for spawn. Thus Windows running
16 workers previously does not establish that this version will choose 16 there.
The Windows run's code revision, peak memory, background load, and paging are
not available; no single OS-specific cause can be proven from this log.

No full-country workload or crash reproduction was launched during this audit.
The available sandbox exposes only a limited process view and no readable
`/sys/fs/cgroup/memory.*` files, so it cannot attribute the supplied low-memory
snapshot to particular host processes or confirm an OOM kill.
