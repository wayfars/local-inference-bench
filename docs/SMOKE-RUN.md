# Live connectivity smoke run

This is a real API run on four authored synthetic tasks, repeated once at concurrency 1 and once at concurrency 2, plus one warmup. All eight scored requests completed and passed their exact checks. The purpose is to exercise streaming transport, completion validation, grading, concurrency grouping, and report generation.

The endpoint was shared with other work, including a simultaneous task-evaluation run. Hardware, model artifact/revision, server commit, background load, cache state, and server defaults were not pinned. The report therefore does not support a model ranking, a concurrency speedup claim, a hardware benchmark, or a speculative-decoding comparison. Server rates retain backend-specific semantics; client-derived token rates remain unavailable. Tiny smoke prompts do not establish general capability or injection resistance.

See [the report](live-connectivity-smoke.md) and [machine-readable results](live-connectivity-smoke.json). Prompts and response text were not saved. Workload and runner hashes identify the exact public source inputs, not the full serving environment.
