# Inference benchmark: live-connectivity-smoke

**Real server run.**

- Created: 2026-10-02T16:59:33.285156+00:00
- Host: {"hardware": "not recorded", "label": "shared-local-smoke", "os": "Linux"}
- Settings: {"case_count": 4, "concurrency": [1, 2], "include_text": false, "repetitions": 1, "timeout_seconds": 30.0, "warmup_requests": 1}

Warmup requests are reported separately and excluded from measured summaries. Warmup is not a cold-start test: this tool does not unload/restart models. See README for methodology and caveats.

## active-local-backend

| Phase / concurrency | Requests | Success | Failures | Accepted / checked | Accepted task IDs | Unique accepted tasks / batch s | Median batch wall (ms) | Median E2E (ms) | P95 E2E (ms) | Median TTFT (ms) | Server rate tok/s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| measured@concurrency=1 | 4 | 4 | 0 | 4 / 4 | synthetic-abstention, synthetic-color-fact, synthetic-instruction-boundary, synthetic-ordering | 0.40 | 9991.01 | 2496.59 | 3126.57 | 2443.78 | 13.72 |
| measured@concurrency=2 | 4 | 4 | 0 | 4 / 4 | synthetic-abstention, synthetic-color-fact, synthetic-instruction-boundary, synthetic-ordering | 0.72 | 5536.55 | 2767.83 | 3336.95 | 2708.03 | 14.07 |
| warmup@concurrency=1 | 1 | 1 | 0 | 1 / 1 | synthetic-color-fact | — | — | 1052.26 | 1052.26 | 1052.03 | 38.81 |

*Token rates are shown only when supplied by the server as `timings.predicted_per_second`; client-derived rates are omitted because reasoning tokens and stream buffering can distort them. `batch wall` measures elapsed time for the complete request batch at that concurrency.

## Failures

No request failures.
