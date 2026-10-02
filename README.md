# Local Inference Bench

A small Python tool for comparing local or private OpenAI-compatible chat completion servers under reproducible workloads. It records streamed time to first token (TTFT), end-to-end latency, server-reported usage where present, request failures, simple task acceptance checks, and bounded concurrent load. The examples use only loopback endpoints and synthetic prompts.

The JSON result and Markdown report share a stable shape: a run manifest, task/case IDs, profile names, per-request outcomes, and summaries. Missing measurements remain `null`; the runner does not infer token counts, speculative engagement, or cold-start behavior.

## Quick start

Requires Python 3.11+. The runner has no runtime dependencies.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
inference-bench --config configs/local-example.json --output runs
```

Before running, edit the example manifest fields with public-safe hardware, model artifact/revision, quantization, runtime version, and exact server commit. Keep the endpoint loopback for a local server. For authenticated endpoints, put the header in the local config as `"headers": {"Authorization": "Bearer ${LOCAL_BENCH_API_KEY}"}` and export the value in the shell; environment references are expanded at runtime. **Headers and endpoint URLs are never copied to the result manifest.** Never put secrets into JSON, prompts, test fixtures, result notes, or a committed `.env` file.

Only the numeric generation parameters `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `n`, `seed`, `frequency_penalty`, and `presence_penalty` are accepted under a profile's `request` object. Other request parameters are rejected before any call so they cannot silently affect a run without appearing in the manifest. The manifest records these configured overrides; server defaults that were not configured through this request object remain unavailable.

`data/synthetic-tasks.json` is a four-case smoke workload, not a validated capability benchmark, safety evaluation, or general answer-quality test. It contains no user, employer, client, or retrieved personal data. Replace or extend it with purpose-built synthetic cases for a meaningful study. Each case needs a unique short public-safe `id`, a prompt, and at most one grader: `expected_exact`, `expected_exact_any`, `expected_json`, or `expected_any`. Exact grading compares the full response case-sensitively after trimming and collapsing whitespace. JSON grading parses the whole response and compares canonical JSON values with type-sensitive matching. `expected_any` is a case-insensitive whole-phrase substring heuristic and must not be presented as semantic grading.

## What it measures

- **TTFT:** elapsed wall time from sending the HTTP request until the first non-empty streamed text delta. Server queue time and prompt processing are included. A usage-only stream does not count as first token.
- **Stream validity:** a response is only successful after a non-empty text response, `finish_reason: "stop"`, and the `[DONE]` event. Truncation, length limits, content filters, tool calls, unknown finish reasons, malformed events, and server error events are not accepted.
- **End-to-end latency:** elapsed time through stream completion. Failed requests are recorded separately and excluded from latency summaries.
- **Completion/prompt token counts:** only values reported in a streamed `usage` object are used. The runner does not derive client-side tokens/second: usage may include hidden reasoning tokens, while visible text can arrive in one buffered chunk, making elapsed time after TTFT misleading. True decode throughput stays unavailable in this runner. `timings.predicted_per_second` is shown as a server-reported rate, with server-specific semantics. The legacy client-rate JSON fields remain `null`.
- **Concurrency:** each configured level submits the benchmark cases/repetitions through a thread pool capped at that level. This is request concurrency, not a claim about server parallel slots. Each profile/concurrency/phase group reports its complete batch wall time, including the final partial worker wave, plus unique accepted tasks per batch second. Summaries keep concurrency levels separate.
- **Correctness:** exact text and JSON-value graders report accepted and checked counts alongside speed. `expected_any` is a clearly labelled substring heuristic. The `checked` denominator includes every request with a grader; endpoint errors, incomplete streams, and truncated responses count as rejected. None of these checks establishes broad answer quality.
- **Failures/timeouts:** each failed call gets a safe status/error category. HTTP response bodies, request headers, and endpoint URLs are not saved.

Warmups are made before measured requests and labelled `warmup`; they are excluded from scored summaries. A warmup is **not a cold-start measurement**. The runner does not unload, restart, or load models. True cold-start benchmarks need an external, controlled orchestration layer and must record the process/model load interval separately. `timeout_seconds` is the per-socket inactivity timeout used by Python's HTTP client; it is not a strict whole-request deadline, so a server that continues sending bytes can run longer.

## Comparison methodology

Use the same task set, model artifact, quantization, context settings, request parameters, host conditions, and concurrency for the profiles being compared. Record server name/version and exact commit. Alternate or randomize profile order across repeated trials when possible; a single run is sensitive to thermal state, background load, cache state, and server scheduling. Compare paired tasks and retain failures and per-task results. Do not report only the fastest aggregate.

For speculative decoding, start from `configs/speculative-comparison.example.json`. It models a no-spec server on loopback port 8000 and a speculative server on 8001. Start both with equivalent target model artifacts and controlled settings, or run separate configs against the same host in randomized order. The tool cannot inspect backend logs and an accepted API response does not prove draft decoding was used. Record backend-specific **positive engagement evidence**, such as a nonzero accepted-draft counter (`draft_n_accepted > 0`) or a backend initialization log explicitly confirming the draft context was created. Put the observed signal, value, and `validated: true` in the profile metadata only after verifying it. The report warns when that evidence is missing. Acceptance rate alone does not show a speed benefit; report accepted tokens/block size if available from the server, otherwise mark it unavailable.

The example profile's `speculative_engagement.validated` value is false by design and its value is a placeholder. Configured text, including `validated: true`, is never treated as proof. The report only marks engagement verified when measured stream metrics contain a positive `draft_n_accepted` count. If the backend uses another positive counter or startup log, record that evidence for humans in study notes and describe the report's counter limitation; speculative settings are server startup configuration, and `request` cannot activate a server-side backend mode.

## Output and privacy

The output directory contains `<run_id>.json` and `<run_id>.md`. By default, prompts and responses are not written into result records. `--include-text` opts into saving both and should be used only with non-sensitive test data. The manifest stores explicitly selected reproducibility metadata, not endpoint URLs or headers. Filesystem-like artifact values are dropped; slash-shaped model artifact references are retained only when written as an explicit public `hf:organization/model` reference. URL- and credential-like metadata values are dropped. The manifest includes SHA-256 hashes of the workload file and runner source and records supported generation settings from the request actually sent. Run, profile, and case IDs must use unique short public-safe identifiers. Treat generated results as data that still needs review before publication.

Before public release, inspect every fixture and output, run secret scanning, and confirm model/data licenses and provider terms permit the planned use. Do not include private logs, machine-specific paths, credentials, or real user content. This tool does not upload results.

## Synthetic offline example

The test suite uses mocked HTTP/SSE responses and emits no model performance claim. Any offline walkthrough or hand-authored report must be labelled **SYNTHETIC DEMO — not a real model measurement**. Do not interpret values from mocked responses as speed results.

## Architecture

`src/inference_bench/runner.py` owns config parsing, streamed HTTP requests, task checks, summaries, manifests, and reports. `cli.py` exposes the command line. The harness uses Python's standard library so the HTTP and timing behavior is inspectable and can run offline. Tests replace only the HTTP transport with deterministic SSE fixtures.

## Limits

This is a starter harness, not a universal benchmarking framework. It does not control GPU clocks, pin CPU affinity, drain hardware, discover model server settings, restart models, inspect runtime logs, read server-side queue/slot metrics, measure energy, calculate monetary cost, or verify semantic quality. Streaming chunk delivery and proxy buffering can affect TTFT. Only OpenAI-compatible `/chat/completions` streaming servers are supported. Report limitations with every published result.

## License and attribution

MIT. See [LICENSE](LICENSE). Uses Python standard library APIs; pytest is an optional development dependency. No third-party benchmark code or model assets are bundled.

## Related portfolio

See [the portfolio map](PORTFOLIO.md) for the related application, task evaluation, inference, and agent operations projects.

## Reviewed live smoke run

A [shared-endpoint connectivity run](docs/SMOKE-RUN.md) exercises the reporting path on the four synthetic tasks. Its environment was not controlled, so its timings do not support comparative performance claims.
