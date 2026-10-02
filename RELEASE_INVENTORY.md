# Release inventory

This directory contains the original implementation for the standalone `inference-bench` release. The Python package under `src/inference_bench/` was written for this release; it does not vendor third-party benchmark implementation code. `data/synthetic-tasks.json` contains small, synthetic smoke tasks. `PORTFOLIO.md` gives public project context. The README, JSON examples, tests, CI workflow, MIT license, and environment-variable example are release documentation and support files.

The release tree contains no private user or employer data, private logs, local model files, credentials, or source-repository history. Example endpoints are loopback addresses and example credentials are placeholders. Build output, virtual environments, bytecode, caches, and egg metadata are excluded by `.gitignore` and are not part of this inventory.

`docs/` contains a reviewed live connectivity smoke report, its machine-readable results with prompts/responses omitted, and explicit limits on interpreting its shared-endpoint timings.
