from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BenchmarkError(Exception):
    """Raised for invalid benchmark configuration or input data."""


def expand_env(value: Any) -> Any:
    """Expand each ${NAME} once; inserted environment values are never rescanned."""
    if isinstance(value, str):
        remainder = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", "", value)
        if "${" in remainder:
            raise BenchmarkError("malformed environment variable reference")
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise BenchmarkError(f"required environment variable {name!r} is not set")
            return os.environ[name]
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-standard JSON constant")


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)


def load_json(path: str | Path) -> Any:
    try:
        return _strict_json_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkError(f"cannot load {path}: {exc}") from exc


@dataclass
class Result:
    run_id: str
    profile: str
    case_id: str
    repetition: int
    phase: str
    concurrency: int
    status: str
    accepted: bool | None
    acceptance_reason: str | None
    ttft_ms: float | None
    e2e_ms: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    # Kept for output-schema compatibility; client-derived token rates are not reliable.
    observed_post_ttft_completion_tokens_per_second: float | None
    response_chars: int | None
    error: str | None = None
    response_text: str | None = None
    prompt_text: str | None = None
    concurrency_batch_ms: float | None = None
    finish_reason: str | None = None
    true_decode_tokens_per_second: float | None = None
    server_decode_tokens_per_second: float | None = None
    draft_tokens_accepted: int | None = None
    draft_tokens_proposed: int | None = None


def _safe_error(exc: BaseException) -> str:
    # Avoid including request headers, URLs with credentials, or response bodies.
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, TimeoutError):
        return "request timed out"
    return type(exc).__name__


def _parse_sse_line(line: bytes) -> tuple[str | None, dict[str, Any] | None]:
    text = line.decode("utf-8", errors="replace").strip()
    if not text or text.startswith(":") or not text.startswith("data:"):
        return None, None
    data = text[5:].strip()
    if data == "[DONE]":
        return "done", None
    try:
        payload = _strict_json_loads(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkError("server returned malformed SSE JSON") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("server returned malformed SSE event")
    return "chunk", payload


def _content_from_chunk(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        raise BenchmarkError("server returned malformed SSE choices")
    if not choices:
        return ""
    if not isinstance(choices[0], dict):
        raise BenchmarkError("server returned malformed SSE choices")
    delta = choices[0].get("delta", {})
    if delta is None:
        delta = {}
    if not isinstance(delta, dict):
        raise BenchmarkError("server returned malformed SSE delta")
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    # Some compatible APIs represent content as a list of text blocks.
    if isinstance(content, list):
        if any(not isinstance(part, dict) or not isinstance(part.get("text", ""), str) for part in content):
            raise BenchmarkError("server returned malformed SSE content blocks")
        return "".join(part.get("text", "") for part in content)
    if content in (None, ""):
        return ""
    raise BenchmarkError("server returned malformed SSE content")


def _request_one(profile: dict[str, Any], case: dict[str, Any], timeout_s: float) -> tuple[float | None, float, int | None, int | None, str, str | None, bool, dict[str, Any]]:
    endpoint = profile["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    headers.update(profile.get("headers", {}))
    body = {
        **profile.get("request", {}),
        "model": profile["model"],
        "messages": [{"role": "user", "content": case["prompt"]}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers, method="POST")
    started = time.perf_counter()
    first: float | None = None
    pieces: list[str] = []
    completion_tokens = prompt_tokens = None
    finish_reason = None
    saw_done = False
    timing_metrics: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        for line in response:
            kind, payload = _parse_sse_line(line)
            if kind == "done":
                saw_done = True
                break
            if payload is None:
                continue
            if payload.get("error"):
                raise BenchmarkError("server returned an error event")
            text = _content_from_chunk(payload)
            if text:
                if first is None:
                    first = time.perf_counter()
                pieces.append(text)
            usage = payload.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("completion_tokens"), int) and not isinstance(usage.get("completion_tokens"), bool) and usage["completion_tokens"] >= 0:
                    completion_tokens = usage["completion_tokens"]
                if isinstance(usage.get("prompt_tokens"), int) and not isinstance(usage.get("prompt_tokens"), bool) and usage["prompt_tokens"] >= 0:
                    prompt_tokens = usage["prompt_tokens"]
            timings = payload.get("timings") or payload.get("metrics") or {}
            if isinstance(timings, dict):
                timing_metrics.update(timings)
            choices = payload.get("choices", [])
            if not isinstance(choices, list):
                raise BenchmarkError("server returned malformed SSE choices")
            if choices and choices[0].get("finish_reason") is not None:
                raw_finish = choices[0]["finish_reason"]
                finish_reason = raw_finish if isinstance(raw_finish, str) and raw_finish in {
                    "stop", "length", "tool_calls", "function_call", "content_filter", "error"
                } else "unknown"
    ended = time.perf_counter()
    return ((first - started) * 1000 if first is not None else None,
            (ended - started) * 1000,
            completion_tokens,
            prompt_tokens,
            "".join(pieces), finish_reason, saw_done, timing_metrics)


def _check(case: dict[str, Any], response: str) -> tuple[bool | None, str | None]:
    def exact_text(value: str) -> str:
        return " ".join(value.split())

    expected_json = case.get("expected_json")
    if expected_json is not None:
        try:
            actual_json = _strict_json_loads(response)
        except (json.JSONDecodeError, ValueError):
            return False, "response is not valid JSON"
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        passed = canonical(actual_json) == canonical(expected_json)
        return (passed, "JSON value match" if passed else "JSON value mismatch")
    exact = case.get("expected_exact")
    if exact is not None:
        passed = exact_text(response) == exact_text(exact)
        return (passed, "exact output match" if passed else "exact output mismatch")
    exact_any = case.get("expected_exact_any")
    if exact_any is not None:
        passed = any(exact_text(response) == exact_text(expected) for expected in exact_any)
        return (passed, "exact output match" if passed else "exact output mismatch")
    expected = case.get("expected_any")
    if expected is None:
        return None, None
    normalized = response.casefold()
    found = [phrase for phrase in expected
             if re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", normalized)]
    return (bool(found), "heuristic substring matched" if found else "heuristic substring did not match")


def _call(profile: dict[str, Any], case: dict[str, Any], *, run_id: str,
          repetition: int, phase: str, concurrency: int, timeout_s: float,
          include_text: bool) -> Result:
    try:
        ttft, e2e, completion, prompt, text, finish_reason, saw_done, timings = _request_one(profile, case, timeout_s)
        incomplete = (not saw_done or finish_reason != "stop" or not text.strip())
        has_check = any(key in case for key in ("expected_any", "expected_exact", "expected_exact_any", "expected_json"))
        accepted, reason = _check(case, text) if not incomplete else (False if has_check else None, "incomplete or truncated response" if incomplete else None)
        return Result(run_id, profile["name"], case["id"], repetition, phase, concurrency,
                      "incomplete" if incomplete else "ok", accepted, reason, ttft, e2e,
                      completion, prompt, None, len(text),
                      error="stream incomplete/truncated" if incomplete else None,
                      response_text=text if include_text else None,
                      prompt_text=case["prompt"] if include_text else None,
                      finish_reason=finish_reason,
                      server_decode_tokens_per_second=_number(timings.get("predicted_per_second")),
                      draft_tokens_accepted=_integer(timings.get("draft_n_accepted")),
                      draft_tokens_proposed=_integer(timings.get("draft_n")))
    except (urllib.error.URLError, TimeoutError, OSError, BenchmarkError, ValueError) as exc:
        return Result(run_id, profile["name"], case["id"], repetition, phase, concurrency,
                      "error", False if any(key in case for key in ("expected_any", "expected_exact", "expected_exact_any", "expected_json")) else None,
                      None, None, None, None, None, None, None, _safe_error(exc))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percentile
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _number(value: Any) -> float | None:
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return number if number is not None and math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


_PRIVATE_VALUE = re.compile(r"(?:https?://|(?:^|\s)~[\\/]|(?:^|[\\/])(?:home|users?|tmp|mnt)[\\/]|(?:^|\s)/(?:[^/\s]+/)+|[A-Za-z]:\\|\b[^\s@]+@[^\s@]+\.[^\s@]+|\b(?:bearer|token|api[_ -]?key|password)\b)", re.I)
_RELATIVE_PATH = re.compile(r"(?:^|[\s=])(?:\.{1,2}|~)[\\/]|(?:^|[\s=])(?:models?|weights?|checkpoints?|private|data|tmp)[\\/]|(?:^|[/\\])[^/\\]+\.(?:gguf|safetensors|bin|pt|pth|onnx|model|ckpt|weights|sqlite|db|jsonl|csv|parquet|txt|pkl)(?:$|[\s?#])", re.I)
_ALLOWED_REQUEST_SETTINGS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "n",
                             "seed", "frequency_penalty", "presence_penalty")


def _safe_manifest_value(value: Any) -> Any:
    """Keep simple public metadata while dropping URLs, paths, and credential-like text."""
    if isinstance(value, str):
        return None if _PRIVATE_VALUE.search(value) or _RELATIVE_PATH.search(value) else value
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _validate_public_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise BenchmarkError(f"{label} must be a short public-safe identifier using letters, numbers, dot, underscore, or dash")
    return value


def _safe_artifact_reference(value: Any) -> str | None:
    """Accept public labels or explicit Hugging Face refs, never filesystem-like refs."""
    if not isinstance(value, str):
        return None
    if "/" in value or "\\" in value:
        return value if re.fullmatch(r"hf:[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", value) else None
    return _safe_manifest_value(value)


def _profile_manifest(profile: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = ("name", "provider", "model", "quantization",
                     "server_name", "server_version", "server_commit", "context_tokens",
                     "speculative_mode")
    safe = {key: value for key in scalar_fields if key in profile
            and (value := _safe_manifest_value(profile[key])) is not None}
    if "model_artifact" in profile:
        artifact = _safe_artifact_reference(profile["model_artifact"])
        if artifact is not None:
            safe["model_artifact"] = artifact
    # Record the actual supported generation parameters sent by the runner.
    request = profile.get("request", {})
    if isinstance(request, dict):
        settings = {key: value for key in _ALLOWED_REQUEST_SETTINGS if key in request
                    and (value := _safe_manifest_value(request[key])) is not None
                    and isinstance(value, (int, float)) and not isinstance(value, bool)}
        safe["request_settings"] = settings
        safe["unconfigured_server_defaults"] = "unavailable"
    engagement = profile.get("speculative_engagement")
    if isinstance(engagement, dict):
        fields = ("signal", "value", "validated")
        clean = {key: value for key in fields if key in engagement
                 and (value := _safe_manifest_value(engagement[key])) is not None}
        if clean:
            safe["speculative_engagement"] = clean
    return safe


def summarize(results: list[Result]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for name in sorted({result.profile for result in results}):
        profile_results = [r for r in results if r.profile == name]
        profiles[name] = {}
        groups = sorted({(r.phase, r.concurrency) for r in profile_results})
        for phase, concurrency in groups:
            group = [r for r in profile_results if r.phase == phase and r.concurrency == concurrency]
            good = [r for r in group if r.status == "ok"]
            durations = [r.e2e_ms for r in good if r.e2e_ms is not None]
            accepted = [r for r in good if r.accepted is True]
            checked = [r for r in group if r.accepted is not None]
            true_decode = [r.true_decode_tokens_per_second for r in good if r.true_decode_tokens_per_second is not None]
            server_decode = [r.server_decode_tokens_per_second for r in good if r.server_decode_tokens_per_second is not None]
            accepted_drafts = [r.draft_tokens_accepted for r in good if r.draft_tokens_accepted is not None]
            proposed_drafts = [r.draft_tokens_proposed for r in good if r.draft_tokens_proposed is not None]
            key = f"{phase}@concurrency={concurrency}"
            batch_walls = [r.concurrency_batch_ms for r in group if r.concurrency_batch_ms is not None]
            batch_wall_ms = statistics.median(batch_walls) if batch_walls else None
            profiles[name][key] = {
                "requests": len(group), "successful": len(good),
                "failed": len(group) - len(good),
                "accepted": len(accepted), "checked": len(checked),
                "acceptance_rate": len(accepted) / len(checked) if checked else None,
                "accepted_task_ids": sorted({r.case_id for r in accepted}),
                "accepted_requests_per_batch_second": (len(accepted) / (batch_wall_ms / 1000)
                                                        if batch_wall_ms and batch_wall_ms > 0 else None),
                "accepted_unique_tasks_per_batch_second": (len({r.case_id for r in accepted}) / (batch_wall_ms / 1000)
                                                           if batch_wall_ms and batch_wall_ms > 0 else None),
                "e2e_ms_median": statistics.median(durations) if durations else None,
                "e2e_ms_p95": _percentile(durations, .95),
                "ttft_ms_median": statistics.median([r.ttft_ms for r in good if r.ttft_ms is not None]) if any(r.ttft_ms is not None for r in good) else None,
                "observed_post_ttft_completion_tokens_per_second_median": None,
                "true_decode_tokens_per_second_median": statistics.median(true_decode) if true_decode else None,
                "server_reported_decode_tokens_per_second_median": statistics.median(server_decode) if server_decode else None,
                "draft_tokens_accepted_total": sum(accepted_drafts) if accepted_drafts else None,
                "draft_tokens_proposed_total": sum(proposed_drafts) if proposed_drafts else None,
                "batch_wall_ms_median": batch_wall_ms,
            }
    return {"profiles": profiles}


def execute(config_path: str | Path, output_dir: str | Path, *, include_text: bool = False) -> tuple[Path, Path]:
    config = expand_env(load_json(config_path))
    config_dir = Path(config_path).resolve().parent
    cases_path = Path(config["cases"])
    if not cases_path.is_absolute():
        cases_path = config_dir / cases_path
    cases = load_json(cases_path)
    profiles = config.get("profiles", [])
    if not profiles:
        raise BenchmarkError("configuration must define at least one profile")
    if include_text is False and config.get("store_response_text"):
        raise BenchmarkError("response text storage requires the explicit --include-text flag")
    run_id = config.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    repetitions_value = config.get("repetitions", 1)
    repetitions = repetitions_value if isinstance(repetitions_value, int) and not isinstance(repetitions_value, bool) else 0
    concurrency_levels = config.get("concurrency", [1])
    timeout_value = config.get("timeout_seconds", 60)
    timeout_s = float(timeout_value) if isinstance(timeout_value, (int, float)) and not isinstance(timeout_value, bool) else float("nan")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkError("cases file must contain a non-empty JSON array")
    run_id = _validate_public_id(run_id, "run_id")
    benchmark_kind = config.get("benchmark_kind", "real")
    if not isinstance(benchmark_kind, str) or benchmark_kind not in {"real", "synthetic"}:
        raise BenchmarkError("benchmark_kind must be 'real' or 'synthetic'")
    case_ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            raise BenchmarkError("each case must be a JSON object")
        case_ids.append(_validate_public_id(case.get("id"), "case id"))
        if not isinstance(case.get("prompt"), str):
            raise BenchmarkError(f"case {case['id']!r} prompt must be a string")
        graders = [key for key in ("expected_exact", "expected_exact_any", "expected_json", "expected_any") if key in case]
        if len(graders) > 1:
            raise BenchmarkError(f"case {case['id']!r} must define at most one grader")
        if "expected_exact" in case and not isinstance(case["expected_exact"], str):
            raise BenchmarkError(f"case {case['id']!r} expected_exact must be a string")
        for key in ("expected_exact_any", "expected_any"):
            if key in case and (not isinstance(case[key], list) or not case[key]
                                or any(not isinstance(item, str) for item in case[key])):
                raise BenchmarkError(f"case {case['id']!r} {key} must be a non-empty list of strings")
    profile_names: list[str] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise BenchmarkError("each profile must be a JSON object")
        profile_names.append(_validate_public_id(profile.get("name"), "profile name"))
        if not isinstance(profile.get("base_url"), str) or not isinstance(profile.get("model"), str):
            raise BenchmarkError(f"profile {profile['name']!r} requires string base_url and model fields")
        if not isinstance(profile.get("request", {}), dict) or not isinstance(profile.get("headers", {}), dict):
            raise BenchmarkError(f"profile {profile['name']!r} request and headers must be JSON objects")
        if "request_settings" in profile:
            raise BenchmarkError(f"profile {profile['name']!r} uses obsolete request_settings; configure actual parameters under request")
        unsupported = sorted(set(profile.get("request", {})) - set(_ALLOWED_REQUEST_SETTINGS))
        if unsupported:
            raise BenchmarkError(f"profile {profile['name']!r} has unsupported request parameter(s): {', '.join(unsupported)}")
        for key, value in profile.get("request", {}).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise BenchmarkError(f"profile {profile['name']!r} request parameter {key!r} must be a finite number")
    if len(set(case_ids)) != len(case_ids) or len(set(profile_names)) != len(profile_names):
        raise BenchmarkError("case IDs and profile names must be unique")
    if (repetitions < 1 or not isinstance(concurrency_levels, list) or not concurrency_levels
            or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in concurrency_levels)):
        raise BenchmarkError("repetitions and concurrency levels must be positive")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise BenchmarkError("timeout_seconds must be positive")
    results: list[Result] = []
    # Warmup runs load/use the model but are kept out of scored measurements.
    warmups_value = config.get("warmup_requests", 0)
    warmups = warmups_value if isinstance(warmups_value, int) and not isinstance(warmups_value, bool) else -1
    if warmups < 0:
        raise BenchmarkError("warmup_requests must be a non-negative integer")
    for profile in profiles:
        for ix in range(warmups):
            case = cases[ix % len(cases)]
            results.append(_call(profile, case, run_id=run_id, repetition=ix + 1,
                                 phase="warmup", concurrency=1, timeout_s=timeout_s,
                                 include_text=include_text))
        for level in concurrency_levels:
            level = int(level)
            jobs = [(case, rep) for rep in range(1, repetitions + 1) for case in cases]
            def run_job(item: tuple[dict[str, Any], int]) -> Result:
                case, rep = item
                return _call(profile, case, run_id=run_id, repetition=rep,
                             phase="measured", concurrency=level, timeout_s=timeout_s,
                             include_text=include_text)
            batch_started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                batch_results = list(pool.map(run_job, jobs))
            batch_ms = (time.perf_counter() - batch_started) * 1000
            for result in batch_results:
                result.concurrency_batch_ms = batch_ms
            results.extend(batch_results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_kind": benchmark_kind,
        "synthetic_demo": bool(config.get("synthetic_demo", False)),
        "host": {key: safe for key in ("label", "hardware", "memory_gib", "os")
                 if isinstance(config.get("host", {}), dict) and key in config.get("host", {})
                 and (safe := _safe_manifest_value(config["host"][key])) is not None},
        # Store declared reproducibility metadata, never endpoints, headers, or
        # arbitrary profile fields that could contain credentials/private URLs.
        "profiles": [_profile_manifest(p) for p in profiles],
        "workload_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_settings": {"repetitions": repetitions, "concurrency": concurrency_levels,
                               "warmup_requests": warmups, "timeout_seconds": timeout_s,
                               "case_count": len(cases), "include_text": include_text},
    }
    result_path = output / f"{run_id}.json"
    result_path.write_text(json.dumps({"manifest": manifest,
                                       "summary": summarize(results),
                                       "results": [asdict(r) for r in results]}, indent=2) + "\n", encoding="utf-8")
    report_path = output / f"{run_id}.md"
    report_path.write_text(render_report(manifest, summarize(results), results), encoding="utf-8")
    return result_path, report_path


def render_report(manifest: dict[str, Any], summary: dict[str, Any], results: list[Result]) -> str:
    label = "SYNTHETIC DEMO — not a real model measurement" if manifest["synthetic_demo"] else "Real server run"
    lines = [f"# Inference benchmark: {manifest['run_id']}", "", f"**{label}.**", "",
             f"- Created: {manifest['created_at']}",
             f"- Host: {json.dumps(manifest.get('host', {}), sort_keys=True)}",
             f"- Settings: {json.dumps(manifest['benchmark_settings'], sort_keys=True)}", "",
             "Warmup requests are reported separately and excluded from measured summaries. Warmup is not a cold-start test: this tool does not unload/restart models. See README for methodology and caveats.", ""]
    for profile, phases in summary["profiles"].items():
        profile_manifest = next((p for p in manifest["profiles"] if p.get("name") == profile), {})
        spec_mode = profile_manifest.get("speculative_mode", "")
        engagement = profile_manifest.get("speculative_engagement", {})
        if spec_mode and spec_mode != "disabled":
            observed_accepted = sum((phase.get("draft_tokens_accepted_total") or 0)
                                    for key, phase in phases.items() if key.startswith("measured@"))
            valid_engagement = observed_accepted > 0
            gate = "positive engagement evidence recorded" if valid_engagement else "UNVERIFIED: do not attribute any speed result to speculative decoding"
            signal = f"server reported draft_n_accepted={observed_accepted}" if observed_accepted else engagement.get("signal", "signal unavailable")
            lines += [f"## {profile}", "", f"Speculative decoding: {spec_mode}. Engagement gate: **{gate}** ({signal}).", "",
                      "| Phase / concurrency | Requests | Success | Failures | Accepted / checked | Accepted task IDs | Unique accepted tasks / batch s | Median batch wall (ms) | Median E2E (ms) | P95 E2E (ms) | Median TTFT (ms) | Server rate tok/s |",
                      "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        else:
            lines += [f"## {profile}", "", "| Phase / concurrency | Requests | Success | Failures | Accepted / checked | Accepted task IDs | Unique accepted tasks / batch s | Median batch wall (ms) | Median E2E (ms) | P95 E2E (ms) | Median TTFT (ms) | Server rate tok/s |",
                      "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for phase, s in phases.items():
            def fmt(v: Any) -> str:
                return "—" if v is None else f"{v:.2f}"
            checked = f"{s['accepted']} / {s['checked']}" if s["checked"] else "—"
            accepted_ids = ", ".join(s["accepted_task_ids"]) or "—"
            lines.append(f"| {phase} | {s['requests']} | {s['successful']} | {s['failed']} | {checked} | {accepted_ids} | {fmt(s['accepted_unique_tasks_per_batch_second'])} | {fmt(s['batch_wall_ms_median'])} | {fmt(s['e2e_ms_median'])} | {fmt(s['e2e_ms_p95'])} | {fmt(s['ttft_ms_median'])} | {fmt(s['server_reported_decode_tokens_per_second_median'])} |")
        lines.append("")
    lines += ["*Token rates are shown only when supplied by the server as `timings.predicted_per_second`; client-derived rates are omitted because reasoning tokens and stream buffering can distort them. `batch wall` measures elapsed time for the complete request batch at that concurrency.", ""]
    lines += ["## Failures", ""]
    failures = [r for r in results if r.status != "ok"]
    if failures:
        lines += ["| Profile | Case | Phase | Error |", "|---|---|---|---|"]
        lines.extend(f"| {r.profile} | {r.case_id} | {r.phase} | {r.error or 'unknown'} |" for r in failures)
    else:
        lines.append("No request failures.")
    lines.append("")
    return "\n".join(lines)
