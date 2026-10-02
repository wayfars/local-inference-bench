import json
import time

import pytest

from inference_bench import runner


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode()


def test_stream_parses_ttft_text_and_usage(monkeypatch):
    response = FakeResponse([
        b": keep-alive\n", sse({"choices": [{"delta": {"content": "hel"}}]}),
        sse({"choices": [{"delta": {"content": "lo"}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2},
             "timings": {"predicted_per_second": 24.5, "draft_n_accepted": 3, "draft_n": 5}}),
        b"data: [DONE]\n\n",
    ])
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: response)
    ttft, e2e, completion, prompt, text, finish, done, metrics = runner._request_one(
        {"base_url": "http://127.0.0.1:8000/v1", "model": "synthetic"},
        {"prompt": "safe"}, 2)
    assert 0 <= ttft <= e2e
    assert completion == 2
    assert prompt == 7
    assert text == "hello"
    assert finish == "stop"
    assert done is True
    assert metrics["predicted_per_second"] == 24.5
    assert metrics["draft_n_accepted"] == 3


def test_sse_usage_without_content_does_not_set_ttft(monkeypatch):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: FakeResponse([
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 0}}), b"data: [DONE]\n"
    ]))
    ttft, _e2e, completion, _prompt, text, _finish, _done, _metrics = runner._request_one(
        {"base_url": "http://127.0.0.1/v1", "model": "x"}, {"prompt": "p"}, 1)
    assert ttft is None
    assert completion == 0
    assert text == ""


def test_acceptance_is_explicit_phrase_check_only():
    assert runner._check({"expected_any": ["blue", "azure"]}, "The answer is blue.") == (True, "heuristic substring matched")
    assert runner._check({"expected_any": ["blue"]}, "not sure") == (False, "heuristic substring did not match")
    assert runner._check({}, "anything") == (None, None)


def test_exact_grader_does_not_accept_explanations_or_negated_answers():
    assert runner._check({"expected_exact": "42"}, "42") == (True, "exact output match")
    assert runner._check({"expected_exact": "42"}, "Answer: 42") == (False, "exact output mismatch")
    assert runner._check({"expected_exact": "a  b"}, "a\nb") == (True, "exact output match")
    assert runner._check({"expected_exact": "42"}, "not 42") == (False, "exact output mismatch")
    assert runner._check({"expected_any": ["42"]}, "not 42") == (True, "heuristic substring matched")


def test_json_grader_requires_full_valid_json_and_structural_equality():
    case = {"expected_json": {"ok": True, "count": 2}}
    assert runner._check(case, '{"count":2,"ok":true}') == (True, "JSON value match")
    assert runner._check(case, '{"ok":true,"count":3}') == (False, "JSON value mismatch")
    assert runner._check({"expected_json": {"ok": True}}, '{"ok":1}') == (False, "JSON value mismatch")
    assert runner._check(case, 'Here: {"ok":true,"count":2}') == (False, "response is not valid JSON")
    assert runner._check(case, '{"ok":true,"ok":true,"count":2}') == (False, "response is not valid JSON")
    assert runner._check(case, '{"ok":NaN,"count":2}') == (False, "response is not valid JSON")


def test_summary_keeps_missing_metrics_unavailable():
    result = runner.Result("r", "p", "c", 1, "measured", 1, "ok", True, "matched:x",
                          10.0, 20.0, None, None, None, 1)
    result.concurrency_batch_ms = 44.0
    failed = runner.Result("r", "p", "c2", 1, "measured", 1, "error", False, None,
                           None, None, None, None, None, None, "timeout")
    summary = runner.summarize([result, failed])["profiles"]["p"]["measured@concurrency=1"]
    assert summary["ttft_ms_median"] == 10
    assert summary["observed_post_ttft_completion_tokens_per_second_median"] is None
    assert summary["acceptance_rate"] == .5
    assert summary["failed"] == 1
    assert summary["batch_wall_ms_median"] == 44
    assert summary["server_reported_decode_tokens_per_second_median"] is None
    assert summary["accepted_requests_per_batch_second"] == pytest.approx(22.7272727)
    assert summary["accepted_unique_tasks_per_batch_second"] == pytest.approx(22.7272727)
    # Endpoint errors with a grader remain in `checked` and count as rejected.
    assert summary["checked"] == 2


def test_manifest_drops_relative_weight_paths_but_keeps_model_repo_id():
    assert runner._safe_manifest_value("models/private.gguf") is None
    assert runner._safe_manifest_value("models/private") is None
    assert runner._safe_manifest_value("../weights") is None
    assert runner._safe_manifest_value("org/model-name") == "org/model-name"
    assert runner._safe_artifact_reference("org/model-name") is None
    assert runner._safe_artifact_reference("hf:org/model-name") == "hf:org/model-name"
    assert runner._safe_artifact_reference("models/private") is None


def test_summary_separates_concurrency_levels():
    one = runner.Result("r", "p", "a", 1, "measured", 1, "ok", True, "ok", 1, 2, 1, 1, None, 1)
    two = runner.Result("r", "p", "b", 1, "measured", 2, "ok", True, "ok", 1, 2, 1, 1, None, 1)
    groups = runner.summarize([one, two])["profiles"]["p"]
    assert set(groups) == {"measured@concurrency=1", "measured@concurrency=2"}


def test_incomplete_stream_is_never_accepted(monkeypatch):
    monkeypatch.setattr(runner, "_request_one", lambda *args: (1, 2, 1, 1, "42", "length", False, {}))
    result = runner._call({"name": "p"}, {"id": "c", "prompt": "p", "expected_exact": "42"},
                          run_id="r", repetition=1, phase="measured", concurrency=1,
                          timeout_s=1, include_text=False)
    assert result.status == "incomplete"
    assert result.accepted is False


def test_server_engagement_and_decode_metrics_are_recorded(monkeypatch):
    monkeypatch.setattr(runner, "_request_one", lambda *args: (1, 3, 4, 2, "ok", "stop", True,
                      {"predicted_per_second": 12.5, "draft_n_accepted": 2, "draft_n": 4}))
    result = runner._call({"name": "p"}, {"id": "c", "prompt": "p", "expected_exact": "ok"},
                          run_id="r", repetition=1, phase="measured", concurrency=1,
                          timeout_s=1, include_text=False)
    assert result.server_decode_tokens_per_second == 12.5
    assert result.observed_post_ttft_completion_tokens_per_second is None
    assert result.draft_tokens_accepted == 2


def test_reasoning_inclusive_usage_and_single_visible_chunk_do_not_create_client_rate(monkeypatch):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: FakeResponse([
        sse({"choices": [{"delta": {"content": "answer"}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 4, "completion_tokens": 1200},
             "timings": {"predicted_per_second": 39}}),
        b"data: [DONE]\n",
    ]))
    result = runner._call({"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x"},
                          {"id": "c", "prompt": "p", "expected_exact": "answer"},
                          run_id="r", repetition=1, phase="measured", concurrency=1,
                          timeout_s=1, include_text=False)
    assert result.status == "ok"
    assert result.completion_tokens == 1200
    assert result.observed_post_ttft_completion_tokens_per_second is None
    assert result.server_decode_tokens_per_second == 39
    summary = runner.summarize([result])
    phase = summary["profiles"]["p"]["measured@concurrency=1"]
    assert phase["observed_post_ttft_completion_tokens_per_second_median"] is None
    report = runner.render_report({"run_id": "r", "synthetic_demo": False,
                                   "created_at": "test", "host": {},
                                   "benchmark_settings": {}, "profiles": [{"name": "p"}]},
                                  summary, [result])
    assert "Post-TTFT" not in report
    assert "Server rate tok/s" in report
    assert "39.00" in report


def test_config_env_expansion_is_fail_closed(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_SECRET", raising=False)
    with pytest.raises(runner.BenchmarkError, match="not set"):
        runner.expand_env("Bearer ${MISSING_TEST_SECRET}")
    monkeypatch.setenv("BENCH_TEST_KEY", "test-secret")
    assert runner.expand_env({"headers": {"Authorization": "Bearer ${BENCH_TEST_KEY}"}}) == {
        "headers": {"Authorization": "Bearer test-secret"}}


def test_env_expansion_is_single_pass_and_cycles_cannot_recurse(monkeypatch):
    monkeypatch.setenv("BENCH_A", "${BENCH_B}")
    monkeypatch.setenv("BENCH_B", "resolved")
    assert runner.expand_env("${BENCH_A}") == "${BENCH_B}"
    assert runner.expand_env("${BENCH_B}") == "resolved"
    with pytest.raises(runner.BenchmarkError, match="malformed"):
        runner.expand_env("prefix ${BROKEN")


def test_manifest_omits_endpoint_and_headers_and_report_marks_demo(tmp_path, monkeypatch):
    config = {
        "run_id": "synthetic-only", "synthetic_demo": True, "benchmark_kind": "synthetic",
        "cases": "cases.json", "repetitions": 1, "concurrency": [1],
        "profiles": [{"name": "fake", "base_url": "http://user:pass@127.0.0.1:3/v1",
                      "model": "synthetic", "headers": {"Authorization": "SECRET"},
                      "server_commit": "abc123", "request": {"temperature": 0}}],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "cases.json").write_text(json.dumps([{"id": "task-1", "prompt": "safe", "expected_any": ["ok"]}]))
    monkeypatch.setattr(runner, "_call", lambda profile, case, **kw: runner.Result(
        kw["run_id"], profile["name"], case["id"], kw["repetition"], kw["phase"],
        kw["concurrency"], "ok", True, "matched:ok", 1, 2, None, None, None, 2))
    result_path, report_path = runner.execute(config_path, tmp_path / "out")
    saved = result_path.read_text()
    report = report_path.read_text()
    assert "SECRET" not in saved
    assert "user:pass" not in saved
    assert "http://" not in saved
    assert "SYNTHETIC DEMO" in report


def test_malformed_sse_is_reported_without_echoing_payload():
    with pytest.raises(runner.BenchmarkError, match="malformed SSE"):
        runner._parse_sse_line(b"data: {not-json}\n")
    with pytest.raises(runner.BenchmarkError, match="malformed SSE"):
        runner._parse_sse_line(b'data: {"x":1,"x":2}\n')
    with pytest.raises(runner.BenchmarkError, match="malformed SSE"):
        runner._parse_sse_line(b'data: {"x":NaN}\n')


@pytest.mark.parametrize("lines,expected", [
    ([sse({"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]})], "incomplete"),
    ([sse({"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}), b"data: [DONE]\n"], "incomplete"),
    ([sse({"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}), b"data: [DONE]\n"], "incomplete"),
])
def test_stream_missing_done_empty_or_truncated_is_never_accepted(monkeypatch, lines, expected):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: FakeResponse(lines))
    result = runner._call({"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x"},
                          {"id": "c", "prompt": "p", "expected_any": ["partial"]},
                          run_id="r", repetition=1, phase="measured", concurrency=1,
                          timeout_s=1, include_text=False)
    assert result.status == expected
    assert result.accepted is False


@pytest.mark.parametrize("finish", ["length", "tool_calls", "content_filter", "future_reason"])
def test_non_stop_finish_reasons_are_never_accepted(monkeypatch, finish):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: FakeResponse([
        sse({"choices": [{"delta": {"content": "looks good"}, "finish_reason": finish}]}),
        b"data: [DONE]\n",
    ]))
    result = runner._call({"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x"},
                          {"id": "c", "prompt": "p", "expected_exact": "looks good"},
                          run_id="r", repetition=1, phase="measured", concurrency=1,
                          timeout_s=1, include_text=False)
    assert result.status == "incomplete"
    assert result.accepted is False
    if finish == "future_reason":
        assert result.finish_reason == "unknown"


@pytest.mark.parametrize("payload", [
    {"choices": "broken"},
    {"choices": [{"delta": "broken", "finish_reason": "stop"}]},
    {"choices": [{"delta": {"content": [{"text": 1}]}, "finish_reason": "stop"}]},
])
def test_malformed_sse_shapes_become_request_errors(monkeypatch, payload):
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda request, timeout: FakeResponse([
        sse(payload), b"data: [DONE]\n",
    ]))
    result = runner._call({"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x"},
                          {"id": "c", "prompt": "p"}, run_id="r", repetition=1,
                          phase="measured", concurrency=1, timeout_s=1, include_text=False)
    assert result.status == "error"
    assert result.error == "BenchmarkError"


def test_manifest_drops_paths_emails_and_expanded_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_PRIVATE", "s3cr3t-value")
    private_model_path = "/" + "mnt" + "/models/private.gguf"
    private_host_path = "/" + "home" + "/alice/private machine"
    config = {
        "run_id": "safe-run", "benchmark_kind": "synthetic", "synthetic_demo": True,
        "cases": "cases.json", "repetitions": 1, "concurrency": [1],
        "host": {"label": "local", "hardware": private_host_path, "os": "Linux"},
        "profiles": [{"name": "p", "base_url": "http://127.0.0.1:3/v1", "model": "x",
                      "headers": {"Authorization": "Bearer ${BENCH_PRIVATE}"},
                      "model_artifact": private_model_path, "provider": "local",
                      "request": {"temperature": 0, "max_tokens": 32},
                      "speculative_engagement": {"signal": "token", "untrusted": "private phrase"}}],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "cases.json").write_text(json.dumps([{"id": "case-1", "prompt": "safe"}]))
    monkeypatch.setattr(runner, "_call", lambda profile, case, **kw: runner.Result(
        kw["run_id"], profile["name"], case["id"], kw["repetition"], kw["phase"], kw["concurrency"],
        "ok", None, None, 1, 2, None, None, None, 2))
    result_path, _ = runner.execute(config_path, tmp_path / "out")
    saved = result_path.read_text()
    assert "s3cr3t-value" not in saved
    assert private_host_path not in saved
    assert private_model_path not in saved
    assert "127.0.0.1" not in saved
    manifest = json.loads(saved)["manifest"]
    profile = manifest["profiles"][0]
    assert profile["request_settings"] == {"temperature": 0, "max_tokens": 32}
    assert profile["unconfigured_server_defaults"] == "unavailable"
    assert "model_artifact" not in profile
    assert "untrusted" not in profile.get("speculative_engagement", {})
    assert len(manifest["workload_sha256"]) == 64
    assert len(manifest["runner_sha256"]) == 64


def test_grader_configuration_rejects_conflicting_graders_before_execution(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"run_id": "bad-case", "cases": "cases.json",
        "profiles": [{"name": "p"}]}))
    (tmp_path / "cases.json").write_text(json.dumps([{"id": "case-1", "prompt": "p",
        "expected_exact": "ok", "expected_any": ["ok"]}]))
    with pytest.raises(runner.BenchmarkError, match="at most one grader"):
        runner.execute(config_path, tmp_path / "out")


@pytest.mark.parametrize("profile_patch,expected", [
    ({"request": {"stop": "private phrase"}}, "unsupported request parameter"),
    ({"request_settings": {"temperature": 0}}, "obsolete request_settings"),
])
def test_unsupported_or_ambiguous_settings_fail_before_requests(tmp_path, monkeypatch, profile_patch, expected):
    config_path = tmp_path / "config.json"
    profile = {"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x", **profile_patch}
    config_path.write_text(json.dumps({"run_id": "bad-settings", "cases": "cases.json",
        "profiles": [profile]}))
    (tmp_path / "cases.json").write_text(json.dumps([{"id": "case-1", "prompt": "p"}]))
    monkeypatch.setattr(runner, "_call", lambda *args, **kwargs: pytest.fail("request must not run"))
    with pytest.raises(runner.BenchmarkError, match=expected):
        runner.execute(config_path, tmp_path / "out")


def test_measurement_batch_respects_concurrency_and_records_wall_time(tmp_path, monkeypatch):
    import threading

    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_call(profile, case, **kw):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.02)
        with lock:
            active -= 1
        return runner.Result(kw["run_id"], profile["name"], case["id"], kw["repetition"],
                             kw["phase"], kw["concurrency"], "ok", True, "exact output match",
                             1, 2, 1, 1, None, 1)

    monkeypatch.setattr(runner, "_call", fake_call)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "run_id": "concurrency-check", "cases": "cases.json", "repetitions": 1,
        "concurrency": [2], "profiles": [{"name": "p", "base_url": "http://127.0.0.1/v1", "model": "x"}],
    }))
    (tmp_path / "cases.json").write_text(json.dumps([{"id": f"case-{n}", "prompt": "p"} for n in range(3)]))
    result_path, _ = runner.execute(config_path, tmp_path / "out")
    summary = json.loads(result_path.read_text())["summary"]["profiles"]["p"]["measured@concurrency=2"]
    assert maximum == 2
    assert summary["requests"] == 3
    assert summary["batch_wall_ms_median"] >= 30
