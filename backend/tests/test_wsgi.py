import io
import json
import multiprocessing
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
import server


_UNSET = object()


def valid_snapshots():
    return {
        astro_id: {"astro_id": astro_id, "simulated": True}
        for astro_id in server.CREW_TO_ASTRONAUT.values()
    }


def _snapshot_process_worker(
    lock_path, snapshot_path, barrier, active, maximum, calls
):
    import server as process_server

    process_server._SNAPSHOT_LOCK_PATH = pathlib.Path(lock_path)
    process_server.SNAPSHOTS_PATH = pathlib.Path(snapshot_path)

    def build():
        with active.get_lock():
            active.value += 1
            calls.value += 1
            maximum.value = max(maximum.value, active.value)
        time.sleep(0.2)
        with active.get_lock():
            active.value -= 1
        return valid_snapshots()

    process_server.build_snapshots = build
    barrier.wait(timeout=30)
    process_server._get_snapshots(generate_missing=True)


@pytest.fixture(autouse=True)
def clear_server_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_game_model", None)
    monkeypatch.setattr(server, "_snapshots", None)
    monkeypatch.setattr(
        server, "SNAPSHOTS_PATH", tmp_path / "aura_snapshots.json"
    )
    monkeypatch.setattr(
        server, "_SNAPSHOT_LOCK_PATH", tmp_path / ".aura-snapshots.lock"
    )


def request(
    path,
    method="GET",
    payload=_UNSET,
    *,
    content_length=_UNSET,
    wsgi_input=_UNSET,
):
    body = b"" if payload is _UNSET else (
        payload if isinstance(payload, bytes)
        else json.dumps(payload).encode("utf-8")
    )
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
    }
    if content_length is _UNSET:
        environ["CONTENT_LENGTH"] = str(len(body))
    elif content_length is not None:
        environ["CONTENT_LENGTH"] = content_length
    if wsgi_input is _UNSET:
        environ["wsgi.input"] = io.BytesIO(body)
    elif wsgi_input is not None:
        environ["wsgi.input"] = wsgi_input

    captured = {}

    def start_response(status, headers):
        captured["status"] = int(status.split(" ", 1)[0])
        captured["headers"] = dict(headers)

    response = b"".join(server.application(environ, start_response))
    return captured["status"], captured["headers"], response


def test_health_returns_model_readiness_and_cors(monkeypatch):
    monkeypatch.setattr(server, "_snapshots", valid_snapshots())
    status, headers, body = request("/api/health")

    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert json.loads(body)["game_ai"]["ready"] is True


def test_wsgi_loads_existing_snapshot_without_running_pipeline(
    monkeypatch, tmp_path
):
    snapshots = valid_snapshots()
    snapshot_path = tmp_path / "aura_snapshots.json"
    snapshot_path.write_text(json.dumps(snapshots), encoding="utf-8")
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)

    def unexpected_pipeline_call():
        raise AssertionError("WSGI must not run the snapshot pipeline")

    monkeypatch.setattr(server.run_pipeline, "main", unexpected_pipeline_call)
    first = request("/api/astronauts/cmdr/snapshot")
    second = request("/api/astronauts/cmdr/snapshot")

    assert first[0] == second[0] == 200
    assert json.loads(first[2]) == snapshots["A101"]
    assert json.loads(second[2]) == snapshots["A101"]


def test_snapshot_initialization_is_safe_for_concurrent_threads(monkeypatch):
    calls = []
    snapshots = valid_snapshots()

    def build():
        calls.append(True)
        time.sleep(0.02)
        return snapshots

    monkeypatch.setattr(server, "build_snapshots", build)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda _: server._get_snapshots(generate_missing=True), range(8)
        ))

    assert len(calls) == 1
    assert all(result is snapshots for result in results)


def test_snapshot_generation_is_serialized_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    active = context.Value("i", 0)
    maximum = context.Value("i", 0)
    calls = context.Value("i", 0)
    barrier = context.Barrier(2)
    lock_path = tmp_path / ".cross-process-snapshots.lock"
    snapshot_path = tmp_path / "aura_snapshots.json"
    workers = [
        context.Process(
            target=_snapshot_process_worker,
            args=(
                str(lock_path),
                str(snapshot_path),
                barrier,
                active,
                maximum,
                calls,
            ),
        )
        for _ in range(2)
    ]

    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join()

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert calls.value == 2
    assert maximum.value == 1


def test_game_assessment_accepts_session_log():
    status, headers, body = request(
        "/api/game/assess", "POST", {"log": {}, "co2": 3.4, "history": []}
    )

    result = json.loads(body)
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert {"state", "readiness", "probabilities", "cabin"} <= result.keys()


def test_options_returns_cors_preflight_response():
    status, headers, body = request("/api/game/assess", "OPTIONS")

    assert status == 204
    assert body == b""
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
    assert headers["Access-Control-Allow-Headers"] == "Content-Type"


def test_unknown_endpoint_returns_json_404_with_cors():
    status, headers, body = request("/api/not-a-route")

    assert status == 404
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body) == {"error": "Not found"}


@pytest.mark.parametrize(
    ("content_length", "expected_status"),
    [
        (None, 413),
        ("", 413),
        ("not-a-number", 400),
        ("-1", 400),
        ("9" * 5000, 400),
    ],
)
def test_invalid_or_missing_content_length_is_json_error(
    content_length, expected_status
):
    status, headers, body = request(
        "/api/game/assess",
        "POST",
        b"{}",
        content_length=content_length,
        wsgi_input=object(),
    )

    assert status == expected_status
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body)["error"]


def test_missing_or_invalid_wsgi_input_is_json_error():
    missing = request(
        "/api/game/assess", "POST", b"{}", wsgi_input=None
    )
    invalid = request(
        "/api/game/assess", "POST", b"{}", wsgi_input=object()
    )

    assert missing[0] == invalid[0] == 400
    assert "input stream" in json.loads(missing[2])["error"]
    assert "input stream" in json.loads(invalid[2])["error"]


def test_truncated_body_and_invalid_json_return_bad_request():
    truncated = request(
        "/api/game/assess",
        "POST",
        content_length="2",
        wsgi_input=io.BytesIO(b"{"),
    )
    invalid_json = request("/api/game/assess", "POST", b"{")

    assert truncated[0] == invalid_json[0] == 400
    assert "CONTENT_LENGTH" in json.loads(truncated[2])["error"]
    assert json.loads(invalid_json[2])["error"].startswith("Bad request:")


@pytest.mark.parametrize("payload", [[], None, "not an object", 7])
def test_non_object_json_payload_returns_bad_request(payload):
    status, _, body = request("/api/game/assess", "POST", payload)

    assert status == 400
    assert json.loads(body)["error"] == "Bad request: JSON body must be an object"


def test_missing_session_log_returns_existing_error():
    status, _, body = request("/api/game/assess", "POST", {})

    assert status == 400
    assert json.loads(body) == {
        "error": "'log' (the session log) is required"
    }


def test_unsupported_method_returns_405_and_allow_header():
    status, headers, body = request("/api/game/assess", "DELETE")

    assert status == 405
    assert headers["Allow"] == "POST, OPTIONS"
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body) == {"error": "Method not allowed"}


def test_model_initialization_failure_can_retry(monkeypatch):
    calls = []
    model = object()

    def fail_then_succeed():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("training failed")
        return model

    monkeypatch.setattr(server.game_ai, "get_model", fail_then_succeed)
    with pytest.raises(server.InitializationError):
        server._get_game_model()

    assert server._game_model is None
    assert server._get_game_model() is model
    assert len(calls) == 2


def test_snapshot_initialization_failure_can_retry(monkeypatch):
    calls = []
    snapshots = valid_snapshots()

    def fail_then_succeed():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("pipeline failed")
        return snapshots

    monkeypatch.setattr(server, "build_snapshots", fail_then_succeed)
    with pytest.raises(server.InitializationError):
        server._get_snapshots(generate_missing=True)

    assert server._snapshots is None
    assert server._get_snapshots(generate_missing=True) is snapshots
    assert len(calls) == 2


def test_health_does_not_report_ready_after_model_initialization_failure(
    monkeypatch,
):
    def fail():
        raise RuntimeError("training failed")

    monkeypatch.setattr(server.game_ai, "get_model", fail)
    status, headers, body = request("/api/health")

    assert status == 503
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert json.loads(body) == {"error": "Backend initialization failed"}


def test_snapshot_initialization_failure_returns_service_unavailable():
    status, _, body = request("/api/astronauts/cmdr/snapshot")

    assert status == 503
    assert json.loads(body) == {"error": "Backend initialization failed"}


def test_health_does_not_report_ready_after_snapshot_initialization_failure():
    status, _, body = request("/api/health")

    assert status == 503
    assert json.loads(body) == {"error": "Backend initialization failed"}


def test_wsgi_initialize_loads_existing_file_without_running_pipeline(
    monkeypatch, tmp_path
):
    snapshots = valid_snapshots()
    snapshot_path = tmp_path / "aura_snapshots.json"
    snapshot_path.write_text(json.dumps(snapshots), encoding="utf-8")
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)
    monkeypatch.setattr(server.game_ai, "get_model", lambda: object())

    def unexpected_pipeline_call():
        raise AssertionError("WSGI initialization must not run the pipeline")

    monkeypatch.setattr(server.run_pipeline, "main", unexpected_pipeline_call)
    server.initialize()

    assert server._snapshots == snapshots


def test_wsgi_initialize_fails_when_snapshot_file_is_missing(monkeypatch):
    monkeypatch.setattr(server.game_ai, "get_model", lambda: object())

    def unexpected_pipeline_call():
        raise AssertionError("WSGI initialization must not run the pipeline")

    monkeypatch.setattr(server.run_pipeline, "main", unexpected_pipeline_call)
    with pytest.raises(
        server.InitializationError, match="--generate-snapshots"
    ):
        server.initialize()
    assert server._snapshots is None


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{not-json", "Could not read valid snapshots"),
        ("[]", "JSON object"),
        (json.dumps({"A101": {"astro_id": "A101"}}), "missing mapped crew"),
        (
            json.dumps({
                **valid_snapshots(),
                "A101": {"astro_id": "not-A101"},
            }),
            "invalid entries",
        ),
    ],
)
def test_wsgi_initialize_fails_for_invalid_or_incomplete_snapshots(
    monkeypatch, tmp_path, contents, message
):
    snapshot_path = tmp_path / "aura_snapshots.json"
    snapshot_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)
    monkeypatch.setattr(server.game_ai, "get_model", lambda: object())

    def unexpected_pipeline_call():
        raise AssertionError("WSGI initialization must not run the pipeline")

    monkeypatch.setattr(server.run_pipeline, "main", unexpected_pipeline_call)
    with pytest.raises(server.InitializationError, match=message):
        server.initialize()
    assert server._snapshots is None


def test_local_initialization_generates_missing_snapshots(monkeypatch, tmp_path):
    snapshots = valid_snapshots()
    calls = []
    snapshot_path = tmp_path / "aura_snapshots.json"
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)
    monkeypatch.setattr(server.game_ai, "get_model", lambda: object())

    def generate():
        calls.append(True)
        snapshot_path.write_text(json.dumps(snapshots), encoding="utf-8")
        return snapshots

    monkeypatch.setattr(server.run_pipeline, "main", generate)
    server.initialize(generate_missing_snapshots=True)

    assert server._snapshots == snapshots
    assert len(calls) == 1


def test_deployment_generation_command_runs_pipeline_once(
    monkeypatch, tmp_path, capsys
):
    snapshots = valid_snapshots()
    calls = []
    snapshot_path = tmp_path / "aura_snapshots.json"
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)

    def generate():
        calls.append(True)
        snapshot_path.write_text(json.dumps(snapshots), encoding="utf-8")
        return snapshots

    monkeypatch.setattr(server.run_pipeline, "main", generate)
    server.main(["--generate-snapshots"])

    assert len(calls) == 1
    assert server._snapshots == snapshots
    assert "Generated and validated" in capsys.readouterr().out


def test_generation_command_does_not_accept_preexisting_file_without_new_output(
    monkeypatch, tmp_path, capsys
):
    snapshot_path = tmp_path / "aura_snapshots.json"
    snapshot_path.write_text(json.dumps(valid_snapshots()), encoding="utf-8")
    monkeypatch.setattr(server, "SNAPSHOTS_PATH", snapshot_path)
    monkeypatch.setattr(server.run_pipeline, "main", lambda: None)

    with pytest.raises(SystemExit) as error:
        server.main(["--generate-snapshots"])

    assert error.value.code == 1
    assert "did not return generated snapshots" in capsys.readouterr().err


def test_deployment_generation_command_fails_clearly(monkeypatch, capsys):
    def fail():
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(server.run_pipeline, "main", fail)
    with pytest.raises(SystemExit) as error:
        server.main(["--generate-snapshots"])

    assert error.value.code == 1
    assert "Snapshot generation failed" in capsys.readouterr().err
