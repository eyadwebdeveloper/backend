"""AURA API with WSGI deployment support and a local development server."""
from __future__ import annotations

import argparse
import json
import errno
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
from pathlib import Path
from urllib.parse import unquote, urlsplit


BACKEND_DIR = Path(__file__).resolve().parent
AI_PROJECT_DIR = BACKEND_DIR / "aura_ai_kit"
RESULTS_DIR = AI_PROJECT_DIR / "results"
SNAPSHOTS_PATH = RESULTS_DIR / "aura_snapshots.json"

sys.path.insert(0, str(AI_PROJECT_DIR))
import run_pipeline  # noqa: E402
from aura.web_snapshot import CREW_TO_ASTRONAUT  # noqa: E402
from aura import game_ai  # noqa: E402

MAX_BODY_BYTES = 512 * 1024
_SNAPSHOT_LOCK_PATH = BACKEND_DIR / ".aura-snapshots.lock"

_initialization_lock = threading.Lock()
_game_model = None
_snapshots: dict | None = None


class InitializationError(RuntimeError):
    """A required model or snapshot could not be initialized."""


class SnapshotFileError(InitializationError):
    """The deployment snapshot file is missing or invalid."""


@contextmanager
def _snapshot_process_lock():
    lock_file = _SNAPSHOT_LOCK_PATH.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while not acquired:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            acquired = True
        yield
    finally:
        try:
            if acquired:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _validate_snapshots(snapshots) -> dict:
    if not isinstance(snapshots, dict):
        raise SnapshotFileError("Snapshot file must contain a JSON object")
    missing = sorted(set(CREW_TO_ASTRONAUT.values()) - set(snapshots))
    if missing:
        raise SnapshotFileError(
            f"Snapshot file is missing mapped crew: {missing}"
        )
    invalid = sorted(
        astro_id for astro_id in CREW_TO_ASTRONAUT.values()
        if not isinstance(snapshots[astro_id], dict)
        or snapshots[astro_id].get("astro_id") != astro_id
    )
    if invalid:
        raise SnapshotFileError(
            f"Snapshot file has invalid entries for crew: {invalid}"
        )
    return snapshots


def _load_snapshot_file() -> dict:
    if not SNAPSHOTS_PATH.is_file():
        raise SnapshotFileError(
            f"Generated snapshots are missing at {SNAPSHOTS_PATH}; "
            "run `python server.py --generate-snapshots` before reloading WSGI"
        )
    try:
        with SNAPSHOTS_PATH.open("r", encoding="utf-8") as file:
            snapshots = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotFileError(
            f"Could not read valid snapshots at {SNAPSHOTS_PATH}: {exc}"
        ) from exc
    return _validate_snapshots(snapshots)


def build_snapshots() -> dict:
    """Run the simulator pipeline once and load its validated snapshot output."""
    print("Running AURA simulator and models; this can take 10-30 seconds...",
          flush=True)
    started = time.perf_counter()
    generated = run_pipeline.main()
    if generated is None:
        raise SnapshotFileError("Pipeline did not return generated snapshots")
    snapshots = _validate_snapshots(generated)
    persisted = _load_snapshot_file()
    if persisted != snapshots:
        raise SnapshotFileError(
            "Pipeline snapshot file does not match the generated snapshots"
        )
    print(f"AURA snapshots ready in {time.perf_counter() - started:.1f}s.",
          flush=True)
    return persisted


def _get_game_model():
    global _game_model
    if _game_model is None:
        with _initialization_lock:
            if _game_model is None:
                try:
                    model = game_ai.get_model()
                    if model is None:
                        raise RuntimeError("Game AI returned no model")
                except Exception as exc:
                    raise InitializationError(
                        "Game AI model initialization failed"
                    ) from exc
                _game_model = model
    return _game_model


def _get_snapshots(*, generate_missing: bool = False) -> dict:
    global _snapshots
    if _snapshots is None:
        with _initialization_lock:
            if _snapshots is None:
                try:
                    with _snapshot_process_lock():
                        try:
                            snapshots = _load_snapshot_file()
                        except SnapshotFileError:
                            if not generate_missing:
                                raise
                            snapshots = build_snapshots()
                        snapshots = _validate_snapshots(snapshots)
                except Exception as exc:
                    raise InitializationError(
                        f"Snapshot initialization failed at {SNAPSHOTS_PATH}: "
                        f"{exc}"
                    ) from exc
                _snapshots = snapshots
    return _snapshots


def initialize(*, generate_missing_snapshots: bool = False) -> None:
    """Initialize models and load snapshots; generation is opt-in for local use."""
    _get_game_model()
    _get_snapshots(generate_missing=generate_missing_snapshots)


def generate_snapshots() -> dict:
    """Run and validate the snapshot pipeline under the cross-process lock."""
    global _snapshots
    with _initialization_lock:
        try:
            with _snapshot_process_lock():
                snapshots = build_snapshots()
        except Exception as exc:
            raise InitializationError(
                f"Snapshot generation failed at {SNAPSHOTS_PATH}: {exc}"
            ) from exc
        _snapshots = snapshots
        return snapshots


def _json_response(
    status: int,
    payload: dict,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> tuple[int, list[tuple[str, str]], bytes]:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        *extra_headers,
    ]
    return status, headers, body


def _method_not_allowed(
    allow: str,
) -> tuple[int, list[tuple[str, str]], bytes]:
    return _json_response(
        405,
        {"error": "Method not allowed"},
        (("Allow", allow),),
    )


def _read_json_object(
    content_length: str | None,
    read_body,
) -> dict | tuple[int, list[tuple[str, str]], bytes]:
    if content_length is None or content_length == "":
        return _json_response(413, {"error": "Body missing or too large"})
    if not isinstance(content_length, str) or re.fullmatch(r"[0-9]+", content_length) is None:
        return _json_response(400, {
            "error": "Bad request: CONTENT_LENGTH must be a non-negative integer"
        })

    try:
        length = int(content_length)
    except ValueError:
        return _json_response(400, {
            "error": "Bad request: CONTENT_LENGTH is too large to parse"
        })
    if length <= 0 or length > MAX_BODY_BYTES:
        return _json_response(413, {"error": "Body missing or too large"})
    if not callable(read_body):
        return _json_response(400, {
            "error": "Bad request: missing or invalid WSGI input stream"
        })

    try:
        raw_body = read_body(length)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return _json_response(400, {"error": f"Bad request: {exc}"})
    if not isinstance(raw_body, (bytes, bytearray)):
        return _json_response(400, {
            "error": "Bad request: input stream must return bytes"
        })
    if len(raw_body) != length:
        return _json_response(400, {
            "error": "Bad request: CONTENT_LENGTH does not match request body"
        })

    try:
        data = json.loads(bytes(raw_body).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _json_response(400, {"error": f"Bad request: {exc}"})
    if not isinstance(data, dict):
        return _json_response(400, {
            "error": "Bad request: JSON body must be an object"
        })
    return data


def _dispatch(
    method: str,
    request_path: str,
    content_length: str | None,
    read_body,
) -> tuple[int, list[tuple[str, str]], bytes]:
    path = urlsplit(request_path).path
    cors_headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]

    if method == "OPTIONS":
        return 204, [*cors_headers, ("Content-Length", "0")], b""

    if path == "/api/health":
        if method != "GET":
            return _method_not_allowed("GET, OPTIONS")
        try:
            model = _get_game_model()
            _get_snapshots()
        except InitializationError:
            logging.exception("AURA backend initialization failed")
            return _json_response(
                503, {"error": "Backend initialization failed"}
            )
        return _json_response(200, {
            "status": "ok",
            "simulated": True,
            "game_ai": {
                "ready": True,
                "holdout_accuracy": round(model.holdout_accuracy, 3),
            },
        })

    if path == "/api/game/assess":
        if method != "POST":
            return _method_not_allowed("POST, OPTIONS")
        data = _read_json_object(content_length, read_body)
        if not isinstance(data, dict):
            return data

        try:
            log = data.get("log")
            if not isinstance(log, dict):
                return _json_response(400, {
                    "error": "'log' (the session log) is required"
                })
            history = [
                item for item in (data.get("history") or [])
                if isinstance(item, dict)
            ][-10:]
            co2 = data.get("co2", game_ai.NORMAL_CO2)
            history_co2 = [
                item.get("co2", game_ai.NORMAL_CO2) for item in history
            ]
            try:
                model = _get_game_model()
            except InitializationError:
                logging.exception("AURA game AI initialization failed")
                return _json_response(
                    503, {"error": "Backend initialization failed"}
                )
            result = model.assess(log, float(co2), history, history_co2)
            return _json_response(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            return _json_response(400, {"error": f"Bad request: {exc}"})

    parts = path.strip("/").split("/")
    is_snapshot_path = (
        len(parts) == 4 and parts[0] == "api"
        and parts[1] == "astronauts" and parts[3] == "snapshot"
    )
    if is_snapshot_path:
        if method != "GET":
            return _method_not_allowed("GET, OPTIONS")
        crew_id = unquote(parts[2])
        astro_id = CREW_TO_ASTRONAUT.get(crew_id)
        if astro_id is None:
            return _json_response(404, {"error": "Unknown demo crew member"})
        try:
            snapshot = _get_snapshots().get(astro_id)
        except InitializationError:
            logging.exception("AURA snapshot initialization failed")
            return _json_response(
                503, {"error": "Backend initialization failed"}
            )
        if snapshot is None:
            return _json_response(404, {"error": "No snapshot for demo crew member"})
        return _json_response(200, snapshot)

    if method not in ("GET", "POST"):
        return _method_not_allowed("GET, POST, OPTIONS")
    return _json_response(404, {"error": "Not found"})


def application(environ, start_response):
    """WSGI entrypoint; pipeline snapshots and the game model initialize on demand."""
    method = environ.get("REQUEST_METHOD", "GET")
    if not isinstance(method, str):
        method = ""
    method = method.upper()
    path = environ.get("PATH_INFO", "/")
    content_length = environ.get("CONTENT_LENGTH")
    input_stream = environ.get("wsgi.input")
    read_body = getattr(input_stream, "read", None)

    status, headers, body = _dispatch(
        method, path, content_length, read_body
    )
    status_line = f"{status} {HTTPStatus(status).phrase}"
    start_response(status_line, headers)
    return [body]


class AuraHandler(BaseHTTPRequestHandler):
    def __getattr__(self, name):
        if name.startswith("do_"):
            return self._handle_request
        raise AttributeError(name)

    def _handle_request(self) -> None:
        status, headers, body = _dispatch(
            self.command,
            self.path,
            self.headers.get("Content-Length", "0"),
            self.rfile.read,
        )
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._handle_request()

    def do_POST(self) -> None:
        self._handle_request()

    def do_GET(self) -> None:
        self._handle_request()

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the AURA API server.")
    parser.add_argument(
        "--generate-snapshots",
        action="store_true",
        help="run the snapshot pipeline once and exit",
    )
    args = parser.parse_args(argv)
    if args.generate_snapshots:
        try:
            snapshots = generate_snapshots()
        except InitializationError as exc:
            print(f"Snapshot generation failed: {exc}", file=sys.stderr,
                  flush=True)
            raise SystemExit(1) from exc
        print(
            f"Generated and validated {len(snapshots)} snapshots at "
            f"{SNAPSHOTS_PATH}",
            flush=True,
        )
        return

    print("Training the game-session AI on simulated runs...", flush=True)
    initialize(generate_missing_snapshots=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AuraHandler)
    print("AURA demo API ready at http://127.0.0.1:8000", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AURA demo API.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
