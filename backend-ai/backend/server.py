"""Local-only HTTP API for the AURA simulated-model website demo."""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def build_snapshots() -> dict:
    print("Running AURA simulator and models; this can take 10-30 seconds...",
          flush=True)
    started = time.perf_counter()
    run_pipeline.main()
    if not SNAPSHOTS_PATH.is_file():
        raise FileNotFoundError(f"Pipeline did not create {SNAPSHOTS_PATH}")
    with SNAPSHOTS_PATH.open("r", encoding="utf-8") as file:
        snapshots = json.load(file)
    missing = sorted(set(CREW_TO_ASTRONAUT.values()) - set(snapshots))
    if missing:
        raise RuntimeError(f"Pipeline snapshots are missing mapped crew: {missing}")
    print(f"AURA snapshots ready in {time.perf_counter() - started:.1f}s.",
          flush=True)
    return snapshots


class AuraHandler(BaseHTTPRequestHandler):
    snapshots: dict = {}

    def _cors(self) -> None:
        # local-only demo API: the Next.js dev server runs on another port
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/api/game/assess":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "Body missing or too large"})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            log = data.get("log")
            if not isinstance(log, dict):
                self._send_json(400, {"error": "'log' (the session log) is required"})
                return
            history = [h for h in (data.get("history") or []) if isinstance(h, dict)][-10:]
            co2 = data.get("co2", game_ai.NORMAL_CO2)
            history_co2 = [h.get("co2", game_ai.NORMAL_CO2) for h in history]
            result = game_ai.get_model().assess(log, float(co2), history, history_co2)
            self._send_json(200, result)
        except (ValueError, TypeError, KeyError) as exc:
            self._send_json(400, {"error": f"Bad request: {exc}"})

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            model = game_ai.get_model()
            self._send_json(200, {"status": "ok", "simulated": True,
                                  "game_ai": {"ready": True,
                                              "holdout_accuracy": round(model.holdout_accuracy, 3)}})
            return

        parts = path.strip("/").split("/")
        if (len(parts) != 4 or parts[0] != "api"
                or parts[1] != "astronauts" or parts[3] != "snapshot"):
            self._send_json(404, {"error": "Not found"})
            return

        crew_id = unquote(parts[2])
        astro_id = CREW_TO_ASTRONAUT.get(crew_id)
        if astro_id is None:
            self._send_json(404, {"error": "Unknown demo crew member"})
            return

        snapshot = self.snapshots.get(astro_id)
        if snapshot is None:
            self._send_json(404, {"error": "No snapshot for demo crew member"})
            return
        self._send_json(200, snapshot)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def main() -> None:
    print("Training the game-session AI on simulated runs...", flush=True)
    game_ai.get_model()
    AuraHandler.snapshots = build_snapshots()
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
