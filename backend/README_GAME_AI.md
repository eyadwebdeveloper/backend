# Game-session AI (Tier 3)

`aura_ai_kit/aura/game_ai.py` reads one finished Astro-Shield run and returns:
state (nominal / fatigue / tension / low_vigor / cabin_co2) with probabilities,
a 0-100 readiness, a strangeness score, top drivers, an "is it me or the cabin?"
probability, a personal-baseline comparison, a consecutive-run notice gate and
a next-run reaction-time trend.

Local development: `pip install -r aura_ai_kit/requirements.txt && python server.py`
(trains the model in about 1 s and generates snapshots only if the existing
snapshot file is missing or invalid).

    POST http://127.0.0.1:8000/api/game/assess
    { "log": <SessionRecord>, "co2": 3.8, "history": [<earlier SessionRecords>] }

## WSGI deployment (PythonAnywhere)

The WSGI callable is `application` in `server.py`. It exposes the same routes
as the local server:

    GET  /api/health
    GET  /api/astronauts/<crew-id>/snapshot
    POST /api/game/assess
    OPTIONS (CORS preflight)

The repository's backend directory is `backend-ai/backend`; the entrypoint is
`backend-ai/backend/server.py`. Deploy that directory at
`/home/<username>/backend-ai/backend` (replace `<username>` with the account
name). In the PythonAnywhere Web tab, create a **Manual Configuration** app
with the same Python version as its virtualenv. Python 3.12 is available on
PythonAnywhere's `innit` system image. For example, create a Python 3.12
virtualenv and select it in the Web tab:

```sh
mkvirtualenv --python=/usr/local/bin/python3.12 aura-backend
workon aura-backend
python -m pip install --no-cache-dir -r /home/<username>/backend-ai/backend/aura_ai_kit/requirements-production.txt
```

Before reloading the Web app, generate the deployment snapshot from that same
backend directory and virtualenv:

```sh
cd /home/<username>/backend-ai/backend
workon aura-backend
python server.py --generate-snapshots
```

This production dependency set includes NumPy, pandas, scikit-learn, SciPy, and
PyYAML, but omits pytest and matplotlib. The deployment generation command
keeps metrics and snapshot output but skips only the optional report and chart
files; `python server.py` and `python run_pipeline.py` still generate them by
default when using the full `aura_ai_kit/requirements.txt` (run the pipeline
directly from `aura_ai_kit`).

The command runs the configured simulator pipeline once, validates the result,
and exits nonzero with an error if generation or validation fails. It writes
`aura_ai_kit/results/aura_snapshots.json`, using the YAML configuration and
paths relative to `aura_ai_kit/run_pipeline.py`. Run it after deploying code
or changing pipeline configuration, then reload the Web app. The account must
be able to write to the backend directory and `aura_ai_kit/results`.

Use this in the PythonAnywhere WSGI configuration file:

```python
import sys

backend_dir = "/home/<username>/backend-ai/backend"
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from server import application
```

WSGI workers only load and validate the generated snapshot file; they never run
the expensive pipeline. Loading is lazy on health or snapshot requests. A
missing, unreadable, malformed, or incomplete file makes health return 503 and
is logged as an initialization error. The local `python server.py` command
still initializes before listening on `127.0.0.1:8000`, automatically generating
snapshots if no valid file is present. Local generation and the explicit
deployment command use the per-process thread lock and cross-process lock at
`backend-ai/backend/.aura-snapshots.lock`; WSGI readers use that same lock, so
they cannot read a partially written snapshot.

The main AURA site reads `VITE_AURA_API_BASE` (set it at build time to
`https://<username>.pythonanywhere.com/api` when the site is hosted separately).
The Astro-Shield game reads `NEXT_PUBLIC_AURA_API` (set it to
`https://<username>.pythonanywhere.com`). Both still use the same backend routes;
the existing wildcard CORS response is retained.

The first worker startup runs the simulator/model pipeline and can take
10-30 seconds. Ensure the PythonAnywhere account has sufficient memory and
startup allowance. PyTorch is optional; without it the existing ridge
forecaster fallback is used. No PythonAnywhere-specific dependency is required.

Frontend: `src/components/astro/ai.ts` (set `NEXT_PUBLIC_AURA_API` to change the URL).
If the backend is down the game keeps working and shows "AI coach is offline".

Tests: `python -m pytest -q tests aura_ai_kit/tests` (from `backend-ai/backend`)

Honest limits: trained on simulated runs (like Tier 2a), so it validates the
pipeline, not medical accuracy. Decision support only, never a diagnosis.
