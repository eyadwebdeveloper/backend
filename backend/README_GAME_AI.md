# Game-session AI (Tier 3)

`aura_ai_kit/aura/game_ai.py` reads one finished Astro-Shield run and returns:
state (nominal / fatigue / tension / low_vigor / cabin_co2) with probabilities,
a 0-100 readiness, a strangeness score, top drivers, an "is it me or the cabin?"
probability, a personal-baseline comparison, a consecutive-run notice gate and
a next-run reaction-time trend.

Run: `pip install -r aura_ai_kit/requirements.txt && python server.py`
(trains the model in about 1 s, then builds the existing pipeline snapshots).

    POST http://127.0.0.1:8000/api/game/assess
    { "log": <SessionRecord>, "co2": 3.8, "history": [<earlier SessionRecords>] }

Frontend: `src/components/astro/ai.ts` (set `NEXT_PUBLIC_AURA_API` to change the URL).
If the backend is down the game keeps working and shows "AI coach is offline".

Tests: `cd aura_ai_kit && python -m pytest tests`

Honest limits: trained on simulated runs (like Tier 2a), so it validates the
pipeline, not medical accuracy. Decision support only, never a diagnosis.
