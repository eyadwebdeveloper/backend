import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from aura import game_ai as g


def test_holdout_accuracy():
    assert g.get_model().holdout_accuracy > 0.9


def test_states_are_recovered():
    m, rng = g.get_model(), np.random.default_rng(1)
    for state in g.STATES:
        hits = 0
        for _ in range(40):
            log, co2 = g.simulate_session(state, rng)
            hits += m.assess(log, co2, [])["state"] == state
        assert hits >= 32, state


def test_cabin_attribution():
    m, rng = g.get_model(), np.random.default_rng(2)
    log, co2 = g.simulate_session("cabin_co2", rng)
    assert m.assess(log, co2, [])["cabin"]["verdict"] == "cabin"
    log, co2 = g.simulate_session("fatigue", rng, co2=3.2)
    assert m.assess(log, co2, [])["cabin"]["verdict"] == "person"


def test_empty_and_partial_logs_do_not_crash():
    m = g.get_model()
    assert m.assess({}, 3.4, [])["readiness"] >= 5
    miss = {"trials": [{"kind": "pvt", "onsetAt": 5000, "rt": None, "outcome": "miss"}]}
    assert m.assess(miss, "4", [{}])["readiness"] >= 5


def test_pattern_gate_needs_consecutive_runs():
    m, rng = g.get_model(), np.random.default_rng(3)
    hist = [g.simulate_session("fatigue", rng)[0] for _ in range(2)]
    log, co2 = g.simulate_session("fatigue", rng, co2=3.2)
    out = m.assess(log, co2, hist)
    assert out["pattern"]["consecutive_non_nominal"] >= 3
    assert any(n["level"] == "act" for n in out["notices"])
