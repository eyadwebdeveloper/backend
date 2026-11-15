"""AI notice gate (AI plan, section 5).

An AI notice is a quiet advisory message. It is NOT an official alert and
can never create one. Rules implemented here, from the AI plan:

  1. a notice appears only if the strangeness score stays high for several
     checks in a row (default 3), not after one odd reading
  2. a cooldown: the same notice is not repeated more than about once per
     cooldown period (default 1 day)
  3. a day with unusable data (data-quality gate failed) gets score NaN and
     can never produce a health notice from that bad data
  4. every notice carries why it appeared and the "not a diagnosis" label

The astronaut's "Noticed / Not an issue" tap is app-side behavior (saved
for medic review, never silences the system) and is out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Notice:
    day: int
    source: str            # e.g. "tier1a_isolation_forest"
    strangeness: float
    explanation: str

    def as_dict(self) -> dict:
        return {"day": self.day, "source": self.source,
                "strangeness": round(float(self.strangeness), 4),
                "explanation": self.explanation}


def fire_notices(scores, explanations, high_thresh: float,
                 consecutive: int = 3, cooldown: int = 1,
                 source: str = "ai") -> list:
    """Evaluate a per-day score series and fire notices.

    scores       array-like of strangeness in [0, 1]; NaN = unusable day
    explanations array-like of strings (why the score is high that day)
    Returns a list of Notice, one per firing (cooldown-respecting).
    """
    notices = []
    last_fire = -10 ** 9
    run = 0
    for d, (s, e) in enumerate(zip(scores, explanations)):
        if s is None or not np.isfinite(s):
            run = 0                      # bad-data day: no notice, streak resets
            continue
        if float(s) >= high_thresh:
            run += 1
        else:
            run = 0
        if run >= consecutive and (d - last_fire) >= cooldown:
            notices.append(Notice(int(d), source, float(s), e))
            last_fire = d
    return notices


def notice_episodes(notices: list) -> list:
    """Group fired notices into episodes: consecutive/adjacent fires (gap <= 1 day)."""
    episodes = []
    for n in notices:
        if episodes and n.day - episodes[-1][-1].day <= 1:
            episodes[-1].append(n)
        else:
            episodes.append([n])
    return episodes
