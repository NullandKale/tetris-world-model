"""When a run's long dreams stop getting better: the measurable point to start its cooldown (2026-10-10).

Every long-dream test dreams the same trials (train_dynamics_ui.py LongDreams: the same games, in the same
order, at every test), so two sets of tests compare trial by trial. At each test, for each variant (no buttons,
the bot's buttons) and each score in SCORES, the last TESTS tests are compared with the TESTS before them: each
trial's score averaged over each set, the change per trial, and its mean and standard error. A score is still
improving when its mean gain beats both NOISE standard errors and the score's smallest change that matters
(SCORES: near-zero scores have near-zero noise, so noise alone would count a gain of 0.00001 as real). A test
is flat when every score of every variant is flat, and the run has plateaued when the last CONFIRM tests were
flat (single tests flicker: a score's noise crosses 2 standard errors now and then).

Replayed over the pixel run world_model_tetris_rope_actnorm (scores without the 1,000-frame ones, which it
predates): still improving to 30k (the bot's-buttons dreams' wrong pixels at +16, piece hits), flat at 32k,
36k, 40k between noisy tests, and plateaued from 46k (flat at 44k and 46k) on, apart from one noisy test at
62k. With no buttons its wrong pixels at +128 frames were flat from about 24k, and its cooldown at 60k did not
improve them; one score alone would have stopped it early: hence every score of both variants.
"""
from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TESTS = 3                                  # tests a set: 6k steps at a test every 2,000
NOISE = 2.0                                # standard errors a gain must beat to count as improving
CONFIRM = 2                                # flat tests in a row that make a plateau
# score -> (how to read it: "lower" better, "higher" better, "one" best at 1 (fall: the dream's falling speed
# over the real game's); the smallest gain over a set that matters)
SCORES = {"wrong_16": ("lower", 0.0002), "wrong_128": ("lower", 0.001), "lost_long": ("lower", 0.05),
          "morph_long": ("lower", 0.05), "fall": ("one", 0.01), "piece_hit_32": ("higher", 0.01)}


@dataclass
class Trend:
    variant: str
    score: str
    before: float                          # mean over the earlier set's tests and trials
    after: float                           # the last set's
    gain: float                            # the mean improvement per trial (positive: better)
    error: float                           # its standard error

    @property
    def improving(self) -> bool:
        return self.gain > max(NOISE * self.error, SCORES[self.score][1])


def _badness(values: np.ndarray, kind: str) -> np.ndarray:
    return {"lower": values, "higher": -values, "one": np.abs(values - 1)}[kind]


def trends(rows: list[dict]) -> list[Trend]:
    """long_dream.csv rows (any number of tests) -> each variant's and score's trend over the last 2 * TESTS
    tests; none for a variant with fewer tests, or a score its tests lack."""
    out = []
    for variant in dict.fromkeys(r["variant"] for r in rows):
        by_step: dict[int, list[dict]] = {}
        for r in rows:
            if r["variant"] == variant:
                by_step.setdefault(int(r["step"]), []).append(r)
        steps = sorted(by_step)[-2 * TESTS:]
        if len(steps) < 2 * TESTS:
            continue
        trials = min(len(by_step[s]) for s in steps)
        for score, (kind, _) in SCORES.items():
            try:
                table = np.array([[float(r[score]) for r in by_step[s][:trials]] for s in steps])  # [tests, trials]
            except (KeyError, ValueError):
                continue
            if not np.isfinite(table).any():
                continue
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # a trial missing in a whole set
                before, after = np.nanmean(table[:TESTS], 0), np.nanmean(table[TESTS:], 0)        # per trial
            change = _badness(before, kind) - _badness(after, kind)
            change = change[np.isfinite(change)]
            if len(change) < 2:
                continue
            out.append(Trend(variant, score, float(np.nanmean(before)), float(np.nanmean(after)),
                             float(change.mean()), float(change.std(ddof=1) / math.sqrt(len(change)))))
    return out


def history(rows: list[dict]) -> list[tuple[int, bool | None]]:
    """Each tested step -> whether that test was flat (None: too few tests yet)."""
    steps = sorted({int(r["step"]) for r in rows})
    out = []
    for step in steps:
        found = trends([r for r in rows if int(r["step"]) <= step])
        out.append((step, (not any(t.improving for t in found)) if found else None))
    return out


def read(run: Path) -> list[dict]:
    path = Path(run) / "long_dream.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def report(name: str, rows: list[dict]) -> str:
    """A run's plateau check as text (the run window's Long dreams tab)."""
    found = trends(rows)
    if not found:
        return (f"{name}: plateau check needs {2 * TESTS} long-dream tests of a variant "
                f"(the last {TESTS} against the {TESTS} before, the same trials)")
    checks = history(rows)
    last = checks[-1][0]
    plateaued = len(checks) >= CONFIRM and all(flat for _, flat in checks[-CONFIRM:])
    still = [t for t in found if t.improving]
    head = (f"{name} at {last:,}: " + (f"PLATEAUED: every score flat at the last {CONFIRM} tests; time for the "
                                       "cooldown" if plateaued else
                                       f"still improving: {len(still)} of {len(found)} scores" if still else
                                       f"flat at this test; a plateau once flat at {CONFIRM} in a row"))
    lines = [head,
             f"    (the last {TESTS} tests against the {TESTS} before, paired over the same trials; improving = "
             f"better by more than {NOISE:g} standard errors and the score's smallest change that matters)",
             "    tests flat: " + "  ".join(f"{s // 1000}k {'flat' if f else 'improving'}"
                                            for s, f in checks[-8:] if f is not None)]
    for t in found:
        lines.append(f"    {t.variant:10s} {t.score:13s} {t.before:9.4f} -> {t.after:9.4f}   gain {t.gain:+.4f} "
                     f"± {t.error:.4f}   {'improving' if t.improving else 'flat'}")
    return "\n".join(lines)
