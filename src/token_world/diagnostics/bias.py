"""Bias tests: does a dream choose among the game's random outcomes as the game does?

A frame-by-frame score cannot tell a model that draws every new piece as the same shape (coherent, but
always one choice) from one that chooses like the game. Here, in Tetris's next-piece preview (the one random
event of a no-button dream; PREVIEW in model frames, data/model_frames.py):

- the preview's shapes are learned from real frames (PieceShapes.learn: the distinct shapes real play shows
  there, so no shape is written in);
- a choice is each time a preview shows a new real shape (the game picked the next piece); frames that show
  no real shape (garbled) are skipped, so a garbled stretch between two shows of the same shape is no choice
  (until 2026-10-07 changes into and out of garbled were counted in choices_dream too);
- a test compares the dreams with the real games over the same trials (scores): how many choices and
  types each made, the most common chosen type's share, the distance between the shares of frames each
  shape is shown in (total variation: 0 the same mix, 1 nothing in common; frames, not choices: 32 real
  128-frame futures make only a handful of choices), and the share of the dreams' previews that show no
  real shape (garbled).

Both trainers write a test's scores to bias.csv (write) at every long-dream test; the run window shows them
(ui/run_viewer.py).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PREVIEW = (slice(124, 148), slice(188, 228))     # the next-piece box in model frames (every shape inside)
MIN_SEEN = 3                                      # a shape is real if real play shows it in this many frames
GARBLED, EMPTY = -1, -2
FIELDS = ("step", "variant", "choices_real", "choices_dream", "types_real", "types_dream", "top_share_real",
          "top_share_dream", "type_distance", "garbled")


class PieceShapes:
    """The preview's real shapes (binary masks of drawn pixels) and the box's empty colour."""

    def __init__(self, shapes: np.ndarray, empty: int):
        self.shapes, self.empty = shapes, empty

    @classmethod
    def learn(cls, frames: np.ndarray) -> "PieceShapes":
        """Real model frames [N, 256, 256] -> the shapes real play shows in the preview."""
        box = frames[:, PREVIEW[0], PREVIEW[1]]
        empty = int(np.bincount(box.ravel()).argmax())                     # the box's background colour
        masks = (box != empty).reshape(len(box), -1)
        masks = masks[masks.any(1)]
        unique, counts = np.unique(masks, axis=0, return_counts=True)
        return cls(unique[counts >= MIN_SEEN], empty)

    def classify(self, frames: np.ndarray) -> np.ndarray:
        """Model frames [N, 256, 256] -> each preview's shape index, EMPTY, or GARBLED (no real shape)."""
        masks = (frames[:, PREVIEW[0], PREVIEW[1]] != self.empty).reshape(len(frames), -1)
        out = np.full(len(frames), GARBLED)
        out[~masks.any(1)] = EMPTY
        for i, shape in enumerate(self.shapes):
            out[(masks == shape).all(1)] = i
        return out


def choices(kinds: np.ndarray) -> list[int]:
    """One sequence's preview shapes [N] -> the real shapes it changed to (each a choice of the next piece);
    empty and garbled frames are skipped; the first shape shown was chosen before the dream."""
    shown = [k for k in kinds if k >= 0]
    return [b for a, b in zip(shown, shown[1:]) if b != a]


def scores(shapes: PieceShapes, real: list[np.ndarray], dreamed: list[np.ndarray],
           context: list[np.ndarray] = ()) -> dict[str, float]:
    """Paired sequences of model frames (each [N, 256, 256]; the real game's and the dream's from the same
    start), and real context frames that only add to the real mix -> the test's bias scores."""
    real_kinds = [shapes.classify(f) for f in real]
    dream_kinds = [shapes.classify(f) for f in dreamed]
    picked_real = [c for k in real_kinds for c in choices(k)]
    picked_dream = [c for k in dream_kinds for c in choices(k)]
    n = len(shapes.shapes)
    mix = lambda kinds: np.bincount(kinds, minlength=n) / max(len(kinds), 1)
    real_shown = np.concatenate(real_kinds + [shapes.classify(f) for f in context])
    top = lambda picks: float(np.bincount(picks).max() / len(picks)) if picks else float("nan")
    shown = np.concatenate(dream_kinds)
    shown = shown[shown != EMPTY]
    return {"choices_real": len(picked_real), "choices_dream": len(picked_dream),
            "types_real": len(set(picked_real)), "types_dream": len(set(picked_dream)),
            "top_share_real": top(picked_real), "top_share_dream": top(picked_dream),
            "type_distance": float(0.5 * np.abs(mix(real_shown[real_shown >= 0]) - mix(shown[shown >= 0])).sum())
            if (real_shown >= 0).any() and (shown >= 0).any() else float("nan"),
            "garbled": float((shown == GARBLED).mean()) if len(shown) else float("nan")}


def write(out: Path, step: int, variant: str, real: list[np.ndarray], dreamed: list[np.ndarray],
          context: list[np.ndarray]) -> dict[str, float]:
    """A long-dream test's bias scores -> appended to out/bias.csv; the shapes are learned from the test's
    real frames (contexts and futures)."""
    shapes = PieceShapes.learn(np.concatenate(context + real))
    row = {"step": step, "variant": variant, **scores(shapes, real, dreamed, context)}
    path = out / "bias.csv"
    new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return row
