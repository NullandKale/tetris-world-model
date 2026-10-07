"""Bias tests (diagnostics/bias.py): the preview's shapes come from real frames, a dream that always picks one
shape is caught, garbled previews are counted, and the scores land in bias.csv."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.diagnostics.bias import EMPTY, GARBLED, PREVIEW, PieceShapes, choices, scores, write

SHAPES = [np.array(s) for s in ([[1, 1, 1, 1], [0, 0, 0, 0]], [[1, 1, 0, 0], [1, 1, 0, 0]],
                                 [[0, 1, 1, 0], [1, 1, 0, 0]], [[1, 1, 1, 0], [0, 1, 0, 0]])]


def frame(shape: int | None) -> np.ndarray:
    """A model frame with shape `shape` (8 x 8 cells) in the preview box, or an empty box."""
    f = np.zeros((256, 256), np.uint8)
    if shape is not None:
        cells = np.kron(SHAPES[shape], np.ones((8, 8), np.uint8)) * 30
        f[PREVIEW[0].start + 4:PREVIEW[0].start + 20, PREVIEW[1].start + 4:PREVIEW[1].start + 36] = cells
    return f


def sequence(shapes: list[int], each: int = 4) -> np.ndarray:
    return np.stack([frame(s) for s in shapes for _ in range(each)])


class BiasTests(unittest.TestCase):
    def setUp(self):
        self.real = [sequence([0, 1, 2, 3]), sequence([2, 3, 0, 1]), sequence([1, 0, 3, 2])]
        self.shapes = PieceShapes.learn(np.concatenate(self.real))

    def test_the_shapes_are_learned_from_real_frames(self):
        self.assertEqual(len(self.shapes.shapes), 4)
        kinds = self.shapes.classify(np.stack([frame(2), frame(None)]))
        self.assertEqual(kinds[1], EMPTY)
        self.assertEqual(self.shapes.classify(frame(2)[None])[0], kinds[0])
        broken = frame(1)
        broken[PREVIEW[0].start + 6, PREVIEW[1].start + 30] = 30            # one stray pixel
        self.assertEqual(self.shapes.classify(broken[None])[0], GARBLED)

    def test_choices_are_the_changes_of_shape(self):
        kinds = self.shapes.classify(sequence([0, 0, 1, 1, 0]))
        self.assertEqual(len(choices(kinds)), 2)
        self.assertEqual(choices(self.shapes.classify(sequence([3, 3, 3]))), [])

    def test_garbled_frames_are_no_choice(self):
        a, b = self.shapes.classify(sequence([0, 1], each=1))
        kinds = np.array([a, GARBLED, GARBLED, a, b])                            # garbled, back, then a new shape
        self.assertEqual(choices(kinds), [b])

    def test_a_dream_that_always_picks_one_shape_is_caught(self):
        same = [sequence([0, 1, 1, 1]), sequence([2, 1, 1, 1]), sequence([1, 1, 1, 1])]
        s = scores(self.shapes, self.real, same)
        self.assertEqual(s["types_dream"], 1)
        self.assertEqual(s["top_share_dream"], 1.0)
        self.assertGreater(s["type_distance"], 0.3)
        self.assertEqual(scores(self.shapes, self.real, self.real)["type_distance"], 0.0)

    def test_garbled_previews_are_counted(self):
        bad = sequence([0, 1])
        bad[4:, PREVIEW[0].start + 6, PREVIEW[1].start + 30] = 30
        s = scores(self.shapes, self.real[:1], [bad])
        self.assertAlmostEqual(s["garbled"], 0.5)

    def test_a_test_is_written_to_bias_csv(self):
        with tempfile.TemporaryDirectory() as d:
            row = write(Path(d), 2000, "1 pass", self.real, self.real, [])
            write(Path(d), 4000, "1 pass", self.real, self.real, [])
            with (Path(d) / "bias.csv").open() as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["step"] for r in rows], ["2000", "4000"])
        self.assertEqual(row["types_dream"], row["types_real"])


if __name__ == "__main__":
    unittest.main()
