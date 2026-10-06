"""Generic coherence scores: real play scores clean, blends and morphs do not, a different legal outcome does."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from token_world.diagnostics.coherence import _WEIGHTS, PatchBank, patch_hashes, score


def falling(column: int, frames: int = 40, rows_per_frame: float = 1.0) -> np.ndarray:
    """A 16 x 16 block falling down a column of an empty 256 x 256 screen, one pixel row a frame."""
    out = np.zeros((frames, 256, 256), np.uint8)
    for t in range(frames):
        top = int(t * rows_per_frame)
        out[t, top:top + 16, column:column + 16] = 9
    return out


class CoherenceTests(unittest.TestCase):
    def setUp(self):
        self.bank = PatchBank()
        for column in (0, 32, 64):                                           # real play: blocks in three columns
            self.bank.add(falling(column))

    def test_hashes_tell_patches_apart(self):
        a, b = np.zeros((1, 256, 256), np.uint8), np.zeros((1, 256, 256), np.uint8)
        b[0, 3, 5] = 1
        self.assertEqual(patch_hashes(a).shape, (1, 256))
        self.assertNotEqual(patch_hashes(a)[0, 0], patch_hashes(b)[0, 0])
        self.assertEqual(patch_hashes(a)[0, 1], patch_hashes(b)[0, 1])

    def test_a_hash_is_the_weighted_pixel_sum_modulo_2_64(self):
        frame = np.random.default_rng(3).integers(0, 64, (1, 256, 256), dtype=np.uint8)
        pixels = frame[0, 16:32, 32:48].ravel()                                # token 1 * 16 + 2
        weights = [int(w) % 2 ** 64 for w in _WEIGHTS]
        want = sum(int(p) * w for p, w in zip(pixels, weights)) % 2 ** 64
        self.assertEqual(int(patch_hashes(frame)[0, 18]) % 2 ** 64, want)

    def test_a_bank_from_tensors_is_the_bank_from_arrays(self):
        other = PatchBank()
        for column in (0, 32, 64):
            other.add(torch.from_numpy(falling(column)))
        self.assertTrue(np.array_equal(other.patches, self.bank.patches))
        self.assertTrue(np.array_equal(other.changes, self.bank.changes))

    def test_a_saved_bank_scores_as_before(self):
        import tempfile
        real = falling(32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.npz"
            self.bank.save(path)
            loaded = PatchBank.load(path)
        self.assertEqual(score(loaded, real[0], real[1:]), score(self.bank, real[0], real[1:]))

    def test_real_play_scores_clean(self):
        real = falling(32)
        out = score(self.bank, real[0], real[1:])
        self.assertEqual((out["unseen_patch"], out["unseen_change"], out["unsure_px"]), (0.0, 0.0, 0.0))
        self.assertGreater(out["change_px"], 0)

    def test_another_legal_outcome_is_coherent(self):
        other = falling(64)                                                  # the block in another column
        self.assertEqual(score(self.bank, other[0], other[1:])["unseen_patch"], 0.0)

    def test_a_blend_is_not(self):
        real = falling(32)
        blend = real.copy()
        blend[1:, :, 32:40] = 0                                              # half the block, as a hedge draws it
        out = score(self.bank, real[0], blend[1:])
        self.assertGreater(out["unseen_patch"], 0.5)
        self.assertGreater(out["unseen_change"], 0.5)

    def test_a_frozen_dream_does_not_change(self):
        real = falling(32)
        frozen = np.repeat(real[:1], 39, 0)
        self.assertEqual(score(self.bank, real[0], frozen)["change_px"], 0.0)

    def test_unsure_pixels_are_counted(self):
        real = falling(32)
        sure = np.full(real[1:].shape, 0.5, np.float32)
        self.assertEqual(score(self.bank, real[0], real[1:], sure)["unsure_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
