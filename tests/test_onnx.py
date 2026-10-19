"""ONNX export (models/onnx_export.py): the exported graphs dream what the pixel model's Dreamer dreams with one
motion step, across window slides."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.models.dynamics import COLOURS, Dreamer, Dynamics, motion_tokens

try:
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401
    from token_world.models.onnx_dreamer import OnnxDreamer
    from token_world.models.onnx_export import export
except ImportError:
    OnnxDreamer = None

from test_dynamics import window

PALETTE = torch.randint(0, 256, (COLOURS, 3), generator=torch.Generator().manual_seed(0)).to(torch.uint8)


def motion_model(**options) -> Dynamics:
    model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, motion=True, motion_first=True,
                     **options).eval()
    with torch.no_grad():                                     # motion that matters (it starts at zero)
        model.motion_in.weight.normal_()
    model.motion_steps = 1
    return model


@unittest.skipIf(OnnxDreamer is None, "needs onnx, onnxruntime and onnxscript")
class OnnxTests(unittest.TestCase):
    def dream_both(self, model):
        """3 real frames, then 7 dreamed (the 6-frame window fills and slides): each frame's colours from the
        graphs and from PyTorch's Dreamer (each pixel's most likely colour)."""
        x, acts = window(1, 10, 3), torch.randint(-1, 256, (1, 10), generator=torch.Generator().manual_seed(5))
        with tempfile.TemporaryDirectory() as folder:
            export(model, Path(folder), PALETTE)
            onnx = OnnxDreamer(Path(folder), x[0, :3].numpy(), acts[0, :3].numpy(), keep=3)
            with torch.no_grad():
                dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
                for a in acts[0, 3:].tolist():
                    expected = PALETTE.numpy()[dreamer.step(a).argmax(-1).numpy()].astype(np.float32)
                    self.assertGreater((onnx.step(a) == expected).all(-1).mean(), 0.999)
                    drawn = motion_tokens(dreamer.motion[None], 16)[0].numpy()             # [N, cells]
                    self.assertGreater((onnx.motion[-1] == drawn).mean(), 0.999)

    def test_a_motion_first_model_dreams_as_pytorch(self):
        torch.manual_seed(0)
        self.dream_both(motion_model(space_rope=True, qk_norm=True, action_norm=True, local_attention=True))

    def test_the_plain_options_dream_as_pytorch_too(self):
        torch.manual_seed(1)
        self.dream_both(motion_model())

    def test_a_model_without_motion_first_is_refused(self):
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8).eval()
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(NotImplementedError):
            export(model, Path(folder), PALETTE)


if __name__ == "__main__":
    unittest.main()
