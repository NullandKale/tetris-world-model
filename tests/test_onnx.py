"""ONNX export: the exported graphs dream what PyTorch's Dreamer dreams, across window slides."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.models.dynamics import COLOURS, Dreamer, Dynamics

try:
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401
    from token_world.models.onnx_dreamer import OnnxDreamer
    from token_world.models.onnx_export import export
except ImportError:
    OnnxDreamer = None

from test_dynamics import tiny, window

PALETTE = torch.randint(0, 256, (COLOURS, 3), generator=torch.Generator().manual_seed(0)).to(torch.uint8)


@unittest.skipIf(OnnxDreamer is None, "needs onnx, onnxruntime and onnxscript")
class OnnxTests(unittest.TestCase):
    def dream_both(self, model, seed: int):
        """Prefill, steps, a slide (re-encoding the last 3 frames) and steps after it: each frame's expected
        colours from the graphs and from PyTorch."""
        x, acts = window(1, 6, seed), torch.randint(-1, 256, (1, 9))      # 3 real frames, 6 dreamed
        with tempfile.TemporaryDirectory() as folder:
            export(model, Path(folder), PALETTE)
            onnx = OnnxDreamer(Path(folder), x[0, :3].numpy(), acts[0, :3].numpy(), keep=3)
            with torch.no_grad():
                dreamer = Dreamer(model, x[0, :3], acts[0, :3], keep=3)
                for a in acts[0, 3:].tolist():                  # 6 frames: the 6-frame window fills and slides once
                    expected = (dreamer.step(a) @ PALETTE.float()).numpy()
                    self.assertTrue(abs(onnx.step(a) - expected).max() < 1e-2)

    def test_onnx_dream_matches_pytorch_through_a_window_slide(self):
        self.dream_both(tiny().eval(), 21)

    def test_modern_blocks_export_too(self):
        """RMSNorm, QK-norm and SwiGLU export as plain ops the browser runs."""
        torch.manual_seed(3)
        model = Dynamics(dim=32, layers=2, heads=2, patch=16, frames=6, colour_dim=8, modern=True).eval()
        with torch.no_grad():
            model.colour_bias.normal_(0, 0.5)
        self.dream_both(model, 23)


if __name__ == "__main__":
    unittest.main()
