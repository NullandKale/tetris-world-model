"""ONNX export: the exported graphs dream exactly what PyTorch's Dreamer dreams, across window slides."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.models.dynamics import Decoding, Dreamer

try:
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401
    from token_world.models.onnx_dreamer import OnnxDreamer
    from token_world.models.onnx_export import export
except ImportError:
    OnnxDreamer = None

from test_dynamics import tiny, window


@unittest.skipIf(OnnxDreamer is None, "needs onnx, onnxruntime and onnxscript")
class OnnxTests(unittest.TestCase):
    def test_onnx_dream_matches_pytorch_through_a_window_slide(self):
        """Prefill, steps, a slide (re-encoding the last 3 frames) and steps after it, frame for frame.
        Not further: this random model's logits have near-ties that float32 rounding in another runtime
        flips (7 of 65,536 pixels at the next slide), and a flipped pixel changes every later frame. The
        trained model dreams 96 identical frames through 6 slides (scripts/export_onnx.py)."""
        model = tiny().eval()
        x, acts = window(1, 6, 21), torch.randint(-1, 256, (1, 9))      # 3 real frames, 6 dreamed
        decoding = Decoding(steps=2, level=1)                 # two steps: the confidence ordering is exported too
        with tempfile.TemporaryDirectory() as folder:
            export(model, Path(folder), decoding)
            onnx = OnnxDreamer(Path(folder), x[0, :3].numpy(), acts[0, :3].numpy(), level=1, keep=3)
            with torch.no_grad():
                dreamer = Dreamer(model, x[0, :3], acts[0, :3], decoding, keep=3)
                for a in acts[0, 3:].tolist():                  # 6 frames: the 6-frame window fills and slides once
                    self.assertTrue((onnx.step(a) == dreamer.step(a).numpy()).all())


if __name__ == "__main__":
    unittest.main()
