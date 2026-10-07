"""ONNX export (models/onnx_export.py): the exported graphs dream what the layered model's Dreamer dreams, across
window slides, for a model with a fixed camera and for one with camera tokens whose camera holds."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from token_world.data.nes_layers import LAYER_COLOURS
from token_world.diagnostics.worlds import composed
from token_world.models.layered import MOVE, MOVE_CLASSES
from token_world.models.layered_pixels import PixelLayers

try:
    import onnxruntime  # noqa: F401
    import onnxscript  # noqa: F401
    from token_world.models.onnx_dreamer import OnnxDreamer
    from token_world.models.onnx_export import INPUTS, export
except ImportError:
    OnnxDreamer = None

from test_layered import window

PALETTE = torch.randint(0, 256, (LAYER_COLOURS, 3), generator=torch.Generator().manual_seed(0)).to(torch.uint8)


@unittest.skipIf(OnnxDreamer is None, "needs onnx, onnxruntime and onnxscript")
class OnnxTests(unittest.TestCase):
    def dream_both(self, model):
        """3 real frames, then 7 dreamed (the 6-frame window fills and slides): each frame's colours from the
        graphs and from PyTorch's Dreamer (temperature 0: the camera its most likely move, which holds)."""
        layers, acts = window(1, 10, 2), torch.randint(-1, 256, (1, 10))
        start = {k: v[0, :3] for k, v in layers.items()}
        with tempfile.TemporaryDirectory() as folder:
            export(model, Path(folder), PALETTE, start)
            onnx = OnnxDreamer(Path(folder), {k: start[k].numpy() for k in INPUTS}, acts[0, :3].numpy(), keep=3)
            with torch.no_grad():
                dreamer = model.Dreamer(model, {k: v[:, :3] for k, v in layers.items()}, acts[:, :3], keep=3,
                                        temperature=0.0)
                for a in acts[0, 3:].tolist():
                    frame = {k: v[0].numpy() for k, v in dreamer.step(torch.tensor([a])).items() if k != "probs"}
                    self.assertTrue((frame["camera"] == start["camera"][-1].numpy()).all())
                    expected = PALETTE.numpy()[composed(frame)].astype(np.float32)
                    same = (onnx.step(a) == expected).all(-1).mean()
                    self.assertGreater(same, 0.999)

    def test_a_fixed_camera_model_dreams_as_pytorch(self):
        torch.manual_seed(0)
        self.dream_both(PixelLayers(dim=32, layers=2, heads=2, frames=6, colour_dim=8, fixed_camera=True).eval())

    def test_a_model_with_camera_tokens_dreams_as_pytorch_while_its_camera_holds(self):
        torch.manual_seed(1)
        model = PixelLayers(dim=32, layers=2, heads=2, frames=6, colour_dim=8, registers=2, space_rope=True).eval()
        with torch.no_grad():                                    # its camera head: "no move", surely
            model.camera_head[-1].bias[MOVE] += 100
            model.camera_head[-1].bias[MOVE_CLASSES + MOVE] += 100
        self.dream_both(model)

    def test_a_moving_start_is_refused(self):
        model = PixelLayers(dim=32, layers=2, heads=2, frames=6, colour_dim=8).eval()
        start = {k: v[0, :3] for k, v in window(1, 3, 2, scroll=4).items()}
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            export(model, Path(folder), PALETTE, start)


if __name__ == "__main__":
    unittest.main()
