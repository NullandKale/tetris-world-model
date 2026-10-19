"""Export a pixel world model's averaged weights for the browser (web/) and check it dreams like PyTorch.

    python scripts/export_onnx.py [checkpoint.pt] [--out web/model] [--frames 96]
    python -m http.server 8000 -d web            # then open http://localhost:8000 (WebGPU: Chrome, Edge)

Writes embed.onnx, prefill.onnx, step.onnx and model.json (models/onnx_export.py), and the page's start:
context.bin, the START model frames at the very beginning of a level-0 game (data/real_tetris.py game_start: the
black frames after Start on the level menu, then the first frames of the playfield, board empty) as palette
indices, uint8 (the shipped screenshot; its motion is not known: the page ships no RAM), and context.json (the
frames' shape, their controller bytes and the palette); the model dreams the game from there. Then dreams --frames
from that start with no buttons through OnnxDreamer (models/onnx_dreamer.py) and the model's own Dreamer (one
motion step, as the graphs), both on the CPU in float32, and reports how many pixels differ and each one's frames
per second (model/reference.bin keeps PyTorch's colours, as bytes, for web/test_dreamer.mjs and the page's ?check).
CPU only, so it can run while training has the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from token_world.data.real_tetris import game_start, start_context
from token_world.diagnostics.worlds import load
from token_world.models.dynamics import Dreamer
from token_world.models.onnx_dreamer import OnnxDreamer
from token_world.models.onnx_export import export


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=Path("D:/token_world_checkpoints/world_model_tetris_s9_cool140/model_step152000.pt"))
    p.add_argument("--out", type=Path, default=ROOT / "web/model")
    p.add_argument("--frames", type=int, default=96, help="frames dreamed by both for the check")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for each runtime")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    world = load(args.checkpoint, "cpu")
    model, palette = world.model, torch.as_tensor(world.palette)
    model.motion_steps = 1                                # as the graphs decode
    started = time.perf_counter()
    frames, actions, _ = start_context(game_start())
    frames = frames.astype(np.uint8)
    export(model, args.out, palette)
    (args.out / "context.bin").write_bytes(np.ascontiguousarray(frames).tobytes())
    (args.out / "context.json").write_text(json.dumps({
        "shape": list(frames.shape), "actions": actions.tolist(), "palette": palette.tolist(), "step": world.step}))
    print(f"step {world.step} exported to {args.out} in {time.perf_counter() - started:.0f} s: "
          + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in sorted(args.out.iterdir())))

    started = time.perf_counter()
    onnx = OnnxDreamer(args.out, frames, actions, threads=args.threads)
    dreamed = [onnx.step(0) for _ in range(args.frames)]
    onnx_fps = args.frames / (time.perf_counter() - started)
    started = time.perf_counter()
    colours = palette.numpy().astype(np.float32)
    with torch.no_grad():
        dreamer = Dreamer(model, torch.from_numpy(frames), torch.from_numpy(actions))
        reference = [colours[dreamer.step(0).argmax(-1).numpy()] for _ in range(args.frames)]
    torch_fps = args.frames / (time.perf_counter() - started)
    (args.out / "reference.bin").write_bytes(np.stack(reference).astype(np.uint8).tobytes())   # web/test_dreamer.mjs
    differ = [int((a != b).any(-1).sum()) for a, b in zip(dreamed, reference)]
    print(f"{args.frames} frames: {sum(differ)} pixels differ from PyTorch's (at most {max(differ)} in a frame, "
          f"first at +{next((i + 1 for i, d in enumerate(differ) if d), 0)}); onnxruntime {onnx_fps:.1f} frames/s, "
          f"PyTorch {torch_fps:.1f} frames/s (CPU, {args.threads} threads)")


if __name__ == "__main__":
    main()
