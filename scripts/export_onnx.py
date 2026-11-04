"""Export a pixel world model's averaged weights for the browser (web/) and check it dreams like PyTorch.

    python scripts/export_onnx.py [checkpoint.pt] [--out web/model] [--frames 96]
    python -m http.server 8000 -d web            # then open http://localhost:8000 (WebGPU: Chrome, Edge)

Writes embed.onnx, prefill.onnx, step.onnx and model.json (models/onnx_export.py), and the page's start:
context.bin, MENU_FRAMES (63: the model's window but the drawn frame) model frames on the level-select menu,
cursor on level 0, nobody pressing anything
(data/real_tetris.py menu_start; since 2026-11-03, before that a level-0 game's first frames) as palette
indices, uint8 (the shipped screenshot; a two-layer model's as its two layers, [START, 2, 256, 256]: given the
picture alone it reads the falling piece as stack; its motion is not known: the page ships no RAM), and
context.json (the frames' shape, their controller bytes and the palette); the model dreams the game from there.
Then dreams --frames
from that start with no buttons through OnnxDreamer (models/onnx_dreamer.py) and the model's own Dreamer (one
motion step, as the graphs), both on the CPU in float32, and reports how many pixels differ and each one's frames
per second (model/reference.bin keeps PyTorch's colours, as bytes, for web/test_dreamer.mjs and the page's ?check).
A frame-skip model is also dreamed --frames steps of 1..skip frames (seeded), each with random bytes, by both, its
trace checked against training's order (diagnostics/input_trace.py); skip_steps.json (each step's bytes, oldest
first) and skip_reference.bin (PyTorch's colours) let web/test_dreamer.mjs do the same in JavaScript.
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

from token_world.data.real_tetris import MENU_FRAMES, menu_start, start_context
from token_world.diagnostics.input_trace import check
from token_world.diagnostics.worlds import blanked, load
from token_world.models.dynamics import Dreamer
from token_world.models.onnx_dreamer import OnnxDreamer
from token_world.models.onnx_export import export


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=ROOT / "output/world_model_tetris_s9_skip/model_latest.pt")
    p.add_argument("--out", type=Path, default=ROOT / "web/model")
    p.add_argument("--frames", type=int, default=96, help="frames dreamed by both for the check")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for each runtime")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    world = load(args.checkpoint, "cpu")
    model, palette = world.model, torch.as_tensor(world.palette)
    model.motion_steps = 1                                # as the graphs decode
    started = time.perf_counter()
    pictures, actions, _, layers = start_context(menu_start(), MENU_FRAMES)
    frames = (layers if model.two_layers else pictures).astype(np.uint8)
    if world.blank:                                       # a BLANK_BORDER model's start: its bands one shade
        frames = blanked(frames, world.palette)
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
    if model.skip > 1:
        skip_check(model, args, frames, actions, colours)


def skip_check(model, args, frames: np.ndarray, actions: np.ndarray, colours: np.ndarray) -> None:
    """A frame-skip model's steps of 1..skip frames through both, and the graph's inputs against training's order."""
    rng = np.random.default_rng(0)
    steps = [rng.integers(-1, 256, rng.integers(1, model.skip + 1)).tolist() for _ in range(args.frames)]
    onnx = OnnxDreamer(args.out, frames, actions, threads=args.threads)
    with torch.no_grad():
        dreamer = Dreamer(model, torch.from_numpy(frames), torch.from_numpy(actions))
        reference = [colours[dreamer.step(s[-1], s[:-1][::-1]).argmax(-1).numpy()] for s in steps]
    dreamed = [onnx.step(s[-1], s[:-1][::-1]) for s in steps]
    result = check({**onnx.trace, "frames": [b for s in steps for b in s]})
    (args.out / "skip_steps.json").write_text(json.dumps({"steps": steps}))
    (args.out / "skip_reference.bin").write_bytes(np.stack(reference).astype(np.uint8).tobytes())
    differ = [int((a != b).any(-1).sum()) for a, b in zip(dreamed, reference)]
    print(f"{len(steps)} steps of 1..{model.skip} frames ({result['frames']} frames, in training's order): "
          f"{sum(differ)} pixels differ from PyTorch's (at most {max(differ)} in a frame)")
    if max(differ) > 64:                                  # a wrong byte or clock changes thousands
        raise SystemExit("the graphs' skip steps do not dream as PyTorch's")


if __name__ == "__main__":
    main()
