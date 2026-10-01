"""Export a world-model run's averaged weights for the browser (web/) and check it dreams like PyTorch.

    python scripts/export_onnx.py [checkpoint.pt] [--out web/model] [--frames 96] [--level 0]
    python -m http.server 8000 -d web            # then open http://localhost:8000 (WebGPU: Chrome, Edge)

Writes prefill.onnx, step.onnx and model.json (models/onnx_export.py), and the page's start: context.bin
(the START frames at the very beginning of a level-0 game: the black frames after Start on the level
menu and the first frame of the playfield, board empty; uint8 palette indices) and context.json (their controller
bytes, the palette, the default level and keep); the model dreams the game from there. Then dreams --frames from that start with no buttons through
OnnxDreamer (models/onnx_dreamer.py) and PyTorch's Dreamer, both on the CPU in float32, and reports where
the frames first differ and each one's frames per second (model/reference.bin keeps PyTorch's frames for
web/test_dreamer.mjs and the page's ?check). CPU only, so it can run while training has the GPU.
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

from token_world.data.real_tetris import IN_GAME, MODE, RealTetris
from token_world.models.dynamics import LEVELS, Decoding, Dreamer, build
from token_world.models.onnx_dreamer import OnnxDreamer
from token_world.models.onnx_export import export

KEEP = 48
START = 3                        # frames the page starts from (prefill takes all but the last, 2 or more)
LEVEL = 0x44                     # the game's level byte


def game_start(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The START frames ending on the first playfield frame of a level-0 game, after its black frames
    (the bot plays the menus; seeds are tried until it picks level 0), and their controller bytes.
    Ending on the playfield: from the black frames alone (even with the 48 real frames before) the model
    dreams the menu back first (game pixels wrong at +9: 2.9% against 1.4%)."""
    for seed in range(seed, seed + 200):
        game = RealTetris(seed)
        for _ in range(20_000):
            frame, _ = game.step(None)
            if game.observation.ram[MODE] == IN_GAME and (frame[16:240] == frame[16, 0]).all():
                break                                    # the first black frame of the game
        else:
            continue
        while True:                                      # to the last one
            frame, _ = game.step(None)
            if not (frame[16:240] == frame[16, 0]).all():
                break
        if game.observation.ram[LEVEL] != 0:
            continue
        frames, actions = list(game.frames)[-START:], list(game.actions)[-START:]
        return np.stack(frames), np.array(actions, np.int64)
    raise RuntimeError("no level-0 game start found")



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=ROOT / "output/world_model_tetris_small/model_latest.pt")
    p.add_argument("--out", type=Path, default=ROOT / "web/model")
    p.add_argument("--frames", type=int, default=96, help="frames dreamed by both for the check")
    p.add_argument("--level", type=int, default=0,
                   help="the level generated frames carry: the page's default and the check's")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for each runtime")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(saved["args"]).eval()
    model.load_state_dict(saved["ema"])
    started = time.perf_counter()
    export(model, args.out)
    context, actions = game_start()
    (args.out / "context.bin").write_bytes(context.astype(np.uint8).tobytes())
    (args.out / "context.json").write_text(json.dumps({
        "actions": actions.tolist(), "palette": saved["palettes"]["tetris"].tolist(),
        "level": args.level, "levels": LEVELS, "keep": KEEP, "step": saved["step"]}))
    print(f"step {saved['step']} exported to {args.out} in {time.perf_counter() - started:.0f} s: "
          + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in sorted(args.out.iterdir())))

    started = time.perf_counter()
    onnx = OnnxDreamer(args.out, context, actions, args.level, KEEP, threads=args.threads)
    dreamed = [onnx.step(0) for _ in range(args.frames)]
    onnx_fps = args.frames / (time.perf_counter() - started)
    started = time.perf_counter()
    with torch.no_grad():
        dreamer = Dreamer(model, torch.from_numpy(context), torch.from_numpy(actions),
                          Decoding(steps=1, level=args.level), KEEP)
        reference = [dreamer.step(0).numpy() for _ in range(args.frames)]
    torch_fps = args.frames / (time.perf_counter() - started)
    (args.out / "reference.bin").write_bytes(np.stack(reference).tobytes())   # web/test_dreamer.mjs checks it
    differ = [i for i, (a, b) in enumerate(zip(dreamed, reference)) if not np.array_equal(a, b)]
    pixels = np.mean([(a != b).mean() for a, b in zip(dreamed, reference)])
    print(f"{args.frames} frames, level {args.level}: "
          + ("identical" if not differ else f"first differs at +{differ[0] + 1}, {len(differ)} frames differ, "
             f"{pixels:.4%} of pixels")
          + f"; onnxruntime {onnx_fps:.1f} frames/s, PyTorch {torch_fps:.1f} frames/s (CPU, {args.threads} threads)")


if __name__ == "__main__":
    main()
