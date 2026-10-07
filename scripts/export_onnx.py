"""Export a layered world model's averaged weights for the browser (web/) and check it dreams like PyTorch.

    python scripts/export_onnx.py [checkpoint.pt] [--out web/model] [--frames 96]
    python -m http.server 8000 -d web            # then open http://localhost:8000 (WebGPU: Chrome, Edge)

Writes embed.onnx, prefill.onnx, step.onnx and model.json (models/onnx_export.py), and the page's start:
context.bin, the START frames at the very beginning of a level-0 game (the black frames after Start on the level
menu, then the first frames of the playfield, board empty) as their layers, uint8 (the shipped "screenshot" is a
frame's layers: background cells, sprite picture, border; the user, 2026-10-06), and context.json (where each
layer sits in context.bin, the frames' controller bytes and the palette); the model dreams the game from there,
holding the start's camera. Then dreams --frames from that start with no buttons through OnnxDreamer
(models/onnx_dreamer.py) and the model's own Dreamer (temperature 0), both on the CPU in float32, and reports how
many pixels differ and each one's frames per second (model/reference.bin keeps PyTorch's colours, as bytes, for
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
from token_world.diagnostics.worlds import composed
from token_world.models.layered import build
from token_world.models.onnx_dreamer import OnnxDreamer
from token_world.models.onnx_export import INPUTS, export

START = 3                        # frames the page starts from (prefill takes all but the last, 2 or more)
LEVEL = 0x44                     # the game's level byte
SETTLE = 8                       # frames into the game the start ends: the camera has moved to the playfield


def game_start(seed: int = 0) -> tuple[dict, np.ndarray]:
    """The START frames ending SETTLE frames into a level-0 game, after its black frames (the bot plays the
    menus; seeds are tried until it picks level 0): their layers (each key [START, ...]) and controller bytes.
    Ending on the playfield: from the black frames alone the model dreams the menu back first; and past the
    playfield's camera switch (the menus' nametable to the playfield's, 2 frames in), as the dreams hold the
    start's camera."""
    for seed in range(seed, seed + 200):
        game = RealTetris(seed, layered=True)
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
        for _ in range(SETTLE):                          # the playfield's camera (its nametable) comes 2 frames in
            game.step(None)
        if game.observation.ram[LEVEL] != 0:             # (set by now: Start held with A adds 10 levels)
            continue
        frames = list(game.layers)[-START:]
        if any((f["camera"] != frames[-1]["camera"]).any() for f in frames):
            continue
        return {k: np.stack([f[k] for f in frames]) for k in frames[0]}, np.array(list(game.actions)[-START:], np.int64)
    raise RuntimeError("no level-0 game start found")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   default=ROOT / "output/world_model_tetris_layered_px/model_latest.pt")
    p.add_argument("--out", type=Path, default=ROOT / "web/model")
    p.add_argument("--frames", type=int, default=96, help="frames dreamed by both for the check")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for each runtime")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(saved["args"]).eval()
    model.load_state_dict(saved["ema"])
    palette = torch.as_tensor(saved["palettes"]["tetris"])
    started = time.perf_counter()
    layers, actions = game_start()
    export(model, args.out, palette, layers)
    blob, inputs, offset = bytearray(), [], 0
    for name in INPUTS:                                  # the start's layers, one after another
        data = np.ascontiguousarray(layers[name].astype(np.uint8))
        inputs.append({"name": name, "shape": list(data.shape), "offset": offset})
        blob += data.tobytes()
        offset += data.nbytes
    (args.out / "context.bin").write_bytes(bytes(blob))
    (args.out / "context.json").write_text(json.dumps({
        "inputs": inputs, "actions": actions.tolist(), "palette": palette.tolist(), "step": saved["step"]}))
    print(f"step {saved['step']} exported to {args.out} in {time.perf_counter() - started:.0f} s: "
          + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.1f} MB" for f in sorted(args.out.iterdir())))

    started = time.perf_counter()
    onnx = OnnxDreamer(args.out, layers, actions, threads=args.threads)
    dreamed = [onnx.step(0) for _ in range(args.frames)]
    onnx_fps = args.frames / (time.perf_counter() - started)
    started = time.perf_counter()
    colours = palette.numpy().astype(np.float32)
    with torch.no_grad():
        torch_layers = {k: torch.from_numpy(np.asarray(v))[None] for k, v in layers.items()}
        dreamer = model.Dreamer(model, torch_layers, torch.from_numpy(actions)[None], temperature=0.0)
        reference = []
        for _ in range(args.frames):
            frame = {k: v[0].numpy() for k, v in dreamer.step(torch.zeros(1, dtype=torch.long)).items() if k != "probs"}
            reference.append(colours[composed(frame)])
    torch_fps = args.frames / (time.perf_counter() - started)
    (args.out / "reference.bin").write_bytes(np.stack(reference).astype(np.uint8).tobytes())   # web/test_dreamer.mjs
    differ = [int((a != b).any(-1).sum()) for a, b in zip(dreamed, reference)]
    print(f"{args.frames} frames: {sum(differ)} pixels differ from PyTorch's (at most {max(differ)} in a frame, "
          f"first at +{next((i + 1 for i, d in enumerate(differ) if d), 0)}); onnxruntime {onnx_fps:.1f} frames/s, "
          f"PyTorch {torch_fps:.1f} frames/s (CPU, {args.threads} threads)")


if __name__ == "__main__":
    main()
