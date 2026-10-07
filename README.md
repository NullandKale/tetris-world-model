# Tetris world model

A 3.0M-parameter world model of NES Tetris over the NES's own layers: the background as cells fixed to
the world, the sprites as their own picture, and the border, 543 tokens a frame of exact palette pixels in
16 x 16 patches. A spatiotemporal MaskGIT transformer (64-frame windows, rotary time) generates every frame
from the frames before it and the controller; no emulator runs while it plays. Trained by next-frame
prediction from clean real context, 58,000 steps ending in a learning-rate decay.

**Play it:** the page in `web/` (published by GitHub Pages) runs the step-58,000 checkpoint with
onnxruntime-web on WebGPU (Chrome or Edge; elsewhere a slow WebAssembly fallback), from the start of a
level-0 game (one frame's layers). Arrows move and drop, X / Z rotate, Enter is Start; on a phone, the
on-screen controller. The dream holds the playfield's camera, so it stays on the game screen.

## The code

- `src/token_world/models/layered.py`, `layered_pixels.py`: the model, its training mask and loss, the
  cached `Dreamer`; `src/token_world/data/nes_layers.py`: the layers, composed back into the screen exactly.
- `scripts/train_layered.py`: training on live emulator histories (a RAM-reading bot plays), with a run
  viewer; `scripts/run_until_stopped.ps1` runs it unattended.
- `src/token_world/diagnostics/`: 128-frame no-button dreams against the real game (`long_dream.py`), game
  events (`tetris_scenarios.py`), line clears and restarts followed to their end (`line_clears.py`,
  `restart.py`), the next piece's choices (`bias.py`) and game-agnostic coherence scores (`coherence.py`),
  for any model through `worlds.py`.
- `scripts/export_onnx.py`, `src/token_world/models/onnx_export.py`, `web/`: the browser version, checked
  frame for frame against PyTorch (`web/test_dreamer.mjs`, `web/check_browser.mjs`).
- `docs/`: the design, the measurements and the decisions behind them.

Training needs **World NES** (the emulator, a separate package, not included) and **your own Tetris
ROM**: set `TETRIS_ROM` to your copy of `Tetris (USA).nes`. No ROM is included in this repository.

```
pip install -e .[onnx]
python -m pytest tests            # tests that need World NES or the ROM skip without them
python scripts/train_layered.py --cooldown 46000 12000
python scripts/export_onnx.py output/world_model_tetris_layered_px/model_latest.pt
```
