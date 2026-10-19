# Tetris world model

A 2.7M-parameter world model of NES Tetris on exact palette pixels: 256 x 256 frames (the game's screen
between a border that shows the game's state) in 16 x 16 patches, 256 tokens a frame. A spatiotemporal MaskGIT
transformer (9 layers, 64-frame windows, rotary time, rotary space, the buttons on every block's norms)
generates every frame from the frames before it and the controller; no emulator runs while it plays. Each
frame's motion (per 8 x 8 cell: still, moved from where, or new) goes in beside its pixels and comes out of
every token, and a dreamed frame draws its motion first, then its pixels given it (VideoJAM's joint appearance
and motion), which is how A and B turn a piece the right way. Trained by masked next-frame prediction on live
emulator play: three layers grown to nine by stacking at step 48,000, then to step 140,000 and a learning-rate
decay to step 152,000.

**Play it:** the page in `web/` (published by GitHub Pages) runs the step-152,000 checkpoint with
onnxruntime-web on WebGPU (Chrome or Edge; elsewhere a slow WebAssembly fallback), from the start of a
level-0 game (three frames) or from any picture ("Start from an image"). Arrows move and drop, X / Z rotate,
Enter is Start; on a phone, the on-screen controller.

**Slow or broken?** The browser console (F12) says which GPU it runs on, how long each step takes and any
GPU error, in lines starting `[world-model]`. Add `?profile` to the address for each graph's GPU time by
operation after 120 frames, `?backend=wasm` to run on the CPU, `?verbose` for onnxruntime's own log. A
laptop may give the browser its integrated GPU (Windows ignores the page's request for the fast one).

## The code

- `src/token_world/models/dynamics.py`: the model, its training mask and loss, the cached `FrameDecoder` and
  `Dreamer`; `src/token_world/data/tetris_motion.py`: each frame's motion, read from the console's RAM.
- `scripts/train_dynamics_ui.py`: training on live emulator histories (a RAM-reading bot plays), with a run
  viewer; `scripts/run_until_stopped.ps1` runs it unattended.
- `src/token_world/diagnostics/`: 1,000-frame dreams with no buttons and with the bot's (`long_dream.py`),
  game events (`tetris_scenarios.py`), line clears and restarts followed to their end (`line_clears.py`,
  `restart.py`), the next piece's choices (`bias.py`) and game-agnostic coherence scores (`coherence.py`),
  for any model through `worlds.py`; `scripts/button_swap_check.py` and `motion_swap_check.py`: whether a
  turn follows A vs B.
- `scripts/export_onnx.py`, `src/token_world/models/onnx_export.py`, `web/`: the browser version, checked
  frame for frame against PyTorch (`web/test_dreamer.mjs`, `web/check_browser.mjs`).
- `docs/`: the design, the measurements and the decisions behind them (`recipe_attempts.md`: every run).

Training needs **World NES** (the emulator, a separate package, not included) and **your own Tetris
ROM**: set `TETRIS_ROM` to your copy of `Tetris (USA).nes`. No ROM is included in this repository.

```
pip install -e .[onnx]
python -m pytest tests            # tests that need World NES or the ROM skip without them
set FLAGS=--space-rope --qk-norm --action-norm --motion --motion-first --local-attention
python scripts/train_dynamics_ui.py --models s3 %FLAGS%                     # three layers, to step 48,000
python scripts/train_dynamics_ui.py --models s9_stack --grow-from <s3 step 48,000> %FLAGS%   # stacked to nine
python scripts/train_dynamics_ui.py --models s9_stack --cooldown 140000 12000 %FLAGS%       # from step 140,000
python scripts/export_onnx.py <the run's model_latest.pt>
```
