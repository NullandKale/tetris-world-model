# Tetris world model

A 2M-parameter world model of NES Tetris: a spatiotemporal MaskGIT transformer over exact palette pixels
(64-frame windows, 16 x 16 tokens). Every frame is generated from the frames before it and the
controller; no emulator runs while it plays.

**Play it:** the page in `web/` (published by GitHub Pages) runs the step-90,607 checkpoint with
onnxruntime-web on WebGPU (Chrome or Edge; elsewhere a slow WebAssembly fallback), from the very
beginning of a level-0 game. Arrows move and drop, X / Z rotate, Enter is Start; on a phone, the
on-screen controller.

## The code

- `src/token_world/models/dynamics.py`: the model, its training mask and loss, context corruption,
  MaskGIT decoding, the cached `Dreamer`.
- `scripts/train_dynamics_ui.py`: training on live emulator histories (a RAM-reading bot plays), with
  a run viewer; `scripts/run_until_stopped.ps1` runs it unattended.
- `src/token_world/diagnostics/long_dream.py`, `scripts/long_dream_check.py`: 128-frame no-button dreams
  against the real game.
- `scripts/export_onnx.py`, `src/token_world/models/onnx_export.py`, `web/`: the browser version, checked
  frame for frame against PyTorch (`web/test_dreamer.mjs`, `web/check_browser.mjs`).
- `docs/dynamics.md`: the design, the measurements and the decisions behind them.

Training needs **World NES** (the emulator, a separate package, not included) and **your own Tetris
ROM**: set `TETRIS_ROM` to your copy of `Tetris (USA).nes`. No ROM is included in this repository.

```
pip install -e .[onnx]
python -m pytest tests            # tests that need World NES or the ROM skip without them
python scripts/train_dynamics_ui.py
python scripts/export_onnx.py output/world_model_tetris_small/model_latest.pt
```
