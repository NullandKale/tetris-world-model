# Tetris world model

A 6.5M-parameter world model of NES Tetris on exact palette pixels: 256 x 256 frames (the game's screen
between a border that shows the game's state) in 16 x 16 patches, 256 tokens a frame. A spatiotemporal MaskGIT
transformer (9 layers 176 wide, 64-frame windows, rotary time, rotary space, the buttons on every block's norms)
generates every frame from the frames before it and the controller; no emulator runs while it plays. Each
frame is two layers, the background and the sprites (the falling piece and the next one), and both pixel ends
are small conv pyramids (2 x 2 steps between a token and its pixels). Each frame's motion (per 8 x 8 cell:
still, moved from where, or new) goes in beside its pixels and comes out of every token, and a dreamed frame
draws its motion first, then its pixels given it (VideoJAM's joint appearance and motion), which is how A and B
turn a piece the right way. Trained by masked next-frame prediction on live emulator play, changed pixels
weighed 10 times: 135,582 steps from scratch, then post-trained to step 151,606 on players that press like
people (buttons held, Down held through turns) with each window carrying the last one's keys and values
(Transformer-XL's memory), as the browser's ring of frames does, then to step 184,000 with frame skip: each step
1 to 6 frames on, with every frame's buttons and a frame clock, so a slow device plays at the game's speed.

**Play it:** the page in `web/` (published by GitHub Pages) runs the step-184,000 checkpoint with
onnxruntime-web on WebGPU (Chrome or Edge; elsewhere a slow WebAssembly fallback), from the level-select menu
(63 frames, as both layers: pick a level, then Start) or from any picture ("Start from an image": all of it background,
so a falling piece in it is read as stack). Arrows move and drop, X / Z rotate, Enter is Start; on a phone, the
on-screen controller.

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
set FLAGS=--space-rope --qk-norm --action-norm --motion --motion-first --local-attention --two-layers ^
    --change-weight 10 --clean-hidden 0.5
python scripts/train_dynamics_ui.py --models s9_w176_conv %FLAGS% --stairs 72.5 --stair-lrs 3e-4 1.9e-4 1.2e-4 ^
    --stair-shares 0.8344 0.0828 0.0828 --final-cooldown 12000               # from scratch, to step 135,582
python scripts/train_dynamics_ui.py --models s9_w176_conv %FLAGS% --post-train <its model_latest.pt> --carry ^
    --hold-lr 1.2e-4 --final-cooldown 4000                                     # Start Cooldown at step 147,606
python scripts/export_onnx.py <the run's model_latest.pt>
```
