# Layered tokens: a design for review (2026-10-06)

A world model whose tokens are the NES picture processor's own layers: a background fixed to the world,
a short list of sprites, and the camera. The tokenizer stays free (exact, no training), the shipped
model is still code, weights and one start screenshot, and the decisions that blurred dreams (which piece
comes next, where a sprite is, how far the camera moved) become categories the model commits to.

Built 2026-10-06 (section 11); first run pending: it needs the rebuilt World NES installed, which waits
for run 4 to stop.

## 1. Why

- **Where today's dreams go wrong.** In stage B's long dreams, 84% of the patches never seen in real play
  are in Tetris's next-piece box (base: 80%; `recipe_attempts.md`, "Where the incoherence is"). At a spawn,
  the model draws a blend of the possible pieces, soft decoding feeds it back, and it stays blended. A
  per-pixel head cannot express "piece A or piece B", only each pixel's probability, so committing pixel
  by pixel erases or speckles. Every reference model commits at inference to tokens that are codes (Genie,
  IRIS, MaskGIT), or denoises in several steps (DIAMOND, GameNGen).
- **Scrolling.** On SMB, today's screen tokens change 17.7% per frame in levels, just from the camera, and
  their vocabulary passed 28,000 after 20,000 frames and was still climbing. A bank of real patches cannot
  commit them: sub-patch scroll offsets and sprites drawn over background make endless new patches.
- **Not shippable: the PPU's tile codes.** Rendering from nametable and OAM tile codes needs the pattern
  tables, which are ROM data. Network weights are fine (the art is mixed in); an exact tile table is not
  (memory: ship-model-only). Every token here is decoded by the network.

## 2. The capture (World NES, done)

New regions in `llm-thing23` (`ppu.hpp`, `ppu.cpp`, `console.cpp`, `bind_console.cpp`, `batch.py`; built
in `build/layers_wheel`, installed once training lets go of the module):

| region | per visible pixel (240 x 256) |
|---|---|
| `ppu_background` | the background's colour, `4 * palette + colour` (0: transparent) |
| `ppu_sprites` | the sprite's colour, `palette << 2 \| colour`, `\| 0x20` when behind the background (0: none) |
| `ppu_sprite_index` | the OAM slot that drew it, + 1 (0: none) |
| `ppu_scroll` | 960 bytes: each scanline's background x (0..511) and y (0..479) as it is drawn |

With `oam` and `palette_ram` they rebuild every frame exactly (SMB 20,000/20,000 frames and 4,869/4,869
in levels from tokens, Tetris 5,989/5,989 from tokens). The palette turns them into final NES colours at
training time; no palette RAM, ROM data or RAM reaches the model's inputs at play time.

## 3. The tokens of a frame

All in final NES colours, through today's colour embedding and patchify.

| layer | tokens | what a token is | Tetris | SMB (levels) |
|---|---|---|---|---|
| background | the world-grid cells in view, at most 288 slots | the 16 x 16 pixels of a cell of the 512 x 480 nametable canvas (sky included), and its canvas row and column | 240 (no scrolling) | ~261, at most 269 |
| sprites | 32 slots, in OAM order | the sprite's 8 x 8 pixels, x, y, in front / behind, present / empty | 8 (4 falling piece, 4 next piece) | ~14, at most 27 |
| camera | one per token row (15) | that row's scroll x and y | constant | 2 regions: status bar (rows 0-1), level (rows 2-14) |
| border | as now | the tick border's band (memory: tick-border-intentional) | | |

Measured choices behind it (`layer_tokens.py`, `smb_slots_splits.py`, session scratchpad):

- **Fixed to the world.** In SMB's levels, of the ~261 background cells in view, ~80 hold more than sky and
  ~1.5 change per frame (median 0): new columns coming into view, a block being hit. A cell keeps its
  canvas position as the camera moves, so its history lines up in time like Tetris's well.
- **Sprites in OAM order.** 94% of SMB's on-screen sprites keep their slot frame to frame (same tile, within
  8 px), 1% move to another slot; Tetris 98% and 0%. The PPU's own order is stable enough to be the slot
  identity: no set matching (DETR-style) needed.
- **A camera per token row.** SMB's only split is at line 32 (a token-row boundary) in every frame;
  Tetris never splits. Splits elsewhere would need a camera per scanline: not needed for these games.
- **Vocabularies** (what the network must learn to draw): Tetris 377 background cells and 7 sprite tiles;
  SMB about 1,650 background cells and 273 sprite tiles after 5,000 level frames.

## 4. The model

Today's spatiotemporal transformer, with more kinds of token per frame (about 300 for SMB, 280 for
Tetris: similar to today's 256).

- **Spatial attention** over all of a frame's tokens: background cells, sprite slots, camera rows, border.
- **Temporal attention** per token identity, causal: a background cell attends to the same world cell in
  earlier frames (it lines up across camera moves); a sprite slot to that slot; a camera row to that row.
  A cell coming into view has no history: masked, as an unseen frame is today.
- **Embeddings:** background, the cell's canvas row and column (learned, 30 x 32); sprites, the slot,
  x and y (learned per value) and the flags; camera, the row. The incoming buttons are added to every
  token of the frame, as now.

## 5. What it predicts, and the losses

| output | form | loss |
|---|---|---|
| camera: each row's scroll change | category, -8..+8 px per frame in x and y (and "jump" for a level change) | cross-entropy |
| sprite slot: present | category | cross-entropy |
| sprite slot: x, y | category: the change from the slot's last position, -8..+8 (and "placed" for a new sprite, then its x, y as 256 / 240 classes) | cross-entropy |
| sprite slot: pixels | 8 x 8 colours, the network's pixel head | cross-entropy on present slots |
| background cell: pixels | 16 x 16 colours, the pixel head, as now | masked cross-entropy, as now |

**The motion loss is the camera and sprite movement categories.** Motion is explicit, its own outputs,
with no reweighting of pixels (decode-time reweighting hurt every score; `recipe_attempts.md`). Predicting
a movement rather than a position is the same for every place on the screen, which is what the model
should learn once.

## 6. Decoding a frame (a dream)

1. The camera: each row's movement, sampled from its categories (or the most likely). The cells in view
   follow from it; cells newly in view get predicted from scratch.
2. The sprites: each slot's present and movement (or placement), sampled; then its pixels.
3. The background cells: their pixels, as now (soft or committed; nearly all are copies).
4. Compose by the PPU's rule (a sprite pixel shows when the background is transparent or the sprite is
   in front; lower slots first), which is code, not data: the frame.

The decisions are categories, so a sampled dream commits: the next piece in Tetris is four cells, each
placed at one position from a category (it cannot be half one piece and half another), and gravity moves
the falling cells by exactly 8 px on some frame. Committing has to be learned as part of training (rollouts
decode this way), not added afterwards.

## 7. The recipe (one stage)

From `recipe_attempts.md` and memory single-stage-goal:

- Masked training as now (MaskGIT masking per token, Genie-style rates to check), from step 0.
- Rollouts ramping in (a frozen copy refreshed every 2,000 steps; the history blend on the last few
  frames; real targets); kept on to the end: teacher forcing after rollouts undid stage B in 2,000 steps.
- A learning-rate cooldown at the end: stage B drifted after its peak at a constant rate.
- The per-frame random latent is left out: the categories carry the decisions it never learned to carry.

## 8. Measuring it

On the composed frames, so it compares with every run so far: the Tetris long-dream scores (hit, ghost,
fall, spawn, timer), the generic coherence scores (unseen patches and changes against real play), the
incoherence breakdown by region (the next-piece box above all), and the game-end restart check. SMB gets
the generic scores and its own events later.

Built (2026-10-06): every check and app takes any model through `diagnostics/worlds.py` (docs/guides/dynamics.md,
"Every model, every check"): the scenario checks (`scripts/tetris_scenarios.py`), the event checks (restarts and line clears,
`scripts/event_checks.py`), long dreams (`scripts/long_dream_check.py`), the play app
(`scripts/play_world_model.py`), and in training the long-dream scores with the bias test
(`diagnostics/bias.py`: which next pieces a dream chooses, against real play) and presence (does the dream
draw any piece at all; the stall warning) and shape (does the falling piece stay the piece it is). The
window's Scenarios tab shows each run's latest scenario and event checks.

## 9. Open questions

- Sprite movement as a change from the last position, or the absolute position? (A change generalises;
  an absolute position is simpler for sprites that appear.)
- 8 x 16 sprites (PPUCTRL bit 5): a slot then holds 8 x 16. Neither game uses them.
- Games that switch pattern or palette mid-frame, or split anywhere but a token row: a camera per scanline
  and per-line palettes. Not needed for Tetris or SMB.
- The PPU's 8-sprites-per-line limit makes some sprites flicker; the slots show it as data, which the model
  must learn (SMB rotates sprites through OAM for this).
- The page: fixed shapes (288 background + 32 sprite + 15 camera + border tokens); its start screenshot
  stored as that frame's layers.

## 10. Build order

1. Install World NES with the layer regions (needs run 4 stopped: its data workers hold the module).
2. A layered stream (Tetris first): live windows of layered tokens in final colours, with tests that they
   rebuild every frame exactly.
3. The model's layered inputs and outputs, the losses, the dream decoder, the composer; CPU tests
   (causality, exact rebuild through the model's token format, a dream runs).
4. The training script's layered mode and scores on composed frames; a short GPU check; then the run.

## 11. What is built (2026-10-06)

| piece | where | checked |
|---|---|---|
| layer capture | llm-thing23 (`ppu.hpp`, `ppu.cpp`, `console.cpp`, `bind_console.cpp`, `batch.py`, `docs/interfaces.md`) | rebuilds every SMB and Tetris frame exactly; line 0's scroll is recorded after the pre-render line's vertical copy (it was stale) |
| layered frames | `data/nes_layers.py` (`LayerCanvas`, `compose`, `model_frame`) | `tests/test_nes_layers.py`; Tetris 3,998/4,000 frames exact (every playing frame), SMB 3,999/4,000 (one split at line 33) |
| layered stream | `world_nes_windows(..., layered=True)`, `WorldNesTetrisStreams(layered=True)`, `LiveBatches` stacks it | the same windows as the plain stream, frame for frame |
| the model | `models/layered.py` (`LayeredDynamics`, `layered_loss`, `layered_mask`, `LayeredDreamer`) | `tests/test_layered.py`: routed temporal attention equals attention per identity; frames never see later frames; the cache equals the full window with the camera scrolling; the losses train; a dream slides and composes |
| trials and scores | `RealTetris(layered=True)`, `trials(layered=True)`, `dream_layered` | a layered trial's last frame composes to its real model frame; a dream is scored as every run |
| trainer | `scripts/train_layered.py`; `run_until_stopped.ps1 -Run layered` | imports; needs the GPU |
| the training window | `train_layered.py` opens `ui/dynamics_app.py` as the pixel trainer does; previews, horizon curves, the rollout sheet and border vs copying come from `diagnostics/run_outputs.py`, which works on frames from any model (the layered previews dream each live window's last 16 frames, committed) | `tests/test_run_outputs.py` |
| stage B | `train_layered.py --rollouts --blend --refresh --grow-from` (`train_step`), `models/layered.py rollout` (a batched `LayeredDreamer`), `layered_loss(seen=)` | `tests/test_layered.py`, `tests/test_run_outputs.py`: a rollout equals the dream of those frames; a batch of dreams equals each dream alone; a stage B step trains; movement is scored from the history's positions |

Sizes: 2.53M parameters (96 wide, 8 layers, 4 heads), 367 tokens a frame (255 cells, 64 sprite slots,
15 camera rows, the frame token, 32 border tokens), 592 temporal identities.

One change from section 6: the camera is decided one frame ahead. A frame laid out by its true camera
would tell its own camera head the answer in training (its cells' world coordinates), and in a dream
there is no true camera; so the next frame's camera comes from this frame's camera tokens and the next
frame's buttons (`camera_logits`), then the frame is laid out and its content predicted.

The losses are summed unweighted (pixels as mean cross-entropy per pixel, the categories per token);
the categories start larger (ln of 240-512 places) and fall fast.

**Stage B** (built 2026-10-06, as the pixel `srr`): every window a rollout. A frozen copy of the model
(refreshed every 2,000 steps) dreams k frames (1..depth) from a real start with the real buttons, as a long
dream does: its sprites, flags and camera committed, its pixels kept as its tokens (the pixel rollouts keep
theirs as tokens too: soft pixels for 62 frames x 12 windows would be ~11 GB). They go in as the history;
the frames after are real and scored. The blend fades the history's last w frames to the real ones (tokens
blended; the committed parts its own while its share is at least half). A sprite's movement target is from
where the history put it to where it really is, so the targets lead back to the real game.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -Run layered -TrainerArgs "--out output\world_model_tetris_layered_srr --grow-from <stage A checkpoint> --rollouts 62 16 8000 --refresh 2000 --blend 8 --lr 3e-5 --warmup 500 --weight-decay 1e-5 --workers 12 --bank output\world_model_tetris_layered\patch_bank.npz"
```

**Launch (after run 4 stops):**

```powershell
cd ..\llm-thing23; python -m pip install -e . --no-deps; cd ..\llm-thing22      # the layer regions
python -m pytest tests/test_nes_layers.py                                           # the live test runs now
python scripts/train_layered.py --steps 300 --long-dream-every 300 --out output\layered_smoke   # memory, speed
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -Run layered -TrainerArgs "--workers 12 --bank output\world_model_tetris_latent\patch_bank.npz"
```

Batch 16 does not fit the 24 GB card (21.3 GB at peak, 26.4 GB reserved: it spilled into system memory
and slowed to a crawl); batch 12 peaks at 15.9 GB and runs 1.9 steps a second (23 windows a second, the
old base 32). Launched 2026-10-06 01:40 (`world_model_tetris_layered`).


## 12. Two layered models (2026-10-06)

The slot design above (sections 3-6) ran first and was dropped at 44k: its pieces broke apart in dreams.
It ran without a working clock (one learned embedding per window slot; the border's frame counter was
right 18-43 times in 128, teacher-forced), so every sprite's drop was a coin flip of about 2% a frame.
Both models now share `models/layered.py`, with rotary time in the temporal attention:

| | `--model pixels` (`models/layered_pixels.py`) | `--model slots` (`models/layered_slots.py`) |
|---|---|---|
| sprites | the screen's sprite picture (the sprite pixels that show), 240 16 x 16 tokens, its own pixel head | the 64 OAM slots: 8 x 8 pixels, x, y, flags; movement and presence as categories |
| a dream | everything soft, as the pixel model | the sprites committed (snapping), decided over `--dream-steps` passes, the surest first |
| tokens a frame | 543 | 367 |
| run folder | `output/world_model_tetris_layered_px` (running since 10:28) | `output/world_model_tetris_layered_slots` |

The data carries both forms (`sprite_layer`, and `sprites`, `sprite_xy`, `sprite_flags`); `compose`
draws the slots by the PPU's rule when a frame has no sprite picture, and both rebuild real Tetris frames
exactly. One trainer, one window: previews, horizon curves, the rollout sheet, long dreams, each model's
own losses, the Compare tab (one table per compared run) and stage B (`--rollouts`, `--blend`,
`--grow-from`) for both. Training uses activation checkpointing and a 12 GB allocator cap (`--vram-gb`).

```powershell
# the slot model with rotary time, compared with the layered pixels and the pixel stage A
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -Run layered -TrainerArgs "--model slots --dream-steps 8 --bank output\world_model_tetris_latent\patch_bank.npz --compare output\world_model_tetris_layered_px output\world_model_tetris_base"
```

## 13. Options for the next run (built 2026-10-06, not run)

From the attention papers in `examples/papers` (index in `examples/README.md`), and the camera, four architecture options, off
unless a run's config sets them (the first layered-pixels run has none; a resumed run keeps its own):

| option | what | from | parameters |
|---|---|---|---|
| `--share-pixels` | the sprite picture goes in and out through the cells' patch and pixel head (the token kind tells them apart) | the user: back to about 2M | 3.03M -> 2.24M |
| `--registers N` | N learned tokens per frame, no target, never hidden, their own temporal identity: room for what a frame is about that no pixel holds (the next piece, from the preview box to the spawn) | ViTs need registers (2309.16588); Perceiver IO | +96 each |
| `--fixed-camera` | for a game whose camera never moves (Tetris): no camera tokens and no camera head; a dream keeps the camera its real frames had. Scrolling games (SMB) leave it off | the first run's head only ever learned "no move" (camera loss 5e-4 since 20k), and its dreams never moved the camera (0 of 1,024 frames, 30k) | -0.21M; 15 fewer tokens a frame |
| `--space-rope` | rotary in the spatial attention by screen place, RoPE-Mixed: half of each head's channels turn along learned 2D directions (diagonals included), so a score depends on the offset between two tokens. Cells (x less the fine scroll), sprite-picture patches and border tokens have a place; camera, frame, registers and the slot model's sprites do not (their rotated channels are zero, as DeepSeek-V2's decoupled RoPE) | RoPE for ViT (2403.13298), Qwen2-VL's M-RoPE | +48 a layer |

Checked on the 28k checkpoint first (CPU, live trials; scratch `sink_scratch_check.py`), and left out:
- **an attention gate** (Qwen's gated attention, against attention sinks): there is no sink. From the hidden
  last frame, every layer puts 0.00-0.01 of its temporal attention on the window's first frame (0.021 would
  be even), and 0.15-0.67 on the frame before;
- **registers as an artifact fix:** no hijacked tokens: empty sprite patches' residual norms are the occupied
  ones' (about 300 against 310), no outliers in any layer. Registers stay as the relay above, not a repair;
- **a compacted temporal identity grid:** the canvas's unused cell identities are 4% of attention pairs, and a
  data-dependent size would break `torch.compile` in every block.

With all four: 2,030,678 parameters and 536 tokens a frame (543 less 15 camera tokens, plus 8 registers). Speed
is about the current run's (0.76 steps/s): parameters are not where the time goes, the tokens are. Skipping empty sprite patches (Run-Length Tokenization, 2411.05222) is the speed, and needs a
decision first: which patches a dreamed frame gives tokens to (the occupied ones and a margin, and where things
enter).

```powershell
# the 2M run, compared with the first layered-pixels run at matched steps
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -Run layered -TrainerArgs "--out output\world_model_tetris_layered_2m --share-pixels --fixed-camera --registers 8 --space-rope --bank output\world_model_tetris_latent\patch_bank.npz --compare output\world_model_tetris_layered_px output\world_model_tetris_base"
```
