# World model: spatiotemporal MaskGIT on exact pixels

`src/token_world/models/dynamics.py`, trained by `scripts/train_dynamics_ui.py`
(unattended: `scripts/run_until_stopped.ps1`). Tests: `tests/test_dynamics.py`.

This is the standard recipe (Genie's dynamics model), on exact frames instead
of a learned tokenizer. The training recipes tried from 2026-10-01 on (standard,
noborder, noborder_soft, stage1), with their reasons and results against the
original small model, are in `recipe_attempts.md`. It replaces the exact-token memory plan
(`exact_token_world.md`): a separately trained recall memory plus a generator.

## Model

- **Frame layout and border (`data/model_frames.py`, version `tokens-v2`,
  2026-09-30):** laid out to fit the 16 x 16 tokens. Token rows 1-14 hold NES
  lines 8-231, the picture a TV shows (NES line y at row y + 8, as before;
  the 8 overscan lines above and below change only with the screen, never in
  play, and are dropped). Token rows 0 and 15 are the border: 8 x 8 cells,
  four per token, each one solid border shade showing a base-3 digit. The
  frame counter takes 3 tokens (12 digits), and each state byte a game draws
  takes 2 (6 digits): for Tetris the fall timer (`$0045`, resets to 0 at each
  gravity drop, e.g. 37 before a drop at level 2) and the autorepeat counter
  (`$0046`). Timers drawn in every frame put their state into every frame's
  loss, not only the frames where they move something, and a dream carries
  them forward in its own frames. The RAM drawn is the one captured with the
  frame; a press at action t changes `$0046` only from frame t + 1 (0% of
  989 presses earlier), so the border never shows the next button. Before,
  8-row bands shared tokens with the game image and the counter's digits
  were scattered through every token. Runs record the version; `load_run`
  and the play app refuse other layouts, so every earlier run (4M, self, 8M,
  and the first small and large runs, renamed `*_oldborder`) is a logged
  baseline only, and the untied pixel head they used is gone.
- **Frames:** 256 x 256 palette indices in one index space for every game: NES
  colours 0-54, the game's three letterbox border shades 55-57
  (`data/nes_palette.py`). Input and target are exact.
- **Tokens:** each 16 x 16 patch is one transformer token, 256 per frame. A
  per-pixel colour embedding (58 -> 32) and a 16 x 16 stride-16 conv (ViT
  patchify). Learned spatial and temporal position embeddings.
- **Blocks:** spatial attention within a frame (bidirectional), then temporal
  attention at the same position over the frames so far (causal), then an MLP.
  The Contra runs used 12 blocks, width 512, 8 heads (62M parameters, block
  activations checkpointed). The Tetris run uses 6 blocks, width 128, 4 heads,
  a 16-dimensional colour embedding and no checkpointing: 4.1M parameters, of
  which 1.9M are the pixel head and 0.5M the patchify (see "Speed").
  `modern` blocks (the "modern" run, from scratch, 2026-10-01) are Llama-style:
  RMSNorm instead of LayerNorm, QK-norm (queries and keys RMS-normalised per
  head, so attention logits stay bounded) and a SwiGLU MLP with hidden 8/3 x
  width (the parameters of the 4 x width GELU MLP). They export to the browser
  unchanged (`tests/test_onnx.py`).
- **Soft, never argmax (2026-10-03, the `stage1` run on).** A generated
  frame is its colour probabilities. It goes back into the context as each
  pixel's probability-weighted colour embedding (`embed_probs`; kept as the
  frame's patch tokens, 98 KB a frame), is shown as each pixel's expected
  colour, and is scored by expectation (expected wrong pixels, expected
  piece mass). Nothing ever picks one colour: an unsure piece stays in the
  context as a ghost for the next frame to resolve, and training on the
  real frames after the model's own ghosts teaches it to make them definite.
  A frame is decoded in one pass from a fully masked frame (a second pass
  with the first pass's guess went with own guesses, 2026-10-04).
  `rollout_ghost_px` logs the pixels per generated frame whose most likely
  colour is under 90% (GHOST). What came before, and why it went:
  - argmax decoding (1 or more MaskGIT steps, confidence-first) erased a
    piece whose position was uncertain: its probability split over rows,
    and the one background colour won;
  - per-token codes (2026-10-01) moved that split to the codes instead of
    removing it, and erased too: the "modern" run, 56k steps, never learned
    the falling piece (piece +32 at most 0.36);
  - soft context on top of argmax output (`noborder_soft`, 2026-10-03) kept
    3/32 pieces at 4k-12k, the same as without it.
- **Actions:** the controller byte that produced frame t (8 button bits and a
  "none" flag for the first frame) is embedded and added to frame t's tokens.
- **Head:** all 256 pixels of a token, each a palette index. Tied to the
  input colour embedding, as BERT and MaskGIT tie their output to their token
  embeddings: each pixel gets a 16-number vector, and its logit for a colour
  is that vector against the colour's embedding (scaled by 1/4, as T5 scales
  its tied output) plus a bias per colour. That costs width x 4,096 weights
  instead of width x 14,848: at width 192, 0.79M instead of 2.87M. The fused
  cross-entropy builds the full [14,848, width] weight from the two parts
  each step (`Dynamics.head_weights`), so speed and memory are unchanged. The
  errors are in dynamics, not sharpness (0.07% of static pixels wrong), so
  the parameters go to the blocks. The first runs (4M, self, 8M) have an
  untied head of their own (`tied_head=False`, read from their saved args by
  `build`); it goes once the new large model beats the 8M.
- **Sizes trained now** (`MODELS` in the trainer), side by side:

  | | width x layers x heads | parameters | blocks | head |
  |---|---|---|---|---|
  | small | 96 x 8 x 4 | 2.01M | 1.19M | 0.40M |
  | large | 192 x 10 x 6 | 7.56M | 5.92M | 0.79M |
- **Random latent (optional, `latent=(groups, classes)`):** a sampled choice per frame, groups
  one-hot categoricals (DreamerV3), added to the frame's tokens like the action through a
  zero-initialised layer (`choice`), so a model grown from one without it starts identical. Training
  reads the choice from the real frame and the one before (`posterior`, straight-through samples) and
  teaches `prior` to predict it from the previous frame's features, attention-pooled (a learned query
  per group), and the frame's incoming buttons (KL balancing 0.5 / 0.1, no
  free nats: with DreamerV3's 1 per frame the choice took over gravity and Start); dreams sample it
  from the prior. Not in the ONNX export yet.
- **Memory:** the 64-frame window is the only memory. It covers Tetris's
  slowest timer (48 frames per drop at level 0). Scrolling needs nothing
  special: attention reaches the shifted position in the previous frame, and a
  step costs the same however many pixels change.

## Training

- **Masking (MaskGIT):** every masked frame draws its own rate from MaskGIT's
  cosine schedule, cos(pi/2 u) with u uniform, which favours high rates;
  that frame's tokens are replaced by a learned mask embedding at that rate.
  In half the windows a cut is drawn and every frame before it is fully visible:
  a clean context followed by frames to predict, which is exactly the
  generation condition. Frame 0 is always visible. The first run used Genie's
  single rate in [0.5, 1) per window, the second a uniform rate per frame; see
  "Pretraining and fine-tunes" below for why each changed.
- **Two stages, as HorizonDrive trains (since 2026-10-04;
  `examples/papers/horizondrive_2605.11596.pdf`, sec. 4.1-4.2, tab. 4):**
  - **The base model (`--models base`): teacher forcing.** Every window is
    real and clean; nothing else. HorizonDrive's base keeps its context clean
    (noise level 0) for 50k steps.
  - **Scheduled rollout recovery (`--models srr --grow-from <base checkpoint>
    --rollouts DEEPEST SHALLOWEST STEPS --blend MAX --refresh R`).** Every
    window is a rollout (`train_step`): from a real start s, a frozen copy of
    the model generates k frames, k uniform in 1..depth, one after another as
    in a dream; they go in as its soft frames, and the frames from the
    boundary s + k on are real and scored, every frame before the boundary
    visible (`training_mask(cut=...)`). The depth bound falls from DEEPEST to
    SHALLOWEST over STEPS steps (HorizonDrive's boundary-decay curriculum, 10
    -> 4 chunks: the deepest drift first). The blend (`own_share`, the
    history side of their eq. 8): the history's last w frames fade linearly
    from the copy's own to the real ones, w drawn uniform in 0..MAX each step,
    so the frame before the first scored one runs from nearly real to fully
    its own (a dream's condition). The targets are always the real frames.
    The copy is refreshed with the model's weights every R steps (2,000;
    HorizonDrive caches its rollouts per clip and refreshes them as often), so
    the rollouts do not chase the model being trained. Its soft frames' tokens
    are its own embedding, mixed with the live model's embedding of the real
    frames (they drift apart by at most R steps). Logged as `rollout_depth`,
    `blend_frames` (the step's w), `rollout_frames` (k) and `rollout_wrong`
    (pixels expected wrong). How we got here (`recipe_attempts.md`): the
    rollouts drawn by the model being trained, at lr 3e-4, lost the gravity
    clock within 400 steps; a frozen copy at lr 3e-5 with HorizonDrive's
    two-sided blend (radius 24, the targets partly the copy's own prediction)
    kept the clock but lost the next piece's spawn, test by test, as the
    model learned to imitate the copy where spawns happen.
  - **Removed with it:** context swaps and noise, the four sets of windows,
    the frame tags, own guesses (and with them the second decoding pass),
    Self Forcing and event tossing. The last recipe with them (stage 1, then
    the blend run `stage1_blend`) and its results are in
    `recipe_attempts.md`; that code is in this repository's history.
- **Loss:** cross-entropy over every pixel of masked tokens, all pixels at the
  same weight, always against the real frames. The letterbox tick border is
  in every frame and in the loss at full weight (the `noborder` runs,
  2026-10-02, tried a blank one and kept fewer pieces than standard; it may
  come back as a stage 3). Game, border and changed-pixel losses are reported, not weighted.
  Measured on the 8M model at step 50,056 (8 live windows, real inputs): the
  border is 6% of masked pixels and 50% of the gradient (sum of 1 - p over
  pixels); changed game pixels 0.16% and 15%; static game pixels 94% and 36%,
  almost all of it from the ~0.2% of static pixels the model is unsure of (a
  piece that could have fallen this frame). SimPLe-style loss clipping (no
  gradient for pixels already right) would drop 1.6% of the gradient at CE
  0.01 and 3.3% at 0.03, so it is not used: the loss already concentrates on
  the hard pixels.
- **Data:** live emulator histories only, windows of 64 frames (63 new frames
  per history per step). No eval set. Every window is trained on, so events
  come at the rate the game has them (the earlier event tossing kept every
  window with a rare event and few others; it went with the two-stage
  recipe). The emulators make several times the frames training uses: 16
  workers kept 18.3 windows/s (1,150 of 3,470 frames/s) with the GPU idle,
  against 12.6 used per second by the 8M run. Currently Tetris only, 16 histories per
  step (`GAMES` in the trainer). The first run (2 Tetris + 2 Contra, stopped at
  step 6,400 in `output/world_model`) had Tetris's static screen 0.1% wrong but
  Contra's static background 14-19% wrong.
- **Optimisation:** AdamW (lr 3e-4, betas 0.9/0.95, weight decay 0.05 on
  matrices), linear warmup (500 steps for the fine-tune, 1000 from scratch),
  gradient clip 1.0, bf16 autocast, torch.compile (about a minute per launch).
- **Memory (`--recompute`, on by default):** backward keeps only the inputs of
  the patch embedding and of each block and recomputes the rest (activation
  checkpointing); the per-pixel colour embedding, 16 values per pixel, is
  built 16 frames at a time directly in bf16. Same model and loss; gradients
  equal up to float rounding (test). Measured, training step with own
  guesses, while another run shared the GPU:

  | batch | mode | peak memory off | on | step off | on |
  |---|---|---|---|---|---|
  | 2 | eager | 2.24 GB | 0.84 GB | 258 ms | 292 ms |
  | 4 | eager | 3.79 GB | 1.00 GB | 483 ms | 559 ms |
  | 8 | compiled | 6.85 GB | 2.17 GB | 659 ms | 738 ms |

  Batch size equals the number of data workers (one live history each), and
  the CPU feeds ~3,500-4,100 new frames/s at 16 workers, so batch 16 is the most
  the data path supports without running several consoles per worker.
- **Averaged weights:** the checkpoint also keeps an exponential moving
  average of the weights (`ema`, decay 0.999, about the last 1,000 steps),
  which the play app, scenario checks and long-dream check load
  (`play_world_model.load_model`); training and previews use the raw weights.
- **Starting point:** a new run loads `--init` (model weights only, fresh
  optimizer); a run that already has its own `model_latest.pt` resumes that
  instead. The checkpoint records `init_step`.

## Pretraining and fine-tunes

**Pretraining** (`output/world_model_contra`, Contra only, 8 histories per
step, one mask rate in [0.5, 1) per window, stopped at step 18,000):

| steps | train loss | loss, changed pixels | loss, border | static wrong +1 | changed wrong +1 / +4 / +16 | border wrong +1 | previews |
|---|---|---|---|---|---|---|---|
| 0-1.2k | 0.615 | 1.184 | 0.778 | 23% | 63% / 68% / 74% | 29.7% | 5 |
| 1.2-4k | 0.111 | 0.293 | 0.213 | 18% | 67% / 72% / 75% | 10.9% | 14 |
| 4-8k | 0.056 | 0.189 | 0.034 | 16% | 65% / 69% / 73% | 7.3% | 20 |
| 8-12k | 0.033 | 0.125 | 0.012 | 16% | 67% / 69% / 74% | 7.6% | 20 |
| 12-16k | 0.024 | 0.095 | 0.007 | 17% | 66% / 70% / 73% | 7.7% | 20 |
| 16-18k | 0.021 | 0.097 | 0.007 | 15% | 65% / 70% / 72% | 7.1% | 11 |

The training losses fell the whole way and the loss curve looked normal, but
rollout quality stopped improving at about step 4,000. The border shows the gap
most clearly: its training loss fell 30x (0.213 to 0.007 per pixel) while the
border in generated frames stayed 7-8% wrong, although the counter follows
exactly from the previous frame.

The reason, confirmed by the diagnosis below: the training task was not the
generation task. In training every past frame was at least half masked; in
generation the whole past is clean, a condition training never produced, and
the model fails on it. (A first guess, that the partly visible target frame
made the training task mere in-frame infilling, did not hold up: with a partly
masked past, even a fully masked frame is predicted well.)

**Fine-tune 1** (`output/world_model_contra_ft`, stopped at step 1,600): the
same model and data from the step-18,000 weights, fresh optimizer, 500-step
warmup, an independent uniform mask rate per frame. Rollouts stayed at the
pretraining plateau (border +1 still 7.9% wrong).

**Diagnosis** (16 live Contra gameplay windows, frame 48 predicted in one
step; copying frame 47 gets 18.0% of game pixels and 4.8% of border pixels
wrong):

| checkpoint | frame 48 | past frames masked | game wrong | static wrong | changed wrong | border wrong |
|---|---|---|---|---|---|---|
| pretrain 18k | fully masked | 0% (clean, as in rollouts) | 24.6% | 16.3% | 62.6% | 10.2% |
| pretrain 18k | fully masked | 25% | 11.3% | 2.6% | 50.9% | 5.1% |
| pretrain 18k | fully masked | 50% | 7.8% | 1.6% | 35.8% | 4.0% |
| fine-tune 1.6k | fully masked | 0% (clean, as in rollouts) | 22.9% | 14.5% | 60.8% | 10.8% |
| fine-tune 1.6k | fully masked | 25% | 1.9% | 1.0% | 5.8% | 2.7% |
| fine-tune 1.6k | fully masked | 50% | 1.7% | 0.7% | 5.9% | 2.6% |
| fine-tune 1.6k | fully masked | 90% | 3.1% | 1.6% | 10.2% | 4.6% |

The model predicts the next frame well once any of the past is masked, and
fails only when the past is completely clean, a condition no training window
had produced (every frame after frame 0 always had some tokens masked). Rollouts
always give it a clean past. MaskGIT decoding was not the problem: a one-shot
argmax of the fully masked frame matched 4, 8 and 16 decode steps (24.6%, 26.1%,
27.2%, 26.9% game pixels wrong for the pretrained checkpoint).

**Fine-tune 2** (`output/world_model_contra_ft2`): from fine-tune 1's step-1,600
weights, fresh optimizer, 500-step warmup, with the clean-context windows and
cosine-schedule rates described under Training. Stopped at step 15,600 to move
to a smaller, faster model. Rollout error (wrong fraction; previews averaged
per bin):

| steps | static +1 / +4 / +16 | changed +1 / +4 / +16 | border +1 / +4 / +16 |
|---|---|---|---|
| pretraining plateau, 8-18k | 16.5% / 14.7% / 17.4% | 66% / 70% / 73% | 7.5% / 7.6% / 8.9% |
| fine-tune 2, 0-2k | 2.6% / 4.5% / 9.5% | 20% / 27% / 54% | 6.3% / 6.9% / 10.3% |
| fine-tune 2, 8-10k | 0.3% / 0.6% / 2.6% | 5% / 5% / 22% | 0.0% / 0.1% / 1.8% |
| fine-tune 2, 14-15.2k | 0.1% / 0.3% / 3.4% | 2% / 3% / 25% | 0.0% / 0.0% / 0.3% |

Up to 4 frames ahead the rollouts are nearly exact, and the border counter is
exact. At +16 the error on changed pixels stayed at 18-30% from step 8k on;
it is mostly content entering the screen and camera timing.

**Contra data.** Measured over 4 streams x 30,000 frames of the training
policy (`mixed`), before and after entering the Konami code at the title (30
lives instead of 3):

| | 3 lives | 30 lives |
|---|---|---|
| frames in play | 77.6% | 60.8% |
| death animation / respawn | 9.3% | 33.4% |
| deaths per 10k frames | 8.0-8.7 | 28-32 |
| new game (from stage 1) | every ~4,000 frames | every ~11,000 frames |
| play frames past stage 1 | 0% | 14% (one stream) |

With 3 lives every game ends within about a minute and play never leaves the
first ~2,700 px of stage 1. With 30 lives the terrain policy dies in a pit at
scroll 2,727 on every respawn, so deaths become a third of all frames. Contra
data needs a policy that gets past that pit before either setting helps.

## Speed

### Small model, paired run and profile (2026-09-30)

With both sizes on the GPU, nvidia-smi showed the GPU full and the CPU very
spiky: the GPU was the limit, the emulators filled their queues and slept.
Profiled alone (`torch.profiler`, one batch of 16 tossed windows, synced
phases), a paired step was small 744 ms + large 1,572 ms. Per model: backward
with recompute ~45%, training rollouts ~30% (each generated frame was five
uncompiled one-frame passes on 8 windows, ~250 kernel launches each: the CPU
took as long to launch them as the GPU to run them), the head loss ~10%.
The large model was paused (`--models`, default small) and the small one made
as fast as possible, each change profiled on an idle GPU:

| change | small step (unsynced) |
|---|---|
| before | 744 ms |
| rollouts through `FrameDecoder`: a fixed-size cache (every position allocated, later ones masked in attention), each frame's work captured once as a CUDA graph and replayed; launches per step 6,694 -> 2,756, rollout CPU 265 -> 23 ms | 698 ms |
| tied head computed as its two factors (width -> 256 x 16, then 16 -> 58) in the loss, own guesses and decoding, 2.3-2.8x fewer operations than the equivalent 14,848-wide map | 715 ms (noise; the uncompiled own-guess argmax was slower, then compiled) |
| no recompute for the small model (`MODELS[...]["recompute"]`; 18.5 GB in training) | 662 -> 636 ms |
| colour-embedding gradient as a one-hot matmul (`ColourEmbedding`): the scatter-add into 58 rows, mostly the background's, was the costliest kernel (81 ms a step); the matmul is 4x faster | 596 ms |
| the decode step compiled inside the graph (a 9-frame rollout 249 -> 191 ms) | not re-profiled |

Then the data path became the limit: a worker is 83% World NES emulation
(10% index conversion, 3% the bot), 16 workers make ~3,300 frames/s and
tossing kept ~18 windows/s, less than the ~25 a 600 ms step needs. So
training windows now start every 32 frames (`WINDOW_STRIDE`, now 16; the stream's
`stride`): each emulated frame is in two windows, at different positions,
and 16 workers keep 28.7 windows/s (55% kept). The scenario checks keep
one-frame-overlap windows.

Live, small model alone: 0.875 steps/s before (data waits 122 ms a step,
GPU 76% busy), 1.34 steps/s after (data waits 28 ms, GPU 97% busy). One-step
rollouts and corrupted context then gave 1.63 steps/s, and the stride is now
16 frames (each frame in four windows): data waits 3 ms, GPU 100% busy, 1.70
steps/s.

To stop a run without its window (the launcher's window is not reachable
from other sessions), create `output/stop_training`: the trainer checkpoints
every model and exits normally, which ends the launcher.

### Before (2026-09)

One optimizer step (batch 8, 64-frame windows, 16 x 16 patches, real masking
and loss, fused AdamW), RTX 3090:

| width / layers / heads | parameters (head / patchify) | compiled | step | peak |
|---|---|---|---|---|
| 512 / 12 / 8, checkpointed | 62.4M (7.6 / 4.2) | no | 2,065 ms | 12.7 GB |
| 256 / 6 / 8 | 12.3M (3.8 / 2.1) | no | 564 ms | 17.4 GB |
| 192 / 6 / 6 | 8.1M (2.9 / 1.6) | no | 466 ms | 14.7 GB |
| 128 / 6 / 4 | 4.1M (1.9 / 0.5) | no | 318 ms | 8.8 GB |
| 128 / 6 / 4 | 4.1M (1.9 / 0.5) | yes | 238 ms | 7.6 GB |
| 64 / 4 / 4 | 1.5M (1.0 / 0.3) | yes | 164 ms | 4.3 GB |

The large model was near the GPU's compute limit (~50 trillion operations per
step). Below width 128 the per-pixel head, patchify and data handling set a
floor of about 150 ms, so smaller models gain little.

### With own guesses (batch 8 profile, then batch 16 in real training)

Profiled alone on the GPU, a batch-8 step with own guesses took 255 ms, and
the head (256 pixels x 58 colours per masked token) cost about as much as the
transformer: its logits were stored, recomputed and copied to float32.
`HeadCrossEntropy` (fused linear cross-entropy: each chunk's logits become
loss and gradients at once, then are dropped) and a compiled argmax for the
guesses brought it to 207 ms; loss equal, gradients within bf16 rounding.

600-step trials from the own-guesses checkpoint at batch 16 (16 workers, one
history each) under real training load:

| change | steps/s | windows/s |
|---|---|---|
| baseline, recompute off | 1.83 | 29.3 |
| recompute on | 1.87 | 29.9 |
| workers at below-normal priority, trainer on 2 CPU threads | 2.08 | 33.3 |
| + per-window async GPU copies, finiteness checked only at checkpoints | 2.12 | 33.9 |
| + CUDA graphs (`torch.compile` mode reduce-overhead) | 1.88 | 30.1 |

Batch 8 had trained at 28.6 windows/s. The same step alone on the GPU runs at
2.8 steps/s at batch 16, so the remaining gap is the CPU: 16 emulator
processes on 16 threads beside the trainer. Data never waited (1 ms). CUDA
graphs were slower (each replay copies the 67 MB batch into its static input)
and were removed. cuDNN attention works on this build but saves only ~4 ms per
step (faster spatially, slower temporally), so it is not used. Stacking the
batch on the CPU before copying had blocked the main thread ~127 ms per step
in isolation; in training the fix was worth ~2%.

Data path alone (RAM bot, DataLoader, idle machine): 2 workers 1,120 new
frames/s, 4: 2,147, 8: 3,454, 12: 3,806, 16: 4,102.

## Tetris runs

`output/world_model_tetris` (stopped at step 2,600): Tetris only, 8 histories
per step, the 4.1M-parameter model above, from scratch, 1,000-step warmup, the
clean-context masking, 64-frame windows, previews every 200 steps. On the
nes-py data path a step took ~0.47 s: 176 ms of it waiting for data, because
workers shipped float32 RGB frames (~400 MB per step).

`output/world_model_tetris_worldnes` (stopped at step 131,000): the same run
on World NES frames, llm-thing19's brains until step 99,800, then the
RAM-driven bot (random game lengths from 113,800). Changed game pixels wrong
in the previews, mean per window of steps:

| steps | changed loss | +1 | +2 | +4 | +8 | +16 | static +16 | border +16 |
|---|---|---|---|---|---|---|---|---|
| 90-97k (old brain) | 0.78 | 21% | | 43% | | 66% | 0.09% | 0.15% |
| 100-110k (new bot) | 0.47 | 29% | 27% | 27% | 35% | 48% | 0.24% | 0.38% |
| 110-121k | 0.31 | 22% | 21% | 24% | 32% | 46% | 0.28% | 0.29% |
| 121-131k | 0.31 | 18% | 19% | 22% | 27% | 40% | 0.07% | 0.07% |

`output/world_model_tetris_guesses` (stopped at step 4,240 for speed work):
fine-tune from that step-131,000 checkpoint with own guesses and the guessed
flag, batch 8.

`output/world_model_tetris_self`: own guesses and rollouts together (half the
batch each), batch 16, from the step-131,000 checkpoint (fresh optimizer,
1,000-step warmup). Compare its horizon curve against the 121-131k row; judge
it after 20k+ steps.

`output/world_model_tetris_8m`: twice the parameters (8.0M: width 192, 8
layers, 6 heads), own guesses and rollouts, batch 16, from scratch, ~0.79
steps/s at 235 W. By step 50,056 changed pixels wrong at +16 fell from 58% to
31%. Resumed from 50,056 (copy kept as `model_step50056.pt`) with patient
pieces, rollouts of 1-32 frames, event tossing and averaged weights, and
stopped at step 50,686 for the pair below.

`output/world_model_tetris_small` and `output/world_model_tetris_large`: the
two sizes above, tied heads, from scratch, trained in one process on the same
batches (`--models`, default both): one set of 16 emulator workers feeds one
batch per step and each model takes its own step on it, so the CPU-bound
data path is shared and the curves compare window for window. Each has its
own optimizer, averaged weights, checkpoints and metrics; the dashboard shows
both (a line style per model). Judge after 20k+ steps; compare against the
8M at the same step and on the scenario and long-dream checks at 20k, 50k and
70k.

## Data: World NES

Frames now come from World NES (`../llm-thing23`, installed editable into the
trainer's Python), which the user accepted on hardware accuracy (it passes the
NES test-ROM suites and is being checked against Mesen) rather than
bit-for-bit parity with nes-py. `data/world_nes_tetris.py` runs one World NES
console per DataLoader worker in external mode, played by the same Tetris
session as before (the level-menu script with sampled start levels, restart
and freeze watchdogs) with a RAM-driven bot in place of llm-thing19's brains
(below). Workers convert the
palette plane to model index frames themselves (`data/model_frames.py`: colour
map, letterbox, tick border identical to llm-thing19's add_game_border) and
send uint8, 12x less data than before. Consecutive windows share one
observation (63 new frames each). The console, frame checks and windows are
one loop for every game (`data/world_nes_windows.py`).

**Contra** runs on World NES too (`data/world_nes_contra.py`, details in
`contra_data.md`): the nes-py session's navigator and terrain policy, moved
onto observations, drive one console per worker through boot, the title
(Konami code, 30 lives), deaths and continues, never resetting it. Its frames
use the same layout with its own steel-blue border shades and no state bytes
yet (the soldier generation timer `$7A` is the candidate). Same windows,
stride and restart count as Tetris, no event tossing. `GameBatches` builds it
when `"contra"` is in `GAMES` (default Tetris only). Measured beside a live
run: 380 new frames/s for one worker, 359 per worker for two (Tetris 399 for
one, same conditions); 58% of frames in play, 35% death and respawn: the
policy dies at the pit at scroll 2,727 on every life, so all of it is the
first ~2,700 px of stage 1.

New frames per second through the DataLoader, one console plus the Python
Tetris policy per worker (16 CPU threads):

| workers | new frames/s | per worker | batch of that many histories every |
|---|---|---|---|
| 1 | 359 | 359 | 0.18 s |
| 2 | 723 | 362 | 0.17 s |
| 4 | 1,317 | 329 | 0.19 s |
| 8 | 2,060 | 258 | 0.24 s |
| 12 | 2,267 | 189 | 0.33 s |
| 16 | 2,491 | 156 | 0.40 s |

World NES emulates about 330 frames/s per core, roughly half nes-py's speed
(its cycle-accurate PPU costs CPU). With 8 workers a batch arrives every
0.24 s, matching the small model's 0.24 s GPU step; more workers add little,
because the CPU is saturated. (Measured with llm-thing19's brains; the bot
below is cheaper.)

### The Tetris bot

llm-thing19's perfect brain found the falling piece from pixels and cleared
under 2 lines per 1,000 frames at level 0, topping out after ~8,000 frames.
`data/tetris_bot.py` reads everything from RAM instead: piece column, row
and orientation (0x40-0x42, the 19 orientation shapes checked against 134
piece locks), the next piece (0xBF) and the settled board (0x400). On each
spawn it drops every move of the piece and every reply of the next piece
straight down (a straight drop rests on the column tops alone, so the search
is a few array operations), scores the boards with the classic weights
(height, lines, holes, bumpiness), then taps A and Left/Right on press edges
and holds Down. The session plays each game with a brain switcher.

The perfect brain almost never tops out, and NES Tetris crashes somewhere
past level 150 (World NES halted on a KIL opcode twice in the first hour of
endless perfect play), so every game has a length, then the player gives up.
`BrainSwitcher` makes each game long or short. Rare events (top-out, curtain,
menus, game start) come once per game and everything else scales with
frames, so most games are short while about half the frames come from long
ones:

- long games (8%, 20,000-45,000 frames): the perfect brain only, from start
  levels 0-9, so they climb through every palette with a level up every 10
  lines (NES Tetris delays the first level up to 100 lines from start levels
  10-19) and reach levels 20+;
- short games (1,500-3,500 frames): perfect 60% / random 25% / idle 15% at
  random intervals, from start levels 0-19: variety and sloppy boards;
- giving up: reckless (random legal placements, soft-dropped) 75% or random
  25% until the stack tops out; the session starts a new game through the
  menu, and the console is never reset;
- in half the games the perfect brain keeps column 9 open and builds for
  tetrises; 30% of its pieces, once lined up, fall by gravity alone for
  30-180 frames before the soft drop (no-button falling was missing: every
  model froze pieces at slow gravity); every brain resets when it takes over
  (a stale target from an earlier piece once steered the falling one).

Census, 4 streams, ~630,000 frames each (one event per N frames):

| | first bot | long/short games | + patient pieces (uncapped) | + 30-180 frame cap |
|---|---|---|---|---|
| falling alone | | | 432 | 295 |
| single | | | 622 | 383 |
| triple | 34k | 10.4k | 15.9k | 15.3k |
| tetris | none | 27k | 42k | 63k |
| level up | 2.4k | 4.6k | 19.9k | 6.2k |
| top-out | 12k | 5.6k | 6.0k | 5.9k |

Uncapped patience (a patient piece fell the whole way) cost too much time at
slow levels; the cap fixed level ups. Since 2026-10-01 half the patient
pieces over a stack of 8 rows or fewer spin while they fall (A or B at
random, 1 frame in 8), then turn back to the plan and drop: rotations went
from 20.6 to 39.9 per 1,000 playing frames (30,000 frames, 3 games each).

Event census (2026-10-02, 4 games x 50,000 frames, RAM events; per 1,000
in-game frames) and the holding brain it led to. The other brains tap and
soft-drop, so held shifts (the game's autorepeat), landings by gravity and long
untouched falls were rare, while a person in the browser holds the arrows and
the long-dream test lets pieces fall untouched. The holding brain plans like
the perfect one but holds Left/Right until the column, holds A/B a few frames
per turn and never presses Down; it takes 25% of short games' brain time
(perfect 45%, random 20%, idle 10%); long games stay perfect-only, since from
level 15 a piece falls the well in ~40 frames, faster than autorepeat moves it.

| per 1,000 in-game frames | before | with the holding brain |
|---|---|---|
| held shift (autorepeat) | 0.45 | 2.52 |
| lock by gravity, then spawn | 1.92 (14% of spawns) | 3.20 (25%) |
| untouched falls of 200+ frames | 1 stretch | 44 (8.8% of frames) |
| untouched falls of 96+ / 48+ frames | 7.7% / 13.0% of frames | 16.1% / 22.8% |
| soft drop / gravity / shift / rotate | 155 / 33 / 30 / 22 | 118 / 40 / 27 / 18 |
| spawn | 13.7 | 12.8 |
| single / double / triple / tetris | 2.45 / 0.46 / 0.10 / 0.015 | 2.14 / 0.32 / 0.06 / 0.025 |
| top-out / game start / level up | 0.18 / 0.20 / 0.17 | 0.21 / 0.23 / 0.14 |

Still never in training: SELECT during play (it hides the next piece). Spawns
draw on screen one frame after the RAM shows them; the entry delay from lock
to spawn is 10-18 frames (median 12-14). Games now last median 3,300 frames,
mean 5,100, and 44% of in-game frames come from games over 20,000 frames.

Forced perfect brain, 30,000 frames per start level:

| start level | 0 | 9 | 18 | 19 |
|---|---|---|---|---|
| top-outs | 0 | 0 | 0 | 0 |
| lines per 1,000 frames | 6.6 | 6.8 | 7.1 | 7.3 |

Policy cost per frame fell from 1.36 ms (the first scalar search) to
0.07 ms: the search was vectorized (54 to 2.7 ms per plan, identical choices
on 200 test boards), and the session's freeze check compares raw palette
planes instead of converting each frame's colours first. With the full brain
mix, four streams made 0-3 in-game restarts per 63,000 frames.

## Measured cost (RTX 3090, before training)

| tokens / frame | context | batch | step | peak |
|---|---|---|---|---|
| 256 (16 px) | 64 | 4 | 1.14 s uncompiled | 10.1 GB |
| 256 (16 px) | 32 | 8 | 0.86 s compiled | 9.9 GB |
| 1024 (8 px) | 16 | 4 | 0.98 s compiled | 5.8 GB |

Flash attention is not in the Windows PyTorch build; the memory-efficient
kernel is used. That, and quadratic spatial attention, is why 8 px tokens cost
4-5x more per frame.

## Every model, every check

Every check and app below takes any model, the pixel model or a layered one (`docs/guides/layered_tokens.md`),
through one interface, `diagnostics/worlds.py`: `load(run or checkpoint)` gives a World that rolls batches of
windows out from real context (`rollout`) or dreams one game live (`dreaming`). Its frames are model frames
either way: the pixel model's soft (scores are expectations), a layered model's committed and composed from its
layers (scores count pixels). The checks' streams are layered and their frames composed from the layers, exactly,
so all models see the same frames. Each script takes run folders or checkpoint files and `--device cpu` while a
run holds the GPU, and appends its summary with the checkpoint's step to the run folder (`scenarios.csv`,
`event_checks.csv`; an archived checkpoint's to `output/<run>`), which the window's **Scenarios** tab shows, runs side
by side.

## Scenario checks

`scripts/tetris_scenarios.py <run> [<run> ...]` scores checkpoints on specific
game events (`diagnostics/tetris_scenarios.py`). The bot plays fresh World NES
streams; events are detected in console RAM and each becomes a 64-frame
window with the event's first visible change at generated frame +2, so the
model must predict it. (Aligning on the screen, not the RAM frame: the top-out
curtain starts more than 16 frames after the play state turns to game over,
and the line-clear animation only steps every few frames.)
All runs are scored on the same instances (same `--seed`, same instances;
nothing is stored between runs). The first version kept the first 24 of each
event from 8 streams, so frequent events all came from the first minutes of 8
games (low levels, empty boards) and differences were within noise; spacing
instances 1,000 frames apart still filled frequent events within the first
minute (15-18 games). Now instances are spaced in time: per stream, one of a
scenario every minutes x streams / per seconds (72 s for 15 min, 8 streams,
100 per scenario), so every scenario is drawn from the whole run, scored in
batches as they arrive, and each mean comes with a 95% interval from
resampling whole games, plus the number of games and the levels covered.

Scenarios: gravity with no buttons, falling alone, soft drop, shift, rotate, spawn, line
clears (single, double, triple, tetris), level up, top-out, game start.
Scores at +2, +8 and +16: event pixels wrong (of the pixels that really
changed since the last context frame, the tick border and the random NEXT box
excluded), playfield false changes (static playfield pixels the model
changed), and exact (no wrong event pixel at +2). It writes `scenarios.csv`
and one contact sheet per scenario to `output/scenarios/<time>/`, and each run's
means to its run folder's `scenarios.csv`.

## Event checks: restarts and line clears, followed to the end

`scripts/event_checks.py <run> [<run> ...] [--checks restart line_clears]`: long events found live, each model
dreaming them from real context fed the bot's real buttons (`World.follow`); sheets go to
`output/event_checks/<time>/`.

- `restart` (`diagnostics/restart.py`): a top-out, then the curtain, the menus and a new game, 320 frames from
  the last 48 before the top-out. Whole-screen pixels wrong at milestones: `curtain` (4 frames before the level
  menu), `menu` (16 into it), `play` and `play64` (8 and 63 frames into the new game), with copying the top-out
  frame as the floor (`copy_menu`, `copy_play`).
- `line_clears` (`diagnostics/line_clears.py`): from 10 frames before a clear starts, 60 frames: the animation,
  the rows gone, the stack dropped. Scored on the well's cells (20 x 10, at x = 96 + 8 col, y = 56 + 8 row) at
  the last frame: `full_rows` the dream keeps (0 in the real game), `cells_wrong`, and `mass` (filled cells over
  the real game's: above 1, rows kept). Averaged over all clears and per size (single .. tetris). The scenario
  checks score only a clear's first 16 frames.

Events come from `data/tetris_events.py` (shared with window tossing).
Measured while building it: the picture shows the RAM of the frame before
it; falling-piece cells sit at x = 92 + 8 col, y = 52 + 8 row in model
frames (100% of sampled cells); play states are 1 falling, 2 lock, 3 check
rows, 4 line-clear animation (~18 frames), 5-8 counters and spawn, 10
top-out; the level byte (0x44) steps at every 10 lines.

## Playing and long dreams

`scripts/play_world_model.py` is a Tk app: the real game in World NES and one
model's dream side by side (real | dream | difference, drawn into one image
and copied to the canvas once per frame), in lockstep, one generation at a
time. You or the bot play; the model dreams from the last 48 real frames with
the same buttons. It lists only long-trained runs (20,000+ steps), pixel or
layered, and loads their averaged weights.

`scripts/long_dream_check.py [run or checkpoint ...]` checks that pieces fall and keep their
shape: at a fresh piece, nobody presses anything for 128 frames, and each
model dreams the same 128 frames; 2 trials per start-level band (0-4, 5-9,
10-14, 15-19). It prints playfield pixels wrong at +16/+32/+64/+128 and block
mass at +128 against the real game (1.0 = as many filled pixels), and writes
strips to `output/long_dream/<time>/`. Baseline before patient pieces and
longer rollouts:

| run | +16 | +32 | +64 | +128 | mass (range) |
|---|---|---|---|---|---|
| 8M @ 50,056 | 1.55% | 1.64% | 2.02% | 2.82% | 0.82 (0.62-1.04) |
| self @ 13,800 | 1.07% | 1.75% | 2.24% | 2.98% | 0.88 (0.72-1.01) |
| worldnes @ 131,000 | 1.97% | 2.11% | 2.35% | 3.06% | 0.83 (0.72-1.00) |

At level 4 no model let a piece fall, and shapes morphed; 1, 4 or 16 decode
steps made no difference. The `falling alone` scenario (no buttons for 16
frames after the player lets go) scores the same thing at short horizons.

The scores (`diagnostics/long_dream.py score`, the same in training's tests
every 2,000 steps). Since 2026-10-04 the falling piece is scored by position
too: the base run's step-14,000 dream kept "24 of 32 pieces intact" by
pixel count while its piece sat frozen at the spawn point.

| score | what it measures |
|---|---|
| `wrong_h` | playfield pixels expected wrong at +16/+32/+64/+128 |
| `mass` | filled pixels (p(block) >= 0.5) at +128 over the real game's |
| `piece_h` | the piece's filled pixels over the real one's, anywhere in the upper playfield (kept, not placed) |
| `piece_hit_h` | the share of the real piece's pixels the dream fills: a frozen or lagging piece scores 0 |
| `piece_ghost_h` | filled pixels where the real frame has none, over the real piece's: stuck, doubled, smeared |
| `fall` | the dream piece's drop over the real one's (centres), up to 64 frames: 1 at the real speed, 0 frozen |
| `spawn`, `spawn_lag` | the next piece appears within 16 frames of the real one's, after the first left the spawn area |
| `timer_wrong_h` | the border's fall-timer cells expected wrong: the gravity clock the model reads |
| `unseen_patch` | changed tokens whose exact 16 x 16 patch never occurs in real play (`diagnostics/coherence.py`) |
| `unseen_change` | token changes (a patch, then the next frame's there) never seen in real play |
| `change_px`, `unsure_px` | pixels changing per frame (frozen or flickering dreams), most likely colour under 0.9 |
| `presence` | of the frames with a real piece in the upper playfield, the share in which the dream draws any piece (right or wrong) |
| `activation` | the dream's expected block pixels there (however faint) over the real piece's |
| `shape` | of the frames with one whole real piece (4 cells) in the well's top 11 rows, the share in which the dream holds the same cells, wherever they are (a swapped, garbled or erased piece scores 0) |

`presence` and `activation` watch for the stall: trained toward fewer wrong pixels, a model can erase the pieces
it is unsure of (an empty well is mostly right), and with nothing drawn where pieces fall it has no wrong piece
left to correct toward a right one; it has to draw wrong pieces before right ones. The training window's status
line warns when presence is under 0.25 or falls by more than 0.15 between tests (and says so when wrong pixels
improve at the same time); the run viewer plots it with the piece scores and shows it in the Compare tab.

The last four are generic: no game knowledge, and a different legal outcome (another piece, another
place) scores as well as the real one, while blends, ghosts, half pieces and morphs do not. Training
adds every history's window to the bank of real patches and changes every 16 steps, hashed on the
GPU, and keeps it in the run folder (`patch_bank.npz`); the real future scores 0.5% / 0.1% against a 160k-frame bank.

The play app's **change weight** multiplies the odds of every pixel change by w as it decodes
(`weigh_changes`): an unsure piece is drawn with too many cells instead of four spread thin. It takes
effect on the next frame. Measured on stage B (docs/guides/recipe_attempts.md), w = 2 or 3 makes
every long-dream score worse; the dreams and the browser decode at 1.

## Generation and previews

**Decoding, measured 2026-10-02** (small 90.6k, no training, 12 games, 512-frame
no-button dreams, the same games for every row; `Decoding.order` added):

| decoding | level | wrong +16 / +128 / +512 | mass @512 | piece +32 | intact |
|---|---|---|---|---|---|
| 1 step | 0 | 0.51% / 2.52% / 14.55% | 0.99 | 0.98 | 11/11 |
| 2 steps, most confident first | 0 | 0.32% / 1.96% / 6.51% | 1.00 | 0.98 | 11/11 |
| 2 steps, random order | 0 | 0.57% / 2.62% / 15.71% | 1.03 | 0.94 | 11/11 |
| 4 steps, most confident first | 0 | 0.68% / 2.32% / 6.26% | 0.99 | 0.99 | 11/11 |
| 4 steps, random order | 0 | 0.87% / 2.85% / 20.14% | 0.91 | 0.88 | 10/11 |
| 8 steps, random order | 0 | 1.13% / 3.51% / 21.23% | 1.01 | 0.86 | 9/11 |
| 4 steps, random, temperature 0.5 | 0 | 0.38% / 3.29% / 14.80% | 0.96 | 0.96 | 11/11 |
| 1 step | 2 | 1.72% / 4.03% / 21.41% | 0.92 | 0.79 | 7/11 |
| 2 steps, most confident first | 2 | 1.81% / 3.62% / 21.26% | 0.91 | 1.01 | 10/11 |

Two steps, most confident first, at level 0 halve the error at +512 and keep
every piece. Random order (1X's open Genie reference, which finds
confident-first copies the previous frame for its VQ tokens) is worse here, and
gets worse with more steps; sampling does not help. Level 0 beats level 2 at
every horizon, as GameNGen finds noise in training and little or none at
inference best. The papers and reference code behind these choices are in
`examples/` (not committed; `examples/README.md`).

A frame is decoded from a fully masked frame in 4 MaskGIT steps: argmax picks,
the most confident tokens (summed pixel log-probability) unmasked first on a
cosine schedule. Frames after the one being decoded cannot influence it, so a
rollout keeps one 64-frame window.

Temporal attention is causal and spatial attention stays within a frame, so
a frame's features depend on earlier frames only through each layer's
temporal keys and values. `rollout` runs the real context through the model
once, keeping those in a `TemporalCache`, then decodes each frame by running
only its 256 tokens, and writes the finished frame's keys and values in. A
test checks it gives the same frames as decoding against the full window. On
the step-131,000 checkpoint, 8 live Tetris windows, bf16, while training
shared the GPU: 16 frames took 1.1 s cached against 72 s for the uncompiled
full-window decode, with identical pixels.

Every 200 steps the preview generates the last 16 frames of each window one by
one from the first 48 real frames and the real actions, and scores per game at
horizons 1, 2, 4, 8, 16: wrong game pixels overall, on pixels that changed since
the last context frame (copying that frame gets exactly these wrong), on static
pixels, and on the border, next to what copying the last context frame gets
wrong on the border (`border_copy`). Four of the windows, from the most
changing to the quietest, are saved as animated GIFs (`previews/`: real |
generated | wrong, frame by frame).

Every 2,000 steps (`--long-dream-every`) the trainer runs the long-dream test
on each model's averaged weights: 32 trials, the same games at every test
and in every run with the same `--seed` (one low-priority worker plays them
from the seed at launch), so steps and runs are compared on the same games;
until 2026-10-01 it was 6 new games every 1,000 steps, too few to tell runs
apart. They are dreamed at corruption level 2 and level 0 with 1 decode
step. Rows are appended to
`long_dream.csv` per trial, with the first trial animated in
`long_dream.gif`. A test takes under 10 s (8 s at step 29,000). The checkpoint of every tested step is
copied to `D:/token_world_checkpoints/<run folder>/model_step<step>.pt`
(`--archive`; 32 MB each for the small model), so a run that gets good and
then worse can go back to the step whose long dream was best.

## In the browser (ONNX, WebGPU)

`web/` plays the model in the browser: every frame is generated from the
frames before it and the keyboard (arrows, X / Z for A / B, Enter, Shift), on
WebGPU through onnxruntime-web, falling back to WebAssembly without it.
Published by GitHub Pages from `site/` (`scripts/ship.py`):
https://nullandkale.github.io/tetris-world-model/.

```powershell
python scripts/export_onnx.py [checkpoint.pt]     # -> web/model/ (needs pip install .[onnx]); since 2026-10-07 the
                                                  # page has the layered model's step 58,000 (layered_tokens.md)
python -m http.server 8000 -d web                 # open http://localhost:8000 in Chrome or Edge
```

The export (`models/onnx_export.py`) writes the layered model's averaged
weights as three graphs, plus the page's start: `context.bin`, one level-0 game
start as its layers (3 frames, 8 frames into the game, after the playfield's
camera switch), and `context.json` (where each layer sits, the buttons, the
palette). A frame is kept as its content (each token's soft pixels and the
backdrop, [543, 96]):

- `embed.onnx`: the start's layers (cells, cell_known, sprite_layer, border,
  backdrop; uint8) -> their content.
- `prefill.onnx`: content [47, 543, dim] and incoming actions -> the key/value
  cache, one keys and one values tensor a layer, positions 0..46 filled. The
  length is fixed (a shorter start is padded after its frames): a dynamic
  length put about 190 shape nodes on the CPU in onnxruntime-web.
- `step.onnx`: the previous frame's content, actions, position and the cache
  -> the new frame composed (cells, the sprite picture over them, the border
  bands; each pixel's most likely colour, through the palette in the graph,
  all in float so it stays on the GPU), its content, and the previous frame's
  keys and values [8, 543, 4, 24], which the host writes into the cache.

A dream holds the start's camera, so every token keeps its identity and the
routed temporal attention is attention per token (each layer's keys are
[543, 4, 24, 64], transposed as attention reads them, its values [543, 4, 64,
24]). The model's camera head moves the camera at a game
start (the menus and the playfield are different nametables), so the start is
taken after that switch; the browser dream stays on the game screen (after a
top-out it cannot change screens). A step is one pass over two frames, the
previous one written into the cache (the Dreamer's commit pass) and the new
one decided, as the pixel model's export did.

Checks, against the model's own Dreamer at temperature 0 on the page's start
with no buttons: the export script (onnxruntime, Python; 96 frames, 0 pixels
differ, a window slide included), `web/test_dreamer.mjs` (`dreamer.js` on
onnxruntime-node: as PyTorch), `tests/test_onnx.py` (small models with a fixed
camera and with camera tokens, through a window slide), and
`web/check_browser.mjs` (the page in Chrome with `?check`, or `--firefox`: as
PyTorch).

Speed (2026-10-07; a laptop's integrated AMD GPU played at 4 frames/s):

- Every node of a step and of the re-encode runs on the GPU: one on the CPU
  stops the GPU for a round trip. ArgMax's int64 indices had sent the picture's
  composition to the CPU, and the dynamic prefill length its shape arithmetic.
- The graphs never copy the cache. A step attends over the cache's earlier
  frames and its own two in one softmax and returns only the previous frame's
  keys and values; `web/dreamer.js` writes them in place with a compute shader
  on the GPU buffers the cache stays in (`onnx_dreamer.py`, and `dreamer.js` on
  the CPU, write them into arrays). Passing the cache through whole rewrote all
  214 MB of it every step, copied about five times (Gather, Where, Concat,
  Transpose): 12-14 ms of a 34 ms step's GPU time on the RTX 3090 (shared with
  training), and integrated GPUs have far less memory bandwidth.
- A step is still about 660 kernels, 32 ms of GPU time on the shared 3090; the
  matrix products are 40% of it, the patch convolutions that re-embed the soft
  picture about 1 ms each.

The page logs to the console (`web/diagnostics.js`, lines starting
`[world-model]`): the browser, the WebGPU adapter (vendor, architecture,
software fallback), its features and limits, or why it has none; each file's
load and each graph's setup time; a lost device or GPU error; and every 5 s the
frame rate and a step's times (the graph's run, the cache write, the draw, the
re-encodes). `?profile` adds onnxruntime's profile of each graph after 120
frames, summed by kind of operation and by node (GPU time from timestamp
queries) with every node placed on the CPU; `?backend=wasm|webgpu` and
`?verbose` (onnxruntime's own log) help on other browsers.

## The training window and the run viewer

The training window is a status line and Stop Training over the run viewer
(`ui/run_viewer.py`), which reads everything back from the run folders every
10 s and when the trainer reports a preview or a long dream:

- **Curves:** loss, changed-pixel error at +1/+4/+16, border error against
  copying, static and corrupted fractions, long-dream error at +16..+128 and
  block mass, by step. Archived `metrics*.csv` files are merged, so the curves
  survive a metrics schema change. Runs passed with `--compare` are drawn dashed
  or dotted on the same axes.
- **Previews:** the four preview GIFs, playing.
- **Long dream:** the latest long-dream GIF and a table of the tests so far.
- **Horizons, Speed, Stats:** the latest error by horizon, steps/s and data
  waits, and the run's settings.

`scripts/view_runs.py run ... [--compare run ...]` opens the same viewer
without training, for example on another session's run or afterwards.
