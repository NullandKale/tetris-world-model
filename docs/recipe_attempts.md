# Recipe attempts, 2026-10-01 to 10-03: chasing the original small model

The world model (`dynamics.md`) is a Genie-style ST-MaskGIT on exact Tetris
pixels. This page records the four training recipes tried after the original
small model, why each was tried, what it got, and the goals and assumptions
of the one running now (`stage1`). Every result here is on the same test.

## The goal

**Very long dreams that stay coherent.** A falling piece stays whole, falls
at the right speed, lands, locks, and the next one spawns, for hundreds of
frames with no real frames to lean on. Two side goals:

- outputs that are definite, not ghosted;
- a browser model that runs at 30+ frames a second.

## The test

The long-dream test (`diagnostics/long_dream.py`) runs every 2,000 steps
during training, on 32 live games:

- Each game is paused at a freshly spawned piece. The model gets its last 48
  real frames, then dreams 128 frames with no buttons pressed.
- **intact:** the falling piece's pixels at +32 are within 0.75–1.33 of the
  real game's.
- **erased:** under 0.25 of them are left.

Pixel error is reported too, but it rewards erasing an uncertain piece:
blank pixels cost less than a piece drawn one row off. So runs are judged on
intact and erased.

Runs with the same `--seed` are tested on the same 32 games. Running a GPU
run's numbers again on the CPU moves the result by about ±4 games, so
smaller differences are noise.

## The baseline: the original small model (to 90.6k)

96 wide × 8 layers × 4 heads, 2.0M parameters. **28/32 intact, 0 erased at
52k steps** (`score_paired.py` on the archived checkpoint). At 90.6k it kept
all 11 pieces on the older 12-game test. Every attempt below uses the same
model size, so the gap is in the recipe.

The original's recipe changed twice during the run:

| steps | context corruption | rollouts | data |
|---|---|---|---|
| 0–24.6k | **own guesses**: masked tokens filled with its own argmax guesses, plus a per-token "guessed" flag | 1–32 frames (4 decoding steps until 21.7k, then 1) | perfect / random / idle brains, **event tossing** |
| 24.6k–90.6k | **swaps**: real tokens copied from the same place 1–8 frames earlier or from elsewhere in the frame, up to 30% then 60% of a frame's tokens, with a level tag 0–9 | 1–32 frames | the same |

Other settings, unchanged throughout:
- **Border:** frame counter, fall timer and autorepeat digits drawn in the frame.
- **Window:** 64 frames.
- **Output:** per-pixel argmax.

It finished with a learning-rate cooldown from 90.6k to 101k. Its 90.6k
checkpoint dreamed better than the cooled-down one.

## Attempt 1: standard (2026-10-01/02)

**Why:** match what the papers do. GameNGen and Genie train with teacher
forcing and context noise, never on their own rollouts. Self Forcing uses
the model's own rollouts only in a short post-training stage.

**What changed from the original:**
- no rollouts (teacher forcing);
- Gaussian context noise up to 0.7 instead of swaps (10 levels, GameNGen);
- no tossing;
- new brains: holding and spinning added for rare events;
- long dreams decoded in 2 steps, most confident first. On the 90.6k model
  this halved the error at 512 frames (14.55% → 6.51%).

**Assumption:** the reference recipe should match or beat our own.

**Result:**

| steps | 8k | 16k | 24k | 32k | 46k (best) | 52k | 56k |
|---|---|---|---|---|---|---|---|
| intact | 5 | 7 | 9 | 13 | 17 | 15 | 14 |
| erased | 24 | 14 | 7 | 7 | 8 | 8 | 9 |

It never got near 28/32, so the assumption was wrong for this game.

**Post-training on top of standard** (each a short fine-tune):

| what | result |
|---|---|
| aggressive tossing | 10/32 intact, 13 erased at 8k (best 14 at 6k). Helped line clears (5/12 vs 1/12), not pieces. |
| Self Forcing: per-frame real targets after rollouts of 32–56 frames, gradients through the rollout | 2/32 intact, **29 erased** at 2k. A long rollout drifts out of step with the real game, so the real frame contradicts the model's own context, and drawing nothing is the cheapest answer. |
| GAN critic, factor 0.1 (8-frame temporal critic) | 7/32 intact, 13 erased at 2k: no gain. |
| GAN critic, factor 1.0 | 12/30 intact, 11 erased at 4k (within noise), but the previews dissolved into grain. Static pixels wrong at +16 jumped from 0.1% to 7–10%. Adaptive weighting only balanced the two pulls at the pixel head; deeper in, the critic's pull was unbounded. |

## Attempt 2: noborder (2026-10-02, to 27.9k)

**Why:** the user's recipe after standard fell short:
- border off, so timing has to come from history;
- 96-frame windows, so history can cover the slowest gravity (48 frames per
  row at level 0);
- rollouts as deep as the window (1–94, log-uniform);
- continuous noise, 0–0.75 per frame, with a float tag;
- every brain, no tossing;
- 1-step decoding.

**Assumptions:**
- a longer history can replace the fall timer in the border;
- deeper rollouts teach longer coherence.

**Result:**

| step | noborder intact / erased | standard intact / erased (same step) |
|---|---|---|
| 12k | 1 / 26 | 7 / 20 |
| 22k | 5 / 22 | 5 / 11 |
| 26k | 3 / 26 | 7 / 6 |

Changed-pixel loss was worse than standard's from ~12k (0.85–1.0 vs
0.56–0.79). In the dreams the piece fell at about the right speed but
changed shape: an S became a bar, lost blocks, and landed as one column. So
the timing came through from history; the piece's identity did not.

Pieces that were gone at +32 were already gone by +16. Training on deeper
rollouts can't fix a loss in the first 16 frames.

## Attempt 3: noborder_soft (2026-10-03, to 15.5k)

**Why:** the user's idea. Feed the model's own frames back as soft colour
probabilities instead of argmax, so an unsure piece stays in the context as
a ghost instead of being rounded away.

**What changed:** only that (`--soft-rollouts 0.5`). Everything else was
exactly noborder.

**Result:** 3/32 intact and 28 erased at every test from 4k to 12k, the same
as noborder. Block mass and 128-frame error were a little better (mass
0.72–0.77 vs 0.66–0.69). Pixels wrong in its own rollout frames were 2–3×
noborder's at 4k–6k, then level by 8k.

Soft context alone didn't help, so the piece loss came from what both
noborder runs shared.

## What the four runs showed

- **The original recipe is the only one that kept pieces.** Every change
  away from it measured worse, or no better.
- **The model's size isn't the problem.** Every run had the same 2M model.
- **What only the original had:**
  - swaps;
  - own guesses in its early training;
  - tossing from step 0;
  - a border with all its runs.

  Standard had the border and still fell short, so the border alone isn't it.
- **Rollout depth isn't what decides piece survival.** Standard, with no
  rollouts, beat noborder, with deep ones. Real-frame targets after long
  rollouts teach erasure (Self Forcing).
- **Pieces are lost early.** The noborder runs lost them within 16 frames of
  spawning.
- **Distribution losses** (the GAN) can damage frames in ways the piece test
  doesn't see. Watch the previews too.

## Attempt 4: stage1 (running since 2026-10-03 12:45)

Designed with the user after reviewing the four runs and the references
(`examples/`: Copilot-4D, 1X's open Genie, HorizonDrive, MiniWorld, GameNGen,
Self Forcing). The design and its code are in `dynamics.md`.

| part | stage 1 | from where |
|---|---|---|
| output and context | **soft only**: no argmax anywhere. A frame is its colour probabilities, kept as context (as patch tokens), shown as expected colours. | the user: uncertainty is information passed between frames, and the model should learn to be definite, not be rounded |
| corruption | four sets per batch: noise only (0–0.75); light swaps 0–20% + noise 0–0.25; heavy swaps growing along the window + noise 0–0.75; rollouts with light corruption | swaps from the original. Amounts from Copilot-4D and 1X's open Genie ((u × 20)%; the heavy growth from their "corrupting later frames more") |
| tags | per frame: swapped share and noise | GameNGen-style, per frame |
| own guesses | half of every set predicts again with its first guess added to the masked tokens (self-conditioning, soft, no flag) | the original's first 24.6k steps |
| rollouts | soft, depth uniform 1–62 | the user: as deep as possible. HorizonDrive (its distilled student): 20-chunk unrolls beat 4 |
| data | tossing on, every brain | the original had tossing from step 0 |
| border | on | the original; noborder lost |
| window, model, optimiser | 64 frames, small (96 × 8 × 4), AdamW β₂ 0.95, learning rate 3e-4 with warmup | unchanged |

**Goals for stage 1:**
1. Match or beat the original's 28/32 intact, 0 erased, by ~50k, on the same
   games.
2. Become definite where the game is definite: unsure pixels in its own frames
   (`rollout_ghost_px`) and static pixels expected wrong should fall toward
   the old models' ~0.1–0.2%. Keep honest doubt only where the game is
   genuinely uncertain: the next piece, or exactly when gravity drops.

**Assumptions, and what would show each one wrong:**

| assumption | wrong if |
|---|---|
| Swaps, own guesses and tossing are what the original had that later runs lacked | stage1 stays near standard (≤ 15/32 intact) at 30–50k |
| Soft frames don't erase an uncertain piece the way argmax did | pieces still vanish within 16 frames while the board stays sharp |
| Heavy, growing corruption teaches repair, not distrust | it never becomes sure of static pixels (expected wrong stays above ~1% past 20k) |
| Uniform deep rollouts help, as in HorizonDrive | pieces survive +32 but fall apart over 64–128 frames |
| Being unsure early is the model honestly not knowing, not a failure to learn | it is still hazy everywhere at 20k, when earlier runs had learned the furniture |

**What changes in the scores:**
- **Old and new scores aren't directly comparable.** Dreams are soft, so
  pixel error is an expectation (1 − the probability of the real colour).
  Earlier runs' errors counted argmax misses, which hides doubt: a pixel 60%
  sure counted as right.
- **Piece and mass now count a pixel as filled when it is at least 50% likely
  to be a block** (`SURE`). The first stage1 test summed probabilities
  instead, and a 2–3% haze over the empty well scored as a kept piece (step
  2000: "0 erased", while the sheet showed the piece gone). Those rows were
  set aside in `long_dream_haze_score.csv`.
- **The trials depend on `--seed` alone** (fixed on 2026-10-03). Before, a
  relaunch reseeded them from the step it resumed at, which broke the pairing
  between tests. stage1 is tested on the same 32 games as standard and
  noborder.

**Early signs (to step 3,000):**
- **Furniture:** learned on the usual schedule. Game loss was 0.73 at 600,
  as in earlier runs.
- **Unsure pixels per generated frame:** fell from all 65,536 to about 1,000
  by 2.8k.
- **Static pixels expected wrong at +16:** fell to 0.9%.
- **The falling piece:** gone in the step-2000 sheet, as in every earlier run
  at 2k.

Judge it at 20k+ (furniture first, pieces after: `training-timescale` in
memory), against the tables above.

## Stage 1's result (stopped at 90.1k, 2026-10-04)

It never learned to keep the falling piece. On the same 32 games, every test from 4k to 88k kept 0-1
pieces with 29-31 erased; block mass rose from 0.51 (16k) to 0.70 (88k) and error at +128 fell from 9.0%
to 6.7%. Standard at 36-56k: 9-16 intact, 3-9 erased, mass 0.91-0.95, about 3.7%.

Committed check (`commit_check.py`, CPU): dreaming the same games with each frame committed token by
token to its most likely colours instead of fed back soft.

| step | soft: intact / erased / mass / wrong +128 | committed: intact / erased / mass / wrong +128 |
|---|---|---|
| 8k | 0 / 29 / 0.55 / 8.8% | 2 / 29 / 0.73 / 3.8% |
| 16k | 0 / 31 / 0.51 / 9.0% | 2 / 29 / 0.53 / 6.7% |
| 88k | 1 / 30 / 0.70 / 6.7% | 0 / 29 / 0.76 / 3.3% |

Committed, the board is as good as standard's (3.3% at +128), but the piece is gone either way: the model
never learned it, so soft feedback was not hiding it, and stage 2 (which can sharpen what the model knows,
not teach it) was not started from this base. Moving pixels one frame ahead stayed about 50% expected
wrong at every step: the first generated frame never became sure.

Suspect (not tested when written): the swaps. Half of them copy a token from another place in the
frame, in Tetris a block where there is none, whose cheapest repair is erasing it; a falling piece is
exactly a floating block. The original model also had swaps and kept pieces, but its first 24.6k steps
had own guesses and no swaps.

**The test, running since 2026-10-04 07:48:** `stage1_noswap`, stage 1's step-88000 weights
(`--grow-from`) with every swap off (`--no-swaps`) and everything else the same, for up to 20k steps.
Movement by ~10k (intact above 1-2, erased below ~25) points at the swaps; none points elsewhere (own
guesses, soft rollouts, the heavy set).

## The blend alone: stage1_blend (from 2026-10-04)

Every failing run trained on rollouts far deeper than 32 frames scored against the real frames after
them; both runs that kept pieces (the original, standard) had none that deep. A deep rollout drifts out
of step with the game (the piece a few rows off, or landed early), and the real frame after it then
contradicts the context: the cheapest answer is to hedge, and a hedged piece is an erased one (Bengio et
al. 2015: long stretches of the model's own predictions train badly; Huszar 2015: the loss pulls toward
the average future). HorizonDrive's fix keeps the long rollouts but fades the last w frames before the
scored one from the model's own to the real ones (`blend_to_real`), w growing 0 -> 8 over 8k of 10k
steps (its SRR stage, `examples/text/horizondrive_2605.11596.txt` section 4.2: every training window
there is a rollout; the 20-beats-4 result is its later distillation stage).

The test changes only that: stage 1's recipe (swaps, noise, four sets, own guesses, tossing, the border,
uniform 1-62 rollouts) from stage 1's step 88,000, plus `--blend 8 8000`, for about 10k steps. It
replaces the swap-free run (stopped at 16k: 0-2 intact / 29-30 erased at every test, 2k-14k;
checkpoints archived).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -TrainerArgs "--models stage1_blend
  --grow-from D:\token_world_checkpoints\world_model_tetris_stage1\model_step88000.pt --blend 8 8000
  --warmup 500"
```

If pieces come back, the mismatch was the cause and long rollouts are fine with the blend; stage 2 then
adds the gradient. If not, the depth itself is: next, rollouts capped at 32 and mostly short, as the
original's were.

**Result so far:** pieces came back, the first run to move them since stage 1 began. Same 32 games
(intact / partial / erased at +32; pixels expected wrong at +128):

| step | blend w | 1 pass | 2 passes | wrong +128 (1 pass) |
|---|---|---|---|---|
| 2k | 2 | 0 / 4 / 28 | 1 / 2 / 29 | 6.7% |
| 4k | 4 | 1 / 4 / 27 | 2 / 5 / 25 | 6.4% |
| 6k | 6 | 3 / 9 / 20 | 3 / 14 / 15 | 4.4% |
| 8k | 8 | 8 / 6 / 16 | 9 / 5 / 16 | 3.6% |
| 10k | 8 | 8 / 9 / 12 | 10 / 11 / 9 | 3.4% |
| 12k | 8 | 11 / 6 / 14 | 11 / 10 / 10 | 3.4% |
| 14k | 8 | 10 / 10 / 11 | 15 / 8 / 7 | 3.3% |
| 16k | 8 | 10 / 11 / 7 | 15 / 8 / 7 | 3.3% |
| 18k | 8 | 17 / 8 / 3 | 15 / 10 / 4 | 3.5% |
| 20k | 8 | 14 / 10 / 4 | 16 / 8 / 5 | 3.4% |
| 22k | 8 | 12 / 10 / 6 | 13 / 8 / 8 | 3.4% |
| 24k | 8 | 19 / 5 / 3 | 17 / 10 / 2 | 3.5% |
| 26k | 8 | 20 / 5 / 3 | 19 / 7 / 4 | 3.5% |

Block mass 0.65 -> 0.88. By 18k it matches standard's best on intact pieces and erases fewer (3-5); by 14k it was near standard's best (17 / 8, about 3.5% at +128), and two passes
now beat one. The previews' changed pixels stayed about 0.6 expected wrong: it keeps pieces (copying is
right most frames: the board changes about one frame in seven) before it moves them. Left training long,
as the original needed 52k-90k for 28 / 0.

## The simplest recipe: HorizonDrive's two stages (built 2026-10-04)

Reading HorizonDrive itself (`examples/papers/horizondrive_2605.11596.pdf`; its code in `examples/repos` is
inference only) showed it uses no sets: a base model trained on clean real context for 50k steps, then
scheduled rollout recovery (SRR) for 10k, where every training window is a rollout. Its blend straddles
the boundary (the first targets are part the model's own prediction too), and its rollout depth falls,
deepest first. It is Wan 2.1 1.3B on 64-96 RTX 5090s, 11 frames of history; the recipe is what carries
over. The user chose the simplest version of it: clean context, blended targets, depth falling.

| | stage A: `base` | stage B: `srr` |
|---|---|---|
| windows | all teacher-forced, real and clean | all rollouts from a real start |
| rollouts | none | 1..depth frames, depth 62 -> 16 over 8k steps |
| blend | none | radius 0 -> 8 over 8k steps, input and target |
| removed | swaps, noise, the four sets, tags, own guesses (one decoding pass), Self Forcing, tossing | same |

```powershell
# stage A
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -TrainerArgs "--models base"
# stage B, from a base checkpoint
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -TrainerArgs "--models srr
  --grow-from D:\token_world_checkpoints\world_model_tetris_base\model_step<N>.pt
  --rollouts 62 16 8000 --blend 8 8000 --warmup 500"
```

The code of the recipes above (sets, swaps, noise, tags, own guesses, Self Forcing, tossing) was removed;
it is in the repository's history. Their checkpoints no longer load in the current model (no tag or guess
layers).

## Stage A's result (base, 2026-10-04/05, stopped at 108.5k)

Plain teacher forcing beat every earlier soft recipe. Same 32 games, with the position scores
(`diagnostics/long_dream.py`, added 2026-10-04 after the piece count was seen to reward a piece frozen
at the spawn point):

| run | hit at +32 | ghost | fall | spawn | mass | wrong at +128 | timer wrong at +128 |
|---|---|---|---|---|---|---|---|
| blend run 26k (re-scored) | 0.23 | 0.80 | 0.31 | 0.16 | 0.88 | 3.5% | 36% |
| base 98k | 0.70 | 0.33 | 1.01 | 0.79 | 1.00 | 3.2% | 6% |
| base, mean of 96-106k | 0.57 | 0.43 | 0.96 | 0.81 | 0.99 | 3.6% | 8% |

It learned in order: keep the piece (frozen, ~10-18k), let it fall (18-36k), the fall timer and the
next piece's spawn (34-48k), the real falling speed (~90k). Placement (hit ~0.55) stopped improving from
~48k while one-frame prediction kept improving: the gap a model trained only on real context has in its
own dreams. On the old count score it matched the original (28.8 / 0.2 against 28 / 0).

Where its dream goes wrong (`history_errors.py`, CPU, the 32 games): its own frames drift in the
playfield within 2-4 frames (a piece a row or a frame out of step; changed pixels 11-27% wrong) and then
stay as an offset; the border clock stays right for ~46 frames; the next-piece box is hedged from the
first frame (the random next piece, never committed to: the likely source of fuzzy spawns). Put the real
playfield or the real last 4 frames back and the next frame is as good as from a real history; the
border and HUD make no difference.

## Stage B attempts (2026-10-05)

| run | rollouts drawn by | lr, weight decay | blend | result |
|---|---|---|---|---|
| `srr_first` | the model being trained | 3e-4, 0.05 | two-sided, radius 0 -> 8 | clock lost within 400 steps (border 0% -> 2.4% wrong); 8k: 19 erased, hit 0.09, fall ~0, timer 28% |
| `srr_wide` | a frozen copy, refreshed every 2,000 steps | 3e-5, 1e-5 | two-sided, radius 24 from step 1 | clock kept; 2k / 4k / 6k: hit 0.72 / 0.64 / 0.56, ghost 0.33 / 0.42 / 0.61, spawn 0.21 / 0.21 / 0.00, wrong +128 2.6% / 2.9% / 3.3% |

The second run's targets after the boundary were partly the copy's own prediction, over the 24 frames
where most next pieces spawn: the model learned to imitate the copy's spawns, and each refresh passed the
errors on. The next run blends the history only (radius drawn 0-8 each step, so the frame before the
first target runs from nearly real to fully its own), with real targets throughout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -TrainerArgs "--models srr
  --grow-from D:\token_world_checkpoints\world_model_tetris_base\model_step98000.pt --rollouts 62 16 8000
  --blend 8 --refresh 2000 --lr 3e-5 --weight-decay 1e-5 --warmup 500"
```

## Stage B's result (`srr`, history-only blend, 2026-10-05, stopped at 28.8k)

The 32 trials every 2,000 steps (hit: the share of trials, then how many of 32 at 0.75 or more):

| step | hit (>= .75) | ghost | fall | spawn | wrong +128 | timer +128 |
|---|---|---|---|---|---|---|
| base 98k | 0.70 (21) | | 1.01 | 0.79 | 3.2% | 6% |
| 2k / 4k / 6k | 0.55 / 0.39 / 0.66 | 0.44 / 0.58 / 0.42 | 0.96 / 0.92 / 0.97 | 0.95 / 0.84 / 0.84 | 3.7% / 3.8% / 3.0% | 8.4% / 4.2% / 4.8% |
| 8k-14k | 0.54 -> 0.68 | 0.48 -> 0.34 | 0.93-0.99 | 0.74-0.89 | 3.5% -> 2.8% | 5-7% |
| 16k | 0.73 (24) | 0.24 | 0.92 | 0.89 | 2.6% | 3.1% |
| 18k | 0.64 (18) | 0.38 | 0.89 | 0.79 | 3.1% | 5.7% |
| **20k** | **0.73 (23)** | **0.25** | 0.95 | 0.89 | 2.6% | **1.7%** |
| 22k | 0.70 (21) | 0.27 | 0.95 | 0.95 | 2.5% | 2.1% |
| 24k / 26k / 28k | 0.43 / 0.57 / 0.49 | 0.56 / 0.40 / 0.49 | 0.91-0.96 | 0.95 | 3.9% / 3.1% / 3.8% | 5.3% / 5.2% / 6.3% |

The first stage B that improved on its base instead of breaking spawns or the clock. It peaked at
16-22k (the depth reached its floor of 16 at 8k) and drifted after, while its training rollouts kept
improving (ghost pixels 90 -> 43): three weak tests in a row, the clock worst. Step 20,000 is the
result and the browser's model.

Generic coherence (`diagnostics/coherence.py`, the 32 trials, a bank of 160k live frames): changed
tokens whose exact patch never occurs in real play, and token changes never seen.

| | unseen patch | unseen change | change px | unsure px |
|---|---|---|---|---|
| real future (floor) | 0.5% | 0.1% | 0.35% | 0% |
| `srr` 20k | 8.8% | 14.7% | 0.35% | 0.07% |
| `srr` 16k | 9.5% | 16.7% | 0.35% | 0.07% |
| base 98k | 12.3% | 19.8% | 0.35% | 0.09% |
| `srr_wide` 6k (spawns broke) | 16.7% | 33.0% | 0.29% | 0.35% |
| `srr_first` 8k (clock broke) | 44.7% | 94.6% | 0.05% | 1.2% |

It ranks the broken runs worst with no Tetris knowledge; training reports it with every long dream.

**Game end -> restart** (8 real top-outs: 48 frames, then 320 with the bot's real buttons through the
curtain, the level menu and Start; whole screen within 10% of the real one). In training data a game
ends about every 5,200 frames (87 s): a restart is 4-5% of frames, but each Start-made screen switch
is one frame in 5,200.

| | on the menu (+16) | back in a game (+8) | back in a game (+63) |
|---|---|---|---|
| base 98k | 0/8 | 1/8 | 1/8 |
| `srr` 20k | 6/8 | 0/8 | 4/8 |

The base sticks in the curtain or on the menu; the rollouts got through the curtain to the menu.
Menu -> game is still late or never (the menu morphs into the game: a hedge between staying and
switching), which the random latent (below) is meant to fix.

**Weighting changes, the free check.** Decoding with the odds of every pixel change times w
(`weigh_changes`, the optimum a loss weighting changed pixels w times trains to), on `srr` 20k:

| w | hit (>= .75) | ghost | wrong +128 | timer +128 | unseen patch / change | restart menu, back +63 |
|---|---|---|---|---|---|---|
| 1 | 0.72 (22) | 0.26 | 2.6% | 2.9% | 8.8% / 14.7% | 6/8, 4/8 |
| 2 | 0.59 (14) | 0.41 | 3.5% | 4.3% | 8.9% / 21.7% | 6/8, 4/8 |
| 3 | 0.33 (5) | 0.56 | 3.9% | 4.2% | 10.6% / 25.4% | 6/8, 5/8 |

Over-filled pixels fed back as history become ghosts. This rules out the shifted outputs, not the loss
weighting's gradient: a weighted run decoded at 1/w (which undoes the shift exactly) would test that,
against an unweighted run at full length. Not run.

## The random latent (`--models latent`, from `srr` 20k, 2026-10-05)

A sampled choice per frame (4 categoricals of 8, DreamerV3: posterior from the real frame and the one
before, prior from the last frame's features, KL balancing 0.5 / 0.1, 1% uniform), added to the frame's
tokens through a zero-initialised layer so it starts as the model it grows from; teacher forcing, lr
1e-4. A dream samples the choice from the prior, so a random event is decided rather than averaged.

**First run, `latent_freenats` (1 free nat on the posterior's side, stopped at 6.1k):**

| | hit (>= .75) | ghost | spawn | wrong +128 | timer +16 / +128 | restart menu, back +63 |
|---|---|---|---|---|---|---|
| `srr` 20k | 0.73 (23) | 0.25 | 0.89 | 2.6% | 0.06% / 1.7% | 6/8, 4/8 |
| 2k | 0.45 | 0.49 | 0.84 | 3.6% | 0.07% / 6.3% | 7/8, 0/8 |
| 4k, sampled choice | 0.51 (9) | 0.47 | 0.74 | 3.5% | / 6.3% | 4/8, 2/8 |
| 4k, most likely choice | 0.77 (24) | 0.17 | 0.74 | 2.5% | / 5.2% | 0/8, 1/8 |
| 6k | 0.41 | 0.67 | 0.53 | 3.8% | 8.9% / 16.0% | |

The KL rose from 0.5 to 1.7 nats a frame. Tetris holds well under 0.1 nats of randomness a frame, so
the free nat let the choice carry deterministic things for nothing: with the most likely choice pieces
fell better than in any model (teacher forcing had not undone stage B), but a dream never reached the
menu and spawned late, and by 6k the border's timer came from the choice (8.9% wrong at +16, every
earlier model near 0%). Sampled, the choice was mostly the prior guessing at them: smeared pieces.

**Second run, `latent_nofree` (every nat paid for, stopped at 2.2k):** the KL started lower (0.2 nats at
400 steps) and climbed again (0.33, 0.62, 0.87, 1.05 by 2k); 2k: hit 0.49, ghost 0.43, spawn 0.95,
timer +128 7.3%. The prior was the flaw: it read the last frame's features averaged over all tokens,
without the buttons going into the frame, so Start (the restart's screen switches), moves and the
timer's few border tokens all looked random to it, and the choice carried them, at a price worth paying.

**Third run, `latent_prior` (stopped at 4.1k):** the prior reads the incoming buttons (the frame's action
embedding) and pools the last frame's tokens by attention (a learned query per group). The KL climbed
as before (0.43, 1.03, 1.34, 1.55 at 800-3,200 steps, 1.18 at 4k). 2k: hit 0.47 (10), ghost 0.49, spawn
1.00, timer +128 7.7%; 4k: hit 0.58, ghost 0.40, spawn 0.74, timer +128 7.9%, unseen 8.4% / 17.9%.

What the choice carries (4k checkpoint, 48 real 64-frame windows, each frame's KL between posterior and
prior lined up with RAM; `choice_content.py`, `choice_codes.py` in the session scratchpad):

| that frame | frames | mean KL (nats) |
|---|---|---|
| nothing happens (only the border's counters change) | 679 | 3.14 |
| the buttons changed | 584 | 2.36 |
| move or rotation | 275 | 1.45 |
| spawn (next piece drawn) | 60 | 1.20 |
| gravity drop | 819 | 1.19 |
| lock, clear, curtain, menu | 1,217 | 0.85 |

The posterior used 16 of its 4,096 codes. Given the code: fall timer ($45) 21% explained, play state
($48) 19%, autorepeat 22%, the next piece 2%, the piece's row 4%. The choice is a coarse phase code
(where in the timer's cycle, which play state), not the random event. Our posterior reads only the real
frames, so it carries whatever is generally useful; DreamerV3's reads the model's state as well (q(z |
h, x)), so it carries only the surprise against what the model already expects. That is the next thing
to try.

**Control, `control_tf`:** stage B 20k fine-tuned with teacher forcing alone, no latent, lr 1e-4, 4k
steps (`--output-root output\control_tf --archive D:\token_world_checkpoints\control_tf`): does
teacher forcing alone take the timer from 1.7% back to the base's 6%? Yes:

| 2k | hit | ghost | spawn | wrong +128 | timer +128 | unseen patch / change |
|---|---|---|---|---|---|---|
| `srr` 20k (the start) | 0.73 | 0.25 | 0.89 | 2.6% | 1.7% | 8.8% / 14.7% |
| `control_tf` (no latent) | 0.52 | 0.42 | 1.00 | 3.5% | 6.4% | 9.9% / 18.9% |
| `latent_nofree` | 0.49 | 0.43 | 0.95 | 3.5% | 7.3% | 10.4% / 18.0% |
| `latent_prior` | 0.47 | 0.49 | 1.00 | 3.7% | 7.7% | 10.1% / 18.5% |

A teacher-forcing fine-tune undoes most of stage B within 2k steps on its own; the second and third
latent runs are within noise of it, so their timer drift was the fine-tune's, not the latent's (the
first run's 6k, timer 8.9% wrong at +16, was the latent's). Latent runs have to be judged against this
control, and trained with stage B's rollouts: the rollouts keep the choice they sampled for each frame,
and the history they dreamed is trained with it. At 4k the control had hit 0.44, ghost 0.56, spawn 0.89, timer +128 4.8%
(`latent_prior` at 4k: 0.58, 0.40, 0.74, 7.9%).

**Fourth run, `latent` (2026-10-05 21:00):** the latent with stage B's rollouts and stage B's settings
exactly (`--rollouts 16 16 1 --blend 8 --refresh 2000 --lr 3e-5 --weight-decay 1e-5`, from `srr` 20k):
each rollout frame keeps the choice the frozen copy drew for it (the compiled CUDA graph draws fresh
ones each replay), and those frames are left out of the KL. Its control is stage B's own run past 20k
at the same settings (22k-28k: hit 0.70, 0.43, 0.57, 0.49).

| `latent` | hit (>= .75) | ghost | spawn | wrong +128 | timer +128 | unseen patch / change | KL | restart menu +16, back +8, back +63 |
|---|---|---|---|---|---|---|---|---|
| 2k | 0.71 (23) | 0.25 | 0.84 | 2.6% | 4.2% | 8.0% / 16.6% | 0.19-0.35 | 0/8, 1/8, 7/8 |
| 4k | 0.60 (14) | 0.39 | 0.79 | 3.4% | 4.1% | 8.7% / 17.6% | 0.41-0.66 | |
| 6k | 0.72 (17) | 0.27 | 0.68 | 2.6% | 7.9% | 8.5% / 18.0% | 0.74-0.82 | |
| 8k | **0.80 (24)** | **0.22** | 0.95 | **2.4%** | 3.9% | 8.1% / 15.4% | 1.11-1.14 | |

Stage B's quality held (teacher forcing lost it within 2k). The choice is barely used yet at lr 3e-5
(19 codes; fall timer 7% explained, next piece 2%), and most likely and sampled choices score the same
(hit 0.70 (23) both; timer 3.8% / 4.7%). Game ends: the dream runs 20-30 frames behind the real one at
each step (curtain, menu, new game) but reaches a new game with an empty board in 7 of 8 (stage B 20k:
4 of 8), some digits garbled. At 4k the code drifts to the same phase code as before (fall timer 14%,
play state 11%, next piece 2%) while the KL rises; stage B's own run was lower at the same distance
(24k: hit 0.43). The posterior averages the whole frame, so the timer's phase, which changes every
frame, outweighs the next-piece box, which matters one frame in about 40; a posterior that also reads
the model's expectation (DreamerV3's q(z | h, x), a second forward pass here) would see the timer as
expected and the next piece as the surprise. Running to 8k to set it against stage B's 22-28k.

At 8k `latent` is the best model so far (hit 0.80, ghost 0.22, wrong +128 2.4%), where stage B's own run
had fallen to hit 0.49 at the same distance (28k). The choice still tracks the timer's phase (21%) and
not the next piece (2%), so the gain is not the latent deciding the next piece: more likely the rollouts
continuing with fresh layers and a stable learning rate. It keeps training.

**Theory of the current model (`latent`, run 4).** Stage B's rollouts, unchanged (a frozen copy
refreshed every 2,000 steps dreams 16 frames from real history; the last 0-8 history frames fade to
real; targets are always real), with a sampled choice per frame added to the frame's tokens through a
zero-initialised layer. Training reads each real frame's choice from the posterior (the real frame and
the one before, pooled) and the dreamed history keeps the choices the copy drew; the prior (the last
frame's tokens, attention-pooled per group, plus the incoming buttons) learns to predict the posterior,
every nat paid for (KL balancing 0.5 / 0.1, 1% uniform). A dream samples the choice from the prior.
What it was meant to fix: averaging over random outcomes (84% of the incoherence is the next-piece box,
below). What it does instead so far: a coarse phase code. Why the posterior misses the piece: it averages
the whole frame, so the timer (changing every frame) outweighs the next-piece box (one frame in about 40);
DreamerV3's posterior also reads the model's state, so it sees only the surprise.

The goal for a full run is one recipe, not stages: the latent from step 0, rollouts ramping in on one
learning-rate schedule.

## Stage 2 (built 2026-10-03, never run; superseded by the two-stage recipe above)

A new run (`stage2`) from a stage-1 checkpoint, when stage 1 has learned the dynamics from real context:
its changed-pixel loss flattens (1.4 at 17k, still falling) and static pixels expected wrong stop
improving (about 1%). The command, after stopping stage 1 (both don't fit on the GPU):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_until_stopped.ps1 -TrainerArgs "--models stage2
  --grow-from D:\token_world_checkpoints\world_model_tetris_stage1\model_step<N>.pt
  --self-forcing 4 --blend 8 4000 --keep-all-windows --lr 1e-4 --warmup 500"
```

| part | stage 2 |
|---|---|
| start | stage 1's averaged weights at step N (`--grow-from`), a fresh optimizer |
| data | tossing off (`--keep-all-windows`): the game's own event rates |
| rollout set | Self Forcing (`self_forced_rollout`): a soft rollout of 1-62 frames, its last 4 frames with their graph, so the loss on the frames after the rollout reaches how they were drawn; the last w frames before the scored ones fade from the model's own to the real ones (`blend_to_real`), w growing from 0 to 8 over the first 4,000 steps (HorizonDrive) |
| other sets | noise, light, heavy as in stage 1, with own guesses |
| learning rate | 1e-4 after a 500-step warmup (stage 1: 3e-4) |

**Why it could work where stage 1 alone cannot:** in stage 1 the rollout frames are made without
gradient, so drawing them blurry costs nothing. In stage 2 the soft frames are differentiable: a frame
drawn blurrier than it needs to be makes the frames after it harder to predict, and the gradient says so.
An argmax model could not get this gradient. The blend keeps it from teaching erasure: the earlier Self
Forcing scored drifted frames against real ones they contradicted (29/32 erased), this scores the frames
after the rollout, with the context nearest them fading to real.

**What it cannot do:** teach dynamics stage 1 never learned (where a piece goes, when it drops).

**Judge it** on the same 32 games (pieces and block mass, every 2,000 steps), against the stage-1
checkpoint it started from, and on the previews (the GAN's damage hid from the piece score).

Checked on the CPU from the step-16000 checkpoint on live windows (2026-10-03): three steps at w = 0, 4,
8 ran with finite losses, about 35 s a step on 4 CPU threads; tests in `tests/test_dynamics.py`
(`SelfForcingTests`).

A possible stage 3: the border off.

## Where the incoherence is (2026-10-05, `srr` 20k, the 32 trials)

`incoherence_breakdown.py` (session scratchpad): every unseen patch in the dreams, by region of the
16 x 16 token grid and by whether the real game spawned within 8 frames. 3,088 unseen patches in 34,893
changed tokens:

| region | when | share of unseen patches | unseen in that region |
|---|---|---|---|
| next-piece box | between spawns | 57% | 30% |
| next-piece box | near a spawn | 27% | 68% |
| playfield, spawn rows | near a spawn | 9% | 24% |
| playfield, spawn rows | between spawns | 2% | 3% |
| playfield below (falling piece, stack) | any | 4% | 1-3% |
| statistics and border counters | any | ~0% | ~0% |

84% is the next-piece box: the one random event of a no-button dream. The model draws a blend of
pieces there at a spawn, soft decoding feeds the blend back, and it stays blended until the next spawn;
the 9% is the blend arriving at the spawn rows. Falling and the stack are nearly clean. The references
all commit at inference (Genie 25 MaskGIT steps at temperature 2, DIAMOND 3 steps, GameNGen 4 DDIM),
and DIAMOND names it: a single-step prediction is the expectation over the possible outcomes, blurred
when they are several. Their tokens are codes, so committing gives a real patch; our head is per pixel,
which is why argmax erased and sampling speckled. Next: commit a token to a real patch at decode time
(the coherence bank's patches at that position, sampled by the model's pixel likelihood).

## Layered NES tokens: SMB (2026-10-05)

World NES now captures the PPU's layers (llm-thing23, regions `ppu_background`, `ppu_sprites`,
`ppu_sprite_index`, `ppu_scroll`; built, not installed while training holds the module).
`smb_layers.py` (session scratchpad): the SMB bot, 20,000 frames (59% in levels):

| | 16 x 16 screen patches (today) | background on the world canvas | sprites as OAM entries |
|---|---|---|---|
| tokens changed per frame in play | 17.7% | 0.74% | ~15 on screen |
| vocabulary after 20k frames | 28,391, climbing | 1,850 | 288 (tile, palette, flip) |

The layers compose back to every frame exactly (20,000 / 20,000). Scrolling makes most of the screen
change and the patch vocabulary explode; in world space the background is nearly as static as Tetris,
and the moving things are a short list of sprites. The PPU's own codes (nametable tiles, OAM tiles) are
a free, exact vocabulary for any NES game: committing a code is always a whole tile. Rendering them needs
the pattern tables (a CHR region: SMB's are ROM, not `chr_ram`).


The design built on this, for review: `docs/guides/layered_tokens.md`. OAM slots are stable enough to be
token identities (SMB 94% continue in their slot, 1% move; Tetris 98%, 0%), and SMB's only scroll split
is at line 32, a token-row boundary.

## The layered base run (`world_model_tetris_layered`, launched 2026-10-06 01:40)

`scripts/train_layered.py --workers 12`, teacher forcing with MaskGIT masking, no rollouts. Long dreams
sample at temperature 1, sprites decided in one pass:

| step | hit | ghost | fall | spawn | wrong +16 / +128 | timer +16 / +128 | unseen patch / change |
|---|---|---|---|---|---|---|---|
| 2k | 0.02 | 0.21 | 1.33 | 0 | 5.19% / 7.56% | 23.7% / 28.1% | 84% / 99% |
| 4k | 0.02 | 0.32 | 2.37 | 0 | 2.97% / 5.14% | 22% / 23% | 81% / 95% |
| 6k | 0.17 | 0.72 | 0.61 | 0 | 2.90% / 4.85% | 30% / 40% | 68% / 93% |
| 8k | 0.18 | 0.75 | 0.53 | 0 | 2.75% / 4.89% | 33% / 41% | 62% / 95% |
| 10k | 0.16 | 0.81 | 0.23 | 0 | 2.76% / 4.43% | 35.6% / 42.0% | 67% / 97% |
| 12k | 0.15 | 0.87 | 0.19 | 0 | 2.60% / 4.33% | 36.8% / 38.7% | 69% / 98% |
| 14k | 0.14 | 0.89 | 0.69 | 0 | 2.74% / 4.32% | 33.3% / 32.4% | 63% / 91% |
| 16k | 0.18 | 0.82 | 0.49 | 0 | 2.45% / 4.20% | 34.7% / 34.9% | 65% / 92% |
| 18k | 0.20 | 0.88 | 0.15 | 0 | 2.29% / 4.37% | 29.5% / 29.7% | 74% / 97% |
| 20k | 0.23 | 0.86 | 0.15 | 0 | 2.23% / 4.30% | 31.7% / 35.3% | 73% / 95% |
| 22k | 0.20 | 0.85 | 0.13 | 0 | 2.27% / 4.28% | 26.8% / 31.1% | 74% / 98% |
| 24k | 0.21 | 0.84 | 0.13 | 0 | 2.23% / 4.19% | 27.4% / 34.3% | 75% / 98% |
| 26k | 0.20 | 0.93 | 0.26 | 0 | 2.21% / 3.95% | 29.8% / 22.9% | 74% / 95% |
| 28k | 0.21 | 0.93 | 1.51 | 0 | 2.18% / 3.60% | 28.2% / 22.7% | 72% / 85% |
| 30k | 0.19 | 0.98 | 0.95 | 0 | 2.09% / 3.72% | 29.6% / 22.8% | 73% / 86% |

**Why the dreamed piece breaks (10.2k; `layered_sprite_check.py`, session scratchpad).** Hiding one
real frame at a time (its history real), the model gets the sprites' movement right: 688 / 693 still
sprites, 56 / 70 moved ones exactly, 4 / 5 placements. In a dream, each of the piece's 4 sprites samples
its movement on its own, so one cell steps left at +0, another drops at +9, a third at +12: the piece
comes apart, a broken piece is unlike anything in training, and the model stops moving it.

**Decoding the sprites in passes** (`layered_decode_check.py`, 8 trials, 10.2k):

| sprites decided in | temperature | hit | ghost | fall | wrong128 | timer128 | unseen patch / change |
|---|---|---|---|---|---|---|---|
| 1 pass | 1 | 0.12 | 0.95 | 0.23 | 4.60% | 40.7% | 61% / 96% |
| 8 passes (one sprite each, surest first) | 1 | **0.29** | **0.61** | 0.07 | 4.53% | 30.8% | 63% / 99.1% |
| 1 pass | 0 | 0.21 | 0.76 | 0.00 | 4.44% | 42.8% | 70% / 99.6% |
| 8 passes | 0 | 0.21 | 0.71 | 0.00 | 4.54% | 31.2% | 65% / 99.7% |

At 16k on the same 8 trials it reversed (1 pass: hit 0.31, ghost 0.68; 8 passes: hit 0.11, ghost 0.94):
8 trials sampled at temperature 1 are too few to rank decodings. On 32 trials at 16k:

| sprites decided in | hit | ghost | fall | wrong128 | timer128 | unseen patch / change |
|---|---|---|---|---|---|---|
| 1 pass | 0.19 | 0.98 | 0.33 | 4.22% | 36.5% | 62.5% / 90.1% |
| 8 passes | 0.13 | 0.95 | 0.96 | 4.41% | 33.6% | 62.5% / 94.1% |

Deciding sprites one at a time does not help the scores (and is 2.5x slower): the 10k gain was noise.

With 8 passes the piece holds together (traced on one trial: the 4 cells keep their shape for 64 frames), as Genie's
and MaskGIT's many-step decoding is meant to. But it never falls.

**Gravity timing is not learned at 10.2k** (`layered_drop_prob.py`): along a still level-0 dream, the
probability that the piece drops 8 px is flat at 1-3% per frame for 72 frames, with no peak where gravity
comes (48 frames at level 0: 1/48 = 2.1%). The model has the rate of falling, not its timing (memory:
gravity-timing-hardest). The tick border's timer, the clock it would count with, is 31-43% wrong after
128 dreamed frames and 12.8% wrong teacher-forced at 8k (the old base at 20k: 14% / 29%).

At 16k the drop probability is still flat (1-5% a frame over 72 frames of a level-0 dream, no peak
near +44 where gravity comes).

**At 20k** (the point we judge a run at): the dream scores have been flat since 8k (hit 0.15-0.23,
wrong +128 4.2-4.9%) and below the old base at the same step (hit 0.35, timer +16 14%). Training is
still learning the sprites: the movement loss averaged 0.157 over 8-10.4k and 0.087 over 20-22.4k;
the cells' loss is flat (0.012-0.013). Left running.

**Why the piece never falls in a dream: the clock stops (30k).** Teacher-forced, the model times gravity
well: at the 147 real gravity drops in the 32 trials' last 32 frames (no button pressed) it gives the
drop a median 0.95 (16k) and 0.97 (30k), against 0.015-0.028 on the 3,100 still frames
(`layered_drop_tf.py`). In a dream it never drops the piece (30k: P(drop) 0.000 for 72 frames of a still
level-0 dream). The border holds the frame counter and Tetris's fall timer (RAM 0x45) as base-3 digits
(`model_frames.py`), and the dream does not advance them (`layered_clock_trace.py`, trial 0):

    real:  691/22 692/23 693/24 694/25 695/26 696/27 697/28 698/0 699/1 700/2 701/3 702/4
    dream: 703/2 709/2 708/2 708/2 702/2 702/2 ... 702/2 705/2 ... 705/2 705/3 705/0 ...

(frame counter / fall timer). The fall timer stays at 2, so gravity never comes. Even with a real
history, the model writes the next frame's numbers right only a third of the time; mostly it copies the
last frame's digits, the lowest one included (it changes every frame) (`layered_clock_tf.py`, 8 trials'
last 16 frames):

| step | next frame counter right | fall timer right |
|---|---|---|
| 10k | 18 / 128 | 23 / 128 |
| 20k | 29 / 128 | 54 / 128 |
| 30k | 43 / 128 | 54 / 128 |

The training windows count by exactly one a frame (`stream_clock.py`): the data is right; adding one in
base 3 is being learned slowly.

**Against the old base at the same steps** (both stage A: masked teacher forcing, no rollouts; long dreams
on 32 trials; the old base logged hit, ghost and timer only from 20k):

| step | wrong +16 base / layered | wrong +128 base / layered | hit base / layered | ghost base / layered | timer +128 base / layered |
|---|---|---|---|---|---|
| 4k | 1.8% / 3.0% | 3.3% / 5.1% | | | |
| 10k | 2.6% / 2.8% | 4.1% / 4.4% | | | |
| 16k | 2.2% / 2.5% | 4.3% / 4.2% | | | |
| 20k | 1.6% / 2.2% | 3.7% / 4.3% | 0.35 / 0.23 | 0.39 / 0.86 | 29.2% / 35.3% |
| 24k | 1.2% / 2.2% | 3.1% / 4.2% | 0.53 / 0.21 | 0.32 / 0.84 | 23.9% / 34.3% |
| 28k | 1.2% / 2.2% | 3.2% / 3.6% | 0.44 / 0.21 | 0.38 / 0.93 | 22.6% / 22.7% |
| 32k | 1.1% / 2.1% | 3.1% / 4.2% | 0.43 / 0.24 | 0.44 / 0.92 | 22.0% / 26.2% |

The layered model is behind from 4k on, and the gap opens after 18k: the old base kept improving (wrong
+16 2.0% to 1.1%, hit 0.35 to ~0.45), the layered one stopped (2.2-2.1%, hit ~0.2). The clock is as wrong
in the old base (timer +128 22-29%), so the clock is not what separates them. The 32k sheet shows what
does: the piece falls a few rows, its four cells drift apart, and they freeze in mid-air (ghost 0.9).
Four sprites make four separate movement decisions; the pixel model's piece is one blob of pixels that
moves together.

## Layered pixels: a new stage A (`world_model_tetris_layered_px`, launched 2026-10-06 10:28)

**Why the slot model was dropped.** Same-step comparison with the pixel stage A (long dreams, the same 32
games, and 24 live windows with real buttons dreamed 16 frames, both committed): it was behind at every step
from 4k to 42k, and its own scores were flat from 20k (hit ~0.2, ghost ~0.95). Its pieces were four sprites
each making its own hard movement choice. No reference model uses object slots; all use patch tokens.

**What replaced it** (option A of the analysis): the old stage A plus layers, soft kept. The background
stays fixed to the world (the scroll fix); the sprites become a screen-space picture of the sprite pixels
that show (exact: the frame is that picture over the background), in 240 more 16 x 16 pixel tokens with
their own pixel head; dreams feed both back soft, as the pixel model does. 543 tokens a frame, 3.03M
parameters.

**The clock defect, found and fixed.** Teacher-forced (each frame hidden in turn, history real; the border's
frame counter and fall timer decoded from the predicted pixels):

| model | step | next frame counter right | fall timer right |
|---|---|---|---|
| pixel stage A | 2k / 4k / 10k / 30k | 5 / 36 / 127 / 128 of 128 | 24 / 33 / 112 / 123 |
| slot model | 10k / 20k / 30k | 18 / 29 / 43 | 23 / 54 / 54 |
| layered pixels, rotary time | **4k** | **125** | **110** |

The data was identical in both streams (decoded counters, fall timer and RAM agree), and the compiled
training forward matched the eager one. Even in the pixel model, the most recency-focused head put at
most 0.27 of a border token's attention on the frame before. Both models' only sense of time was one
learned embedding per window slot, added at the input (as 1X's Genie). The new model uses rotary time
(RoPE) inside the temporal attention, as Open-Oasis does: a query and a key score by how many frames apart
they are, in every layer. At 4k it counts as well as the pixel model did at 10k.

**Memory.** Activation checkpointing per block (`recompute`, as the pixel trainer's option) and a
12 GB cap on the allocator (`--vram-gb`: without one it reserved 19 GB for 8.6 GB in use): 12.6-12.9 GB on
the card, 0.85 steps/s.

**Long dreams against the pixel stage A** (the window's Compare tab):

| step | wrong +16 | wrong +128 | mass | hit | unseen change |
|---|---|---|---|---|---|
| 2k | 6.1% / 4.5% | 8.2% / 6.7% | | 0.00 | 100% |
| 4k | 1.9% / 1.8% | 3.3% / 3.3% | 0.79 | 0.02 | 48% |
| 6k | 1.5% / 1.8% | 2.9% / 3.3% | 0.79 | 0.02 | 15% |
| 8k | 1.6% / 2.0% | 2.9% / 3.7% | 0.79 / 0.80 | 0.04 | 14% |

**At 20k, every check** (2026-10-06; layered pixels / pixel stage A, both 20k, CPU; `scripts/tetris_scenarios.py`,
`scripts/event_checks.py`). Scenario events, pixels wrong at +2 / +16 (11-14 games each):

| event | layered pixels 20k | pixel stage A 20k |
|---|---|---|
| gravity, no buttons | 33% / 34% | 43% / 55% |
| soft drop | 23% / 15% | 43% / 45% |
| rotate | 28% / 31% | 59% / 49% |
| spawn | 32% / 52% | 69% / 77% |
| shift | 56% / 39% | 71% / 62% |
| line clears (all sizes) | 99-100% / 91-98% | 91-100% / 82-94% |
| top-out, game start, level up | 88-100% | 89-100% |

Line clears followed for 60 frames (12 clears, 11 singles): full rows left 0.67 / 0.75 (the real game: 0),
filled cells over the real game's 1.53 / 1.22. Neither model clears: the full row stays and the stack never
drops (the layered model also spawns the wrong piece after). Restarts (12): pixels wrong 16 frames into the
menu 41% / 40% against 40% for copying the top-out frame, 8 frames into the new game 10% / 9% against 8%:
neither model gets through the menus. The model's gains are all in the moving piece; the rare whole-board
events (clears, top-out, menus, level up) are learned by neither.

## Every model side by side (2026-10-06, layered pixels at 28k)

What each model is:

| | original `small` | pixel stage A `base` | pixel stage B `srr` | layered slots | **layered pixels** |
|---|---|---|---|---|---|
| a frame's tokens | 256 screen patches, 16 x 16 | same | same | 255 background cells + 64 OAM slots + camera, frame, border: 367 | 255 background cells + 240 sprite-picture patches + camera, frame, border: 543 |
| background | in the screen patches | same | same | cells fixed to the world canvas | same as slots |
| sprites (the falling piece, the preview) | in the screen patches | same | same | 64 slots: 8 x 8 pixels, x, y, flags; moves as categories | their own picture: the sprite pixels that show, 16 x 16 patches |
| time in attention | one learned embedding per window slot | same | same | same (dropped at 44k) | **rotary (RoPE) in temporal attention** |
| temporal attention | by screen position | same | same | by identity: cells by world column, the rest by index | same as slots |
| a dream | soft: colour probabilities fed back | same | same | sprites committed (snapped) | soft, as the pixel model: cells, sprite picture and border fed back as probabilities (shown and scored as the most likely colour) |
| training | own guesses, then swaps; rollouts 1-32; tossing | teacher forcing on clean real context | `base` 98k + rollouts (depth 62 -> 16), history-only blend | teacher forcing, clean | teacher forcing, clean |
| size | 96 x 8 x 4, 2.0M | 2.0M | 2.0M | 2.5M | 3.0M |
| windows a step, speed | 16, 1.6 steps/s | 16, 2.0 steps/s | 16, 1.6 steps/s | 12 | 12, 0.76 steps/s (recompute, 12 GB cap) |
| start of a dream needs | screenshots | same | same | the frames' layers | same as slots: shipped as one start frame in its layers (background, sprite picture, border; the user, 2026-10-06) |

Long dreams, the same 32 live trials (hit counts trials at 0.75 or more; the original predates the
position scores: 28/32 intact, 0 erased at 52k on the old count score):

| model @ step | hit +32 | ghost | fall | spawn | wrong +16 / +128 | timer +128 | unseen change |
|---|---|---|---|---|---|---|---|
| layered slots @ 28k | 0.21 (3) | 0.93 | 1.51 | 0.00 | 2.2% / 3.6% | 23% | 85% |
| `base` @ 20k | 0.35 | 0.39 | 0.62 | 0.26 | 1.6% / 3.7% | 29% | |
| layered pixels @ 20k | 0.50 (15) | 0.62 | 0.88 | 0.58 | 1.4% / 3.0% | 18% | 11% |
| `base` @ 28k | 0.44 (9) | 0.38 | 0.83 | 0.32 | 1.2% / 3.2% | 23% | |
| **layered pixels @ 28k** | **0.81 (26)** | 0.36 | **0.96** | 0.79 | **0.6% / 2.3%** | 7.3% | 10% |
| `base` @ 50k | 0.72 (23) | 0.29 | 0.84 | 0.79 | 0.9% / 2.7% | 5.7% | |
| `base` @ 76k (the same 10 hours) | 0.60 (17) | 0.36 | 0.89 | 0.79 | 1.3% / 3.5% | 9.9% | |
| `base` @ 98k | 0.70 (21) | 0.33 | 1.01 | 0.79 | 1.0% / 3.2% | 5.7% | |
| `srr` @ 20k (stage B, shipped) | 0.73 (23) | 0.25 | 0.95 | 0.89 | — / 2.6% | 1.7% | 15% |

At 28k (10.2 hours) the layered pixels have the best hit, fall, wrong +16 and wrong +128 of any model,
stage B included, with stage A's recipe alone. It passed `base`'s best hit after 336k training windows;
`base` needed 800k (50k steps) to get there. Per hour of training it is ahead on hit and wrong pixels and
level on the clock (`base` at 76k, the same 10 hours). Which change did it is not measured (rotary time, the sprite layer, or both; the user's bet: rotary). The
layered model is kept either way: it is built for SMB's scrolling, and a layered start frame ships fine.
Still behind: spawn timing and the clock against stage B (0.79 / 7% against 0.89 / 1.7%), and which
piece spawns (the 28k sheet spawns an L where the preview held an S). Neither family learns line clears,
top-out or the menus (above).

## Layered pixels overnight: 33k to 58k with a cooldown (2026-10-06/07)

Stopped at 33.3k to pick up the new scores (shape, presence, bias) and relaunched with a cooldown: constant
learning rate to 46k, then linear to zero at 58k (`--cooldown 46000 12000`, `-Until 08:30`). The relaunch first
crash-looped: the checkpoint's optimizer state was in the parameter order of the code the run started with,
before the shared-model refactor, and it is matched by position. The trainer now saves the optimizer state with
each parameter's name and resumes by name (the checkpoint was migrated once; the original is kept as
`model_latest_step33300_positional_optimizer.pt` in the archive).

| step | hit (>= .75) | ghost | fall | spawn | timer +128 | wrong +16 / +128 | shape |
|---|---|---|---|---|---|---|---|
| 34k | 0.84 (27) | 0.33 | 0.86 | 0.74 | 12.5% | 0.56% / 2.47% | 0.66 |
| 38k | 0.92 (30) | 0.16 | 0.96 | 0.84 | 7.1% | 0.31% / 2.23% | 0.72 |
| 42k | 0.98 (32) | 0.05 | 1.00 | 1.00 | 4.2% | 0.10% / 1.98% | 0.71 |
| 46k (decay starts) | 0.87 (28) | 0.37 | 0.90 | 0.79 | 4.2% | 0.41% / 1.85% | 0.80 |
| 48k | 0.93 (30) | 0.16 | 1.02 | 0.89 | 4.2% | 0.23% / 1.48% | 0.76 |
| 52k | 0.87 (28) | 0.36 | 0.90 | 0.79 | 3.6% | 0.39% / 1.46% | 0.80 |
| 54k | 0.93 (30) | 0.20 | 0.92 | 0.89 | 2.1% | 0.25% / 1.47% | 0.73 |
| 56k | 1.00 (32) | 0.00 | 1.01 | 0.95 | 3.1% | 0.00% / 1.15% | |

Unlike the original small model's, this cooldown helped: wrong pixels at +128 from about 2.2% to 1.15-1.5%
(stage B: 2.6%), the clock from about 8% to 2-4%. Hit alternates between two levels (28 or 30-32 trials), a
couple of trials flipping, not a drift. Presence stayed 0.96-1.00: no stall.

Not learned: choosing the next piece. At every test (the bias test) the dream's preview changed only to one
piece type (two at 42k), its mix of shown pieces 0.64-0.75 from real play's, and 36-72% of its preview frames
garbled (less in the cooldown). The next piece spawns on time and whole, as the wrong piece. The bias test's
choice count included changes into and out of garbled previews; fixed on 2026-10-07 (the run's rows before
then count them).

**The final checks** (2026-10-07, GPU; layered pixels 42k and 58k against stage B 20k, the shipped model).
Long dreams at 58k: hit 1.00 (32/32), ghost 0.01, fall 1.00, spawn 0.95, wrong +16 / +128 0.01% / 1.22%, timer
+16 / +128 0.00% / 2.6%. Scenario events (25-31 games each, 40 instances; pixels wrong at +2 / +16, and exact
at +2):

| event | layered 58k | layered 42k | stage B 20k |
|---|---|---|---|
| gravity, no buttons | 2.2% / 6.7%, 80% | 6.2% / 11.7%, 83% | 15.7% / 19.6%, 65% |
| soft drop | 9.5% / 6.8%, 83% | 11.1% / 8.3%, 80% | 35.8% / 16.7%, 45% |
| shift | 4.6% / 3.5%, 85% | 7.2% / 8.2%, 73% | 54.2% / 43.1%, 5% |
| rotate | 5.9% / 9.9%, 88% | 7.0% / 14.5%, 85% | 43.9% / 39.5%, 33% |
| spawn | 20.8% / 23.7%, 0% | 24.5% / 32.6%, 0% | 18.0% / 53.2%, 55% |
| line clear single / double | 57-61% / 19-33% | 63% / 23-31% | 40-44% / 3-4% |
| line clear triple / tetris (16, 7) | 69-99% / 31-42% | 66-99% / 32-35% | 64-99% / 15% |
| level up (16) | 88% / 74% | 89% / 76% | 66% / 31% |
| top-out (22) | 30% / 0.5% | 59% / 3.6% | 43% / 0.1% |
| game start (23) | 2.1% / 3.8% | 68% / 7.0% | 97% / 97% |

Line clears followed to +60 (12; full rows left, well cells wrong, filled cells over real): 58k 0 / 2.8% / 1.02,
42k 0.25 / 5.0% / 1.19, stage B 0 / 2.6% / 1.00 (the layered model at 20k: 0.67 / 8.9% / 1.53). Restarts
(12; pixels wrong 16 frames into the menu, 8 and 63 into the new game; copying the top-out frame 40% / 8%): 58k
21% / 6.7% / 7.6%, 42k 24% / 12% / 13%, stage B 18% / 30% / 19%.

58k is the best model: far ahead of stage B on everything the player moves and on starting a game; level with
it on clearing lines at +60; behind on the clear's animation at +16, on level up, and on a spawn's exact first
frames. The spawned piece is right when it was already in the preview (8 of 8 first spawns, 10 games); the
random next piece is not chosen (one type at every test).

**Decoding the next piece (2026-10-07, 58k, 12 live trials, inference only; scratch `decode_experiment.py`).**
Variants of the sprite picture's decoding: passes (MaskGIT, surest tokens first), each decided token's pixels as
argmax, sampled per pixel, or sampled with one draw per token, and the frame fed back soft or committed:

| decoding | hit | ghost | shape | wrong +128 | next pieces chosen (real 10, 6 types) | garbled | later spawns follow own preview |
|---|---|---|---|---|---|---|---|
| 1 pass, argmax, soft (as trained) | 1.00 | 0.00 | 0.79 | 1.34% | 0 | 49% | 0/5 |
| 1 pass, argmax, committed | 0.95 | 0.09 | 0.67 | 1.50% | 0 | 68% | 0/7 |
| 4 passes, argmax, committed | 1.00 | 0.00 | 0.77 | 1.70% | 0 | 73% | 1/3 |
| 4 passes, one draw per token | 0.93 | 0.23 | 0.29 | 2.40% | 0 | 80% | 1/4 |
| 4 passes, pixels sampled | 1.00 | 0.00 | 0.73 | 1.79% | 0 | 57% | 2/7 |

With the bias count corrected (garbled previews are no choice), no decoding makes a dream show a new real piece:
after a spawn its preview turns garbled and stays so. The model's distribution there is itself a blur, with no
coherent piece in it for a sampler to pick; committing and sampling only add garbling. The decoding stays as
trained; choosing the next piece needs a discrete choice in the model (as Genie's codes: a token a whole patch
from a vocabulary), not in the decoder.
