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
moves together. (Confounded, found 2026-10-07: this model had the broken clock, the learned per-slot time
below, and freezing in mid-air is what a fall timer that never advances does. Rerun with rotary time the same
day, it failed the same way: see the slot retry at the end.)

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

## The slot model retried with rotary time (`world_model_tetris_slots_rope`, 2026-10-07, stopped at 14k)

The first slot run had the broken clock (learned per-slot time), so its verdict was confounded. The user's
idea (sprite tokens with their own position) got a retry: the same model and settings, with the layered models'
rotary time (`train_layered.py --model slots`), 1.2-1.4 steps/s at 12 windows (367 tokens a frame). Long dreams
on the same 32 games, against the first slot run and the layered pixels run at the same steps:

| step | wrong +16 / +128 retry, first, layered px | hit retry, first, px | ghost retry, first, px | timer wrong +128 retry, first, px |
|---|---|---|---|---|
| 4k | 3.4/5.4%, 3.0/5.1%, 1.9/3.3% | 0.13, 0.02, 0.02 | 0.57, 0.32, 0.00 | 30%, 23%, 24% |
| 6k | 3.0/4.5%, 2.9/4.8%, 1.5/2.9% | 0.10, 0.17, 0.02 | 0.67, 0.72, 0.01 | 32%, 40%, 29% |
| 10k | 2.5/4.5%, 2.8/4.4%, 1.7/3.0% | 0.19, 0.16, 0.05 | 0.79, 0.81, 0.04 | 38%, 42%, 25% |
| 12k | 2.5/4.4% | 0.17 | 0.87 | 39% |
| 14k | 2.2/4.2% | 0.20 | 0.85 | 39% |

Rotary time changed nothing for the slot model: it tracked the first slot run step for step, its ghosts grew
(0.57 to 0.87; the layered pixels model's stayed near 0), and its clock never came (timer +128 ~39%, where the
layered pixels model counted by 4k). So the clock was not what failed the slots; the per-sprite position
choices are. Called by the user at 14k.

## Rotary time on the pixel model, then attention variants (2026-10-07)

**Rotary pixel (`world_model_tetris_rotary`):** the old `base` (pixel model, 256 tokens, clean context) with its
learned time positions replaced by the layered model's rotary time, nothing else changed; 2.2-2.3 steps/s at 16
windows (about 2.7 times the layered runs' windows a second). Stopped at 11.9k by the user to try attention
variants. Long dreams (the same 32 games):

| step | hit | ghost | fall | wrong +16 / +128 | timer wrong +16 / +128 | spawn |
|---|---|---|---|---|---|---|
| 4k | 0.03 | 0.00 | 0.12 | 1.8% / 3.3% | 22% / 27% | 0.00 |
| 8k | 0.55 | 0.62 | 0.70 | 1.3% / 3.3% | 6.6% / 27% | 0.00 |
| 10k | **0.89** | **0.05** | 1.01 | **0.28%** / 2.6% | 0.4% / 22% | 0.00 |

At 10k it beats `base` at 98k (hit 0.70, ghost 0.33) and stage B at 20k (0.73, 0.25), and is far ahead of the
layered pixels run at the same step (hit 0.05): rotary time, not the layers, made most of the jump (the user's bet).
Still behind the layered 58k on spawns (0.00, learned last by `base`, at 34-48k) and the long clock (+128).

**Attention variants on the rotary pixel model, 10k steps each, against its 10k:**

| variant | hit | ghost | wrong +16 / +128 | timer wrong +16 / +128 |
|---|---|---|---|---|
| rotary pixel (baseline) | 0.89 | 0.05 | 0.28% / 2.6% | 0.4% / 22% |
| + 2D rotary in space (RoPE-Mixed, `--space-rope`, no learned place) | 0.62 | 0.15 | 0.80% / 2.6% | 0.9% / 16% |
| + QK-norm (`--qk-norm`, both attentions) | 0.58 | 0.14 | 0.77% / 2.9% | 1.5% / 20% |

2D rotary in space: level with the baseline to 8k (hit 0.47 / 0.55, fewer ghosts, a better clock), behind at 10k
on pieces. One test at one step (about +-4 games of noise) and before the 20k the runs are usually judged at: not
better at 10k, not disproven. Its checkpoint is kept to resume.

QK-norm: level or a little ahead at 8k (hit 0.58 / 0.55, ghost 0.43 / 0.62, the first spawns, 11%), behind at
10k like the other variant. All three were level at 8k (hit 0.47-0.58) and both variants sit near 0.6 at 10k, so
the baseline's 0.89 at 10k may be a lucky test (it stopped at 11.9k: no 12k to tell). Reading: the baseline is at
least as good as either variant; neither showed a gain worth keeping. Both checkpoints are kept.

**Both on, no learned place (`world_model_tetris_rope_qk`, 2026-10-07 19:33, stopped at 20k):** behind every run at
10k (hit 0.39, ghost 0.37, wrong 1.5% / 3.3%, timer +16 14%), then caught up: at 20k hit 0.50, ghost 0.19, **spawn
0.79, timer wrong +128 6.9%**, wrong 1.1% / 3.2% (the layered pixels at 20k: 0.50, 0.62, 0.58, 18%; `base` at 20k:
0.35, 0.39, 0.26, 29%). So the variants were slower to start, not worse: 10k was too early to judge them (the user's
doubt).

The space_rope runs had dropped the learned place per patch: no token knew where it was on screen, only how far it
was from the others, while Tetris tokens carry meaning by place (the clock digits, the next-piece box). The
RoPE-Mixed paper keeps it at the training resolution (RoPE-Mixed + APE about doubles RoPE's gain there; the layered
model's space_rope kept its place embeddings too). `space_rope` now keeps `space_pos` (2,006,682 parameters), and the
overnight run is `world_model_tetris_rope_ape_qk` (rotary time, learned place, 2D rotary in space, QK-norm), from
22:30, no end set.

**The overnight run and its cooldown (`world_model_tetris_rope_ape_qk`, 22:30 to 06:33, then 51.4k-63.4k).** Long
dreams peaked at 26k (hit 0.70, ghost 0.18, spawn 0.84, wrong 0.8% / 2.2%, timer +128 2.6%) and drifted at the
constant learning rate (48k: hit 0.25, ghost 0.76); spawns and the clock held (spawn 0.8-0.9, timer +128 3-7%). The
run died at 06:33 when the Microsoft Store updated Windows Terminal and closed its console (runs now start in
`conhost`). A cooldown from 51.4k over 12k steps made it worse, not better (58k: hit 0.30, ghost 0.71; 62k: 0.35,
0.60): the lower the learning rate, the more it hedges a piece between places.

**The play check (`event_checks.py --checks play`, 8 live stretches with real buttons, the same as 2026-10-07's):**

| model | cells lost / 100 frames | morphs / 100 frames | shape kept (4 stretches) |
|---|---|---|---|
| stage B 20k (pixel, learned time, rollouts) | 0.31 | **0.57** | 0.43 |
| rope 26k (rotary time, place, 2D rotary, QK-norm) | **0.21** | 0.73 | 0.27 |
| rope 63.4k (after the cooldown) | 0.52 | 1.04 | **0.50** |
| rotary 10k | 0.89 | 1.93 | 0.20 |
| layered 58k (the page's) | 0.89 | 1.51 | 0.20 |

Rope 26k beats the page's layered 58k on all three and is the best at not erasing cells; stage B still morphs least.
The sheets show where pieces break: at the player's rotation. In stretch 1 the real T rotates within a few frames
and every new model turns it into something else there (a 5-block bar, a plus, a 2-block bar, an O) and then lets
it fall correctly. A rotation moves blocks by a jump, not a slide; stage B, the one model trained on its own
rollouts, kept that T (0.87). Shape rests on 4 stretches, so it is rough.

## What the play check missed: the controls (2026-10-08)

**The scenario checks with a real history** (`tetris_scenarios.py`, about 100 instances per event, rope 26k and
the layered 58k on the same instances; pixels wrong at +2 of the pixels that changed / exactly right at +2):

| event | rope 26k | layered 58k |
|---|---|---|
| gravity, no buttons | 16% / 63% | 2.7% / 85% |
| shift | 59% / **3%** | 5% / 84% |
| rotate | 48% / **11%** | 8% / 80% |
| soft drop | 47% / 41% | 5% / 88% |
| line clears | about level | about level |
| game start | 99% wrong | 2% wrong |

Rope 26k had not learned what the buttons do: with a perfect history it shifts a piece exactly right 3% of the
time. Its long dreams press no buttons, so they could not see it, and its better play score was mostly pieces
that barely moved. The earlier pixel models were as bad (stage B: shift 5%, rotate 33% exact). The layered model
learned the controls: so the 58k model's jump had two parts, rotary time (the clock, long dreams) and something
in the layered model that lets it learn the buttons.

**Rollouts did not fix it** (both grown from rope 26k, stage B's settings, learning rate 3e-5, a frozen copy
refreshed every 2k): deep rollouts (`world_model_tetris_rope_srr`, 62 -> 16 frames, blend 8) made long dreams
steadily worse (10k: hit 0.54, ghost 0.38, from 0.70 / 0.18), spawns better (0.95); 2-frame rollouts
(`world_model_tetris_rope_r2`, 1-2 frames, no blend) worse still (10k: hit 0.32, ghost 0.52). Scoring the model on
the real frame after its own (often slightly off) frames pushes it to hedge, and a model that does not know the
controls cannot be taught them by its own rollouts.

**The training data is right** (`stream_audit.py`, session scratchpad; 300 windows of the pixel stream, 150 of
the layered): every rotation (357/357) and every one-column shift came one step after its recorded press (the
other 52 "shifts" were spawns one column over); both streams pair frames and buttons the same way (the same
`world_nes_windows` code and `incoming_actions`) and show the move with the same timing; a piece shifts in 3.4%
of steps and rotates in 1.9%.

**How the buttons enter:** in both models one learned vector per controller byte is added to every token of the
frame it produced, once, at the input. The press shows on screen two frames later (the game reads it, then the
picture shows the RAM of the frame before), so the model must carry it through 8 blocks and reach back a frame.

**Action conditioning (`--action-norm`, `world_model_tetris_rope_actnorm`, from scratch, rope's settings):** the
buttons also set a scale and shift on every block's three norms (adaptive LayerNorm), from zero. 2.45M
parameters, the same speed (1.82 against 1.84 steps/s). The first change that improves how the pixel model
follows buttons: its previews (frames dreamed from real history with the real buttons, changed pixels wrong)
are about 15% better one frame ahead from 3k on (6k: 0.24 against 0.28; 16 frames ahead 0.41 against 0.45). Long
dreams at 6k were mixed (hit 0.21 / 0.36, ghost 0.90 / 0.73), but they press no buttons and settle later. The
scenario checks at 10k, against rope at 10k (same instances, CPU; exactly right at +2, with the layered 58k for
reference):

| event | actnorm 10k | rope 10k | layered 58k |
|---|---|---|---|
| shift | **48%** | 0% | 84% |
| rotate | **38%** | 0% | 80% |
| soft drop | **81%** | 54% | 88% |
| gravity, no buttons | 57% | 55% | 85% |
| game start (pixels wrong at +2) | **15%** | 99% | 2% |

Line clears, level up and top-out are level. The buttons work now: a pixel model shifts and rotates pieces
for the first time, with no other change, and starts a game from the menu. Half the way to the 58k at a sixth of
its steps; the run keeps training. The no-button long dreams could never have shown this, so since 2026-10-08
every long-dream test also dreams the same games with the bot playing on (below).

## Long dreams with buttons (2026-10-08)

Each long-dream test (both trainers, every 2,000 steps; `long_dream_check.py --buttons both|none|pressed`) now has
two sets of 32 trials from the same games: no buttons (as before, variant `1 pass` / `layered t=1`) and the bot
playing on, its presses given to the real game and to the dream alike (variant `buttons` / `buttons t=1`). The
scores are the same; with buttons, hit and shape ask that the dream moves and turns the piece where and when the
real one does. The window draws each kind in its own row of the curves, the Compare tab compares each kind with
the other run's same kind, and long_dream.gif/.png show the no-button trial above the pressed one. Runs from
before have no `buttons` rows until retested (`long_dream_check.py` on their archived checkpoints).

**Actnorm 14k against the layered 58k (2026-10-08, CPU, the same instances for both).** Scenario checks, exactly
right at +2: shift 81% / 92%, rotate 72% / 72%, soft drop 88% / 85%, gravity 79% / 85%; game start 7% / 2%
wrong; spawn 26% / 14% wrong; single line clear 86% / 56% wrong (0% / 30% exact). Long dreams (8 games each):
no buttons, hit 0.97 / 1.00, wrong +128 1.4% / 0.6%, shape 0.77 / 0.87; the bot's buttons, hit 0.37 / 0.68,
ghost 0.45 / 0.18, shape 0.21 / 0.43, fall 0.80 / 1.02. At a quarter of the steps the pixel model with action
conditioning has caught up on single presses (shift, rotate, soft drop); it is behind on line clears, spawns and
holding a played piece over a long pressed dream. The layered model too keeps only about two thirds of a played
piece at +32 (the pressed long dreams' first numbers for it).

**Overnight to 60k, then a cooldown (`--cooldown 60140 12000`, 2026-10-09).** The training tests (32 trials of each
kind, the same games every test) kept improving to about 42k, then held:

| step | no buttons: hit / wrong +128 | buttons: hit | ghost | spawn | mass |
|---|---|---|---|---|---|
| 14k | 1.00 / 1.98% | 0.64 | 0.23 | 0.86 | 1.02 |
| 30k | 1.00 / 1.31% | 0.91 | 0.13 | 0.86 | 1.02 |
| 42k | 1.00 / 1.39% | 0.96 | 0.07 | 0.96 | 1.03 |
| 58k | 1.00 / 1.10% | 0.95 | 0.04 | 0.96 | 1.09 |
| 62k (cooldown) | 1.00 / 1.40% | 0.97 | 0.02 | 1.00 | 1.04 |

The rope run without action conditioning (identical otherwise, the same data) had no-button hit 0.35, ghost 0.62
at 60k, and its finished cooldown (to 63.4k, learning rate 2.5e-8) did not change that.

### What the action-conditioning run taught us

1. **The buttons were the bottleneck, not the recipe.** Added once at the input, the buttons barely reached the
   prediction: the pixel models never learned to shift or rotate (0-5% exact). Conditioning every block's norms
   on them (adaptive LayerNorm, what DiT and Open-Oasis do) fixed it with no other change: 48% exact shifts at
   10k, 81% at 14k.
2. **It also fixed the no-button dreams.** The one difference from the rope run took hit at +32 from 0.35 to
   1.00 and ghosts from 0.62 to 0. Most likely: a model that cannot see why a piece moved sees moves as random,
   so it learns to hedge (ghosts, smears, stalled pieces) everywhere, including when nothing is pressed. Once the
   buttons explain the moves, what is left (gravity, spawns) is predictable.
3. **Rollouts were treating a symptom.** We tried deep and 2-frame rollouts against the drift; both made it worse,
   and this model needs none to dream 128 frames cleanly. Whether rollouts still add something on top is open.
4. **The layers were not what made the 58k model good at controls** in a way the pixel model cannot match; why the
   layered model learned the buttons through the same input-only path is still unexplained (its moving piece is
   separate tokens with their own move and place losses, which may have made the presses easier to find).
5. **Tests must press buttons.** The no-button long dreams scored rope 26k as good while it ignored every button;
   the scenario checks found it. Long-dream tests now dream both kinds.
6. **It learns fast.** Furniture in about 800 steps; the 58k layered model's no-button dreams matched by 14k.
7. **Still weak:** line clears (at 14k: 0% exact at +2, against 30% for the 58k) and extra blocks late in pressed
   dreams (mass up to 1.09 at 58k, 1.04 in the cooldown).

**Actnorm 58k against the layered 58k, equal steps** (scenario checks, CPU, the same instances; exactly right at
+2, pixels wrong at +16 for the clears):

| event | actnorm 58k | layered 58k |
|---|---|---|
| gravity, no buttons | **91%** | 76% |
| falling alone | **96%** | 83% |
| soft drop | 93% | 89% |
| shift | **97%** | 90% |
| rotate | **88%** | 73% |
| spawn | **68%** | 0% |
| game start | **77%** | 0% |
| single clear (exact; wrong at +16) | 48%; 0.0% | 38%; 14% |
| double clear (exact; wrong at +16) | 31%; 1.9% | 0%; 36% |
| level up (wrong at +16) | 55% | 78% |

At the same number of steps the pixel model with action conditioning beats the layered model on every event
but level up, including line clears, which it was worst at in the 14k check. It is the best model so far.

**Stuck after the curtain is right (2026-10-09).** In the play window the actnorm models stayed on the covered
board after a top-out, where older models went back to the menu on their own. The real game does the same:
after the curtain it waits for Start indefinitely (no level menu after 1,800 frames with no buttons; 2-22 frames
after a tap). A dream that tops out before the real game gets no Start (the bot presses Start for the real
game's end), so it waits; the older models left without one because they ignored the buttons. With the real
history and buttons (restart check, 15 restarts) actnorm 72k reaches the menu in 11 (menu wrong under 5%,
copying 40%), the layered 58k in 9. The play window's S key now taps Start into the dreams alone.

## Resynced rollouts (2026-10-09)

**Why:** every rollout run so far made long dreams worse. A rollout's history drifts out of sync with the real
stream: the bot pressed left because the real piece was in column 5, and the dream's is in column 4, so the real
buttons and the real targets contradict the history, and the cheapest answer is to hedge. Papers score rollouts
against a teacher model instead of the real frames (Self Forcing's distribution matching); we have no memory for
a second model. Instead the rollout stays in sync: after each rollout frame, every patch with at least
`--resync` pixels whose most likely colour is not the real one is replaced by the real patch before the next
frame reads it (`FrameDecoder`, `models/dynamics.py`); the rest keeps the model's own soft pixels.

**The threshold** (actnorm 72k, mid-game windows): one step from real history 99.94% of the well's patches are
exactly right; over 16 dreamed frames 0.8% go off, nearly all by a whole block (48-64 pixels), and almost none
by 1-7. So `--resync 8` separates going off from the small errors cleanly, and what the model trains on is the
real game's content with its own haze (the probability spread it dreams with), which teacher forcing never shows
it. (The console's boot screen is the exception: the model draws the copyright text badly, the one screen it
rarely sees.)

**Run:** `world_model_tetris_actnorm_resync`, grown from actnorm 72k; `--models srr --rollouts 16 16 1 --resync 8
--refresh 2000 --lr 3e-5 --weight-decay 1e-5 --warmup 500` (the old stage B's fine-tune settings). 1.38 steps/s
against 1.7 teacher-forced (the prefill of the real history before each rollout is most of it); 20 GB.

**To 8k, the 128-frame tests did not move** (hit 1.00 / 0.95-0.98 with buttons, ghost 0.01-0.03, within noise)
and, unlike every earlier rollout run, did not get worse: out-of-sync history is what broke those. The test was
saturated, so it became 1,000 frames (below).

## Long dreams of 1,000 frames (2026-10-09)

The long-dream test dreams 1,000 frames (`long_dream.py` FUTURE); the scores so far look at the first 128 (FIRST),
the same numbers as before, and the play check's integrity scores (`play.py integrity`) run over all 1,000, per
100 frames: `lost_long` (cells vanishing outside a clear), `morph_long` (a piece's block count changing),
`full_long` (full rows that never clear; boards topped out, under the curtain, left out). They need nothing of
the real game, which a dream stops following once it chooses its own random pieces. Trials are dreamed 16 side by
side (`Dreamer` takes a batch): 32 trials x 1,000 frames in about 78 s, a test about 2.6 min every 2,000 steps;
the trial queue keeps only the trials the tests use.

On the trainer's own trials (seed 43):

| | hit +32 (none / buttons) | shape (buttons) | lost / morph / full per 100 frames, buttons | no buttons |
|---|---|---|---|---|
| actnorm 72k | 1.00 / 0.97 | 0.63 | 0.61 / 0.93 / 0.35 | 0.04 / 0.07 / 0 |
| resync 8k | 1.00 / 0.95 | 0.69 | 0.59 / 0.78 / 0.56 | 0.06 / 0.09 / 0 |

Over 1,000 frames of play the model loses about 6 cells and morphs about 9 times: the drift the 128-frame test
could not see. Resync at 8k is level with its start; the run goes on with the new test.

**Size (2026-10-09).** The aim of the long dreams is coherence, not following the real game: past a few pieces a
dream picks its own (the game's pseudo-random choices cannot be followed), so a long dream is judged on the board
alone. `morph` missed the case the user named, "too small or too big block sets": a piece is born in the top
rows, where scores are left out (pieces come into view a row at a time), so one born with 3 or 5 blocks is
already wrong when it leaves them and never changes after. `size` (play.py, `size_long` over 1,000 frames)
compares the cell count across each spawn or clear: it must change by 4 per piece less 10 per cleared row. Top-outs
(5 or more full rows: the curtain; until the well is empty again) are left out of size and full. Ten real games,
30,000 frames: size 0, full one stray frame.

## The slot model again, with the buttons conditioning every block (2026-10-09)

**Why:** the slot retry failed with growing ghosts (0.57 to 0.87), hit near 0.2 and no clock. Action conditioning
took the pixel model's ghosts from 0.62 to 0 with nothing else changed (the rope run against actnorm): a model
that cannot see why a piece moved hedges, and a slot model's four sprites hedge each on their own. Slots also
commit (a sprite is shown or not, moves by a chosen amount), the opposite of the pixel model drawing a piece
with the wrong number of cells.

**Restored** (`layered_slots.py`, never committed, rebuilt from the session transcript: the file as written,
then its later edits), with `build()` choosing it again, the dream passes (`steps`), `train_layered.py --model
slots --dream-steps`, and its tests. New: `--action-norm` for every layered model (`LayeredBlock`, as the pixel
model's), and the layered long dreams 16 side by side (`dreams_layered`): the retry's 128-frame test took 6 min
one trial at a time, which at 1,000 frames and both kinds would have been about 90.

**Run:** `world_model_tetris_slots_actnorm_small`, the user's choice of the smaller slot model: the retry's
settings (lr 3e-4) plus `--action-norm`, at dim 64 and 6 layers; 1.44M parameters; compared with the retry,
`world_model_tetris_slots_rope`. Started from scratch at 17:5x with 32 histories (the user: bigger batches for a
steadier loss), `--vram-gb 22`. At 12 histories it trained at 2.07 steps/s (24.8 windows/s, 5.1 GB) and at 32
at 0.80 (25.6 windows/s, 13.6 GB): the GPU was already at full compute, so the batch buys steadier steps, not
more data a second. The first attempt (12 histories, `--vram-gb 7`) ran out of memory at every long-dream test
(the batched layered dreams need about 3 GB more) and resumed four times to 8k; each resume starts every history
at a fresh console, so it trained on game starts far too often: kept as `..._looped`.
(A full-size run with it was started and stopped at once; dim 96 needed `--vram-gb` above 10. The first small
run died at step 600 in a driver recovery set off by force-killing the other run; its folder is kept as
`..._crashed`.)

## Where the pixel model breaks pieces: rotations (2026-10-14)

After the latent route (video VAEs n1, c4/c6, n1s, n1sn and their generators; docs/guides/video_vae.md, latent_world_model.md) failed to beat the pixel model on coherence: per 100 frames of 1,000-frame dreams, n1sn's generator at 22k lost 3.9 cells and morphed 4.4 times with no buttons (actnorm + resync: 0.03 / 0.08) and 7.0 / 8.1 with them (0.40 / 0.60).

**Breakdown** (scratchpad coherence_breakdown.py: the 32 long-dream trials with the bot's buttons, every frame where cells vanish or the count changes by anything but 0 / +4 under the integrity rules, attributed per 100 frames of each situation). actnorm 72k, by the incoming button: none 0.92 morphs per 100 frames (66% of frames), down 0.53, left/right 4.6, rotate 10.0. Resync 18k: none 0.76, down 0.50, left/right 2.2 (resync halved the shifts), rotate 9.9. By frames since the last rotate press (resync 18k): 0-2 frames 13.2, 3-7 1.6, 8-15 0.9, 16-47 0.1, 48+ 0.35; since the last shift: 0-2 4.1, then 0.1-1.3. About two thirds of all morphs fall within two frames of a rotation.

**One frame ahead from real history** (scratchpad split_check.py, live play windows, the pixels that change; the incoming buttons, as trained: a first try passed the stream's actions a frame early and blamed the model): actnorm 72k rotate 94% exact, 6% a split (the real colour has some share, the top under 0.9), 0% confidently wrong; shift 98 / 2 / 0; soft drop and gravity 100%. Resync 18k: rotate 92 / 8 / 0, shift 100 / 0 / 0. So the model knows how a rotation looks; on about one rotation in 13-16 it spreads its probability between rotated and not, and in a dream that blend goes back into the context and the piece morphs.

**Which rotations** (scratchpad rotation_check.py: every A/B press edge while a piece falls, live play at stride 1, one frame ahead from 63 real frames, scored on the piece's 6 x 6-cell neighbourhood). Resync 18k, 300 presses: blocked turns (the NES has no wall kicks) 99% exact; accepted turns 40% exact, 52% split, 8% confidently wrong. By piece: S 100%, O 93%, I 92%, Z 88%, but L 37%, T 33%, J 24%: the pieces with four orientations, where a turn can go two ways. (split_check's 92-94% averaged every piece and blocked presses over the changed pixels.) What the model drew on accepted turns: the real turn 135, the opposite turn 48, a mix 43, no turn 2.

**The model ignores which rotate button was pressed** (scratchpad swap_check.py: J/L/T turns where the opposite turn would also fit; the same history, one frame ahead, with the real press, the press swapped A <-> B, and no press). Resync 18k, 200 turns: real press 37% the real turn / 24% the opposite / 39% a mix; swapped 34 / 28 / 37; no press 100% no turn. actnorm 72k: 30 / 24 / 44 against 28 / 27 / 44. actnorm 20k: 10 / 8 / 64 mix. The shipped layered 58k: 12 / 12 / 68 mix against 12 / 16 / 63. Every model knows *that* a rotate button was pressed (no press: never a turn) and none uses *which*: swapping A and B changes the drawn turn by 3 points or less. Shifts, by the same test (scratchpad swap_shift.py, 150 shifts): real 98% the real move, swapped 98% the other way. So the button reaches the model through the same path (a tap whose effect shows a frame after the frame it is attached to) and left/right is used fully; A/B is not.

The data is consistent: in the training stream (WORLD_MODEL screens), every J/L/T turn under A goes +1 in the orientation cycle (1,901) and under B goes -1 (447); B is 19% of turns. The perfect brain (45% of short games, all of the long ones) always turns with A toward its planned orientation; B comes only from its random spins and from the holding brain. The first picture to show a turn is the one after the picture whose incoming button is the press (render lag; the same for shifts).

So the rotation morphs are a hedge between the two directions, which the button would settle. A fix has to make the model learn A vs B.

## One stage: actnorm, resync rollouts from step 0, short-way turns (`world_model_tetris_onestage`, 2026-10-14)

**Why:** a full run that folds in what the runs since actnorm taught, in one stage (the goal: one recipe, one
stage, about 100k steps), chosen with the user:
- **the model:** actnorm's, unchanged (2.45M parameters: 8 layers of 96, 4 heads, 16 x 16 patches, 64 frames,
  rotary time and space with the learned place, QK-norm, the buttons on every block's norms);
- **resync rollouts from the first step** (`--rollouts 16 16 1 --resync 8 --refresh 2000`), not as a second stage
  grown from a base: as a fine-tune they took lost / morph with buttons from 0.61 / 0.93 to 0.40 / 0.60 per 100
  frames and halved the shift morphs. From scratch they schedule themselves: while the model is a blur, nearly
  every dreamed patch is 8 or more pixels off and is replaced by the real one, so early training is close to
  teacher forcing; the model's own frames stay in the history only as it gets them right. (Untested from
  scratch: the risk is a slower copy phase.) About 19% slower a step (1.38 against 1.7 steps/s);
- **short-way turns in the data** (tetris_bot.turn_button): the perfect brain turned only with A, so 81% of turns
  were A and no model learned which way B turns (above). Now A 1,042 and B 1,030 J/L/T turns in the training
  stream, each button still always the same way;
- **the base run's schedule:** lr 3e-4, warmup 1,000, weight decay 0.05, 16 histories, seed 43; cooldown from
  88,000 for 12,000 steps (the end at 100k);
- checked and left alone: the pixel head's loss under bf16 (the VAE's precision bug) agrees with float32 within
  0.4% eager and compiled (resync 18k, 3 masks); the tick border stays on.

**The check routine, every 2,000 steps** (report-only: the run does not stop on its own): the trainer's long dreams
(1,000 frames, no buttons and the bot's buttons; their trials now come from today's bot, so the old runs'
curves compare only roughly: `output/trends/baseline_newbot.csv` has actnorm 10k, 20k, 40k, 72k and resync 18k on
the new trials), `scripts/button_swap_check.py` (turn and shift gaps), the event checks, and at 10k steps the
scenario checks.

**Predictions (written before launch):**
1. **Copy and motion phases at actnorm's pace:** the previews' changed pixels wrong at +1 within 1.5x actnorm's at
   the same step from 4k on. Worse than that at 4-8k means the early rollouts slow the copy phase.
2. **The turn gap opens; the shift gap stays:** every model so far had a turn gap of +0.00 to +0.03 (actnorm 20k,
   72k, resync 18k). Here over +0.3 by 20k and over +0.5 by 40k; the shift gap over +0.9 from 14k (actnorm had 81%
   exact shifts at 14k). If the turn gap stays under +0.1 to 30k, balancing the data was not enough and the model
   is next (each frame given the button before it as well).
3. **Coherence with buttons beats actnorm at the same step** on lost and morph per 100 frames over 1,000 frames,
   and ends under resync 18k's 0.40 / 0.60 (on the new trials' baseline), most of the gain within 2 frames of a
   turn.
4. **No-button dreams as actnorm's:** hit 1.00 by 14k, lost / morph near 0.03 / 0.08.
5. **The clock follows coherence** (learning phases): spawn with buttons near actnorm's at the same step, and
   higher once morphs are lower.

**Baselines on the new trials** (the trainer's 32 + 32 trials of seed 43 from today's bot; scratchpad
baseline_dreams.py, `output/trends/baseline_newbot.csv`; the swap gaps from `scripts/button_swap_check.py`, each
run's `button_swap.csv`). Per 100 frames over 1,000, with the bot's buttons (no buttons in brackets):

| | lost | morph | full | shape | spawn | turn gap | shift gap |
|---|---|---|---|---|---|---|---|
| actnorm 10k | 1.24 (0.23) | 2.32 (0.54) | 3.01 | 0.17 | 0.71 | +0.01 | +0.35 |
| actnorm 20k | 1.26 (0.06) | 2.16 (0.30) | 0.26 | 0.29 | 0.92 | +0.00 | +0.78 |
| actnorm 40k | 0.87 (0.03) | 1.31 (0.08) | 0 | 0.45 | 0.96 | +0.01 | +0.95 |
| actnorm 72k | 0.48 (0.01) | 0.78 (0.02) | 0.32 | 0.55 | 1.00 | +0.08 | +0.98 |
| resync 18k | 0.47 (0.01) | 0.60 (0.02) | 0.53 | 0.54 | 1.00 | +0.07 | +0.99 |

**Onestage stopped at 37,430 (2026-10-14 20:11), for the experiment below; it resumes from its last checkpoint.**
To 36k: no-button dreams the best of any pixel run (wrong +128 1.2-1.4%, lost / morph 0.03-0.12 / 0.09-0.21 per
100 frames, against actnorm 40k's 2.4% and 0.03 / 0.08 on the same trials); with buttons 0.96-1.11 / 1.47-1.66
against actnorm 40k's 0.87 / 1.31, about 4-6k steps behind it since the motion tier came late (shape 0.6 at 14k
against 8k). Prediction 2 failed: the turn gap +0.03 at 30k (shift +0.72). The finer measure (button_swap_check's
p gap: the probability of the real frame on the cells only one of the two turns fills, real press minus swapped)
is +0.002 at onestage 30k and +0.026 at actnorm 72k (shifts +0.67, +0.76): the button carries no turn direction
at all, not a little that grows. Tapped shifts are as exact as held ones (actnorm 72k: 100% either way), so the
frame between the press and the picture that shows it is not what stops turns.

## A shallow model with hidden frames (`world_model_tetris_s3_hidden`, 2026-10-14)

**Why:** the masked training shows the model about a third of each predicted frame on average (the cosine rate's
mean is 0.64), so when part of a turned piece is visible it can read the direction off the frame; a dream draws
every frame with nothing of it visible, and there only the button tells. `--hidden 0.5`: half the masked frames
are hidden entirely (training_mask), so most turns are trained the way they are dreamed (the piece's patches all
hidden: about 42% of turns before, about 71% now). The user's choice of a fast experiment: a shallow model,
96 x 3 (`s3`, about 1M parameters), teacher forcing (rollouts cost about 40% a step and the turns failed with and
without them), otherwise actnorm's settings (lr 3e-4, warmup 1,000, weight decay 0.05, 16 histories, seed 43, the
short-way-turn bot), until stopped. No unhidden 96 x 3 beside it: every model so far had a p gap of 0 to +0.03,
so a gap that opens here is the hidden frames; one that does not may also be the depth.

**Predictions (before launch):**
1. Faster: about 2-2.5 steps/s (actnorm 1.75 teacher-forced at 8 layers; the embedding and the head do not shrink).
2. The turn p gap opens: over +0.05 by 20k, over +0.15 by 30k (onestage +0.002 at 30k); the turn gap (drawings)
   follows later. If the p gap is still under +0.03 at 20k, hidden frames are not it, and the change-weighted
   loss is next.
3. The cost: more of the history the model learns from is hidden (half the masked frames whole), so copying and
   the board in long dreams come later than actnorm's; and at 3 layers whole pieces may stall (the 128 x 2 latent
   model stuck at about 2 lost pieces per 100 frames). Both are judged against actnorm at the same steps.

**s3_hidden to 16k (stopped at 16,045, 2026-10-14 22:32):** the board held (wrong +128 3.7%, actnorm 10k on the
same trials 3.0%) but whole pieces never came: shape 0.03-0.08 from 8k to 16k (actnorm 0.68 at 10k, onestage
0.68 at 16k), spawn 0, wrong +16 flat at about 2% (actnorm 10k 0.3%); and at 8k it did not react to any button
(the press, the swapped press and none drew the same). The motion tier stalled, from the hidden frames or the
depth: no unhidden 96 x 3 ran beside it. The fused head and the embedding-bag patches (dynamics.md, Speed) took it
from 2.3 to 3.0-3.3 steps/s from 8,938 on.

## One more layer: s4_hidden (`world_model_tetris_s4_hidden`, 2026-10-14)

**Why:** the user's choice before deciding between the hidden frames and the depth: grow s3_hidden at 16,045 by
one layer (`deepen` now inserts single blocks: the old blocks at round(i L' / L), the new one third, its outputs
zero, so it starts computing what the 3-layer model did) and train 10,000 steps more, everything else the same
(`--hidden 0.5`, teacher forcing; a fresh optimizer with the 1,000-step warmup). Steps count from 0 again: its
step k is s3_hidden's 16,045 + k.

**Predictions (before launch):**
1. If depth was the stall: shape over 0.3 within 6k of the growth (s3_hidden's 16k: 0.08), wrong +16 under 1%,
   and the shift gap above 0 by 10k (onestage reached shape 0.35 two thousand steps after its stall ended).
2. If shape stays under 0.15 at 10k, depth is not it, and the hidden frames are the likely cost: the next run is
   s3 (or s4) without them.
3. Speed: about 2.7-3.0 steps/s (a fourth block on a step the head and patches no longer dominate).

**The hidden chain (2026-10-14/15):** s4_hidden to s10_hidden, each grown one layer (`deepen`) and trained 10,000
steps, run by a chain script; stopped by the user's choice at s10's 2,540 (total 88.5k steps).
- Each layer added a tier: 4 layers broke the 3-layer stall (shape 0.08 -> 0.48, wrong +16 1.8% -> 0.3%); 5 brought
  the clock (spawn 0.11 -> 0.68); by 9 layers (76k steps) the no-button dreams matched or beat actnorm 72k on the
  same trials (wrong +128 1.7% against 1.8%, shape 0.75 against 0.69, spawn 1.00 against 0.96), with more breaks
  (0.07 / 0.13 lost / morph per 100 frames against 0.01 / 0.02).
- With buttons it reached actnorm 40k's lost cells (0.85 / 1.52 against 0.87 / 1.31) but about twice actnorm
  72k's morphs (0.78).
- Shifts firmed with depth (exact-move gap +0.05 at 4 layers to +0.43 at 9; p gap +0.17 to +0.49). Turns: the p
  gap stayed at +0.000 at every depth; from 9 layers turns came out whole sometimes, either way alike.
- So the hidden frames did not do what they were for: with about 70% of turns trained with the piece hidden, the
  button still told the model nothing about the direction. Whether they cost anything cannot be read from this
  chain (growth, the bot's turns and the hidden frames all changed at once).

## The plain chain (`world_model_tetris_s3_plain` ... `s12_plain`, 2026-10-15)

**Why:** the user: "masking is the only big change we made, maybe that's not helping"; "let's try from 3 with it
off and see where it goes". The hidden chain's schedule exactly, without `--hidden`: s3 from scratch for 16,000
steps, then a layer every 10,000 steps (grown, fresh optimizer and warmup each time) to 12. Each stage compares
with the hidden chain's stage at the same depth and total step.

**Predictions (before launch):**
1. If the hidden frames slowed the motion tier: s3_plain's shape passes 0.3 by 16k (s3_hidden: 0.08), and every
   stage's pressed-dream coherence beats the hidden stage at the same depth.
2. If they did not: s3_plain stalls the same way (shape under 0.15 at 16k), and the stages track the hidden ones.
3. The turn p gap stays near 0 at every depth either way (the hidden frames never moved it).

**s3_plain (2026-10-15):** without hidden frames 3 layers did not stall: shape 0.34 at 12k, 0.64 at 16k (s3_hidden
0.08), wrong +16 0.15% (1.8%); prediction 1 held. The 16k checkpoint is the comparison with s3_hidden. The user then
kept it training while it improved (to 26,054): the no-button clock came (spawn 0 at 16k to 0.74 at 24k) while
shape stayed at about 0.65 (actnorm 0.82). Shifts reached a +0.45 gap at 22.8k (s9_hidden: +0.43); turns stayed
blends with no direction (p gap +0.004). The user asked why the no-button scores led: the model is for play with
buttons, and judged on those, 3 layers had stopped improving at 16k (lost / morph per 100 frames 0.94 / 1.82 at
16k, 1.36-1.86 / 2.18-2.83 after; shape with buttons 0.14 -> 0.21; spawn with buttons flat at about 0.75). Since
then the plateau rule and the reports lead with the button dreams: lost, morph, shape and spawn with buttons and
the swap gaps; a depth grows when none of them has beaten its best by more than a test's swing (morph 0.2, lost
0.15, shape and spawn 0.05, gaps 0.05) for three tests (6k steps), 10k to 30k steps a depth. s4_plain grown from
s3_plain's 26,054, no step limit.

## Grown layers were near no-ops; copied layers instead (2026-10-15)

**The check** (scratchpad block_use.py: per block, on 8 live windows with one mask, what it adds to the residual
stream relative to the stream, and the masked loss with it skipped): every zero-start grown block stayed close to
a no-op. s4_hidden's new block, 10k steps after growing: adds 0.027, skipped 1.0x the loss; s9_hidden's six grown
blocks: 0.002-0.048, skipped 1.0-1.1x; s4_plain's at 1.2k: 0.003 (its output weights 0.1-0.2). The 3 original
blocks (at 0, 1 and last) did the work: skipped 1.2-1.4x, 3.2-4.9x and 6-15x. So the hidden chain's gains came
from the old blocks training on (and maybe the optimizer restarting at each growth), not from depth: the reading
above that each layer added a tier is wrong. s3_plain fits this: 3 layers without hidden frames reached shape 0.64.

**Copy start** (`deepen(..., new_at, copy=True)`): the new block is a copy of the block before it, and both have
their output projections halved, so the pair adds about what the one did and the new block trains from working
weights (the stacking used in growth papers, kept near the model's function). Copying, where the new block goes
comes down to which block is doubled. On s3_plain at 26,054 (the features' change and the loss at the growth):
block 1 0.17, 1.07x; block 2 0.18, 1.00x; block 3 0.25, 0.64x (the last block, which does most, in two half
steps is better at once).

**Run:** `s4copy_b1`, `s4copy_b2`, `s4copy_b3` on shared batches from s3_plain's 26,054 (the user: test where
the new layer goes; copy start). Four zero-start placements ran 89 steps before the switch and were dropped.

**Predictions (before launch):** each copied block keeps being used (adds over 0.1 of the stream and the loss
rises over 1.2x with it skipped, at 2k and 4k), unlike every zero-start one; on the button dreams the arm that
doubles the last block leads at 4k (it starts lowest), and all three beat s3_plain's best lost / morph with
buttons (0.94 / 1.64) by 6k.

**The copy arms to 6k, and what the blocks do (2026-10-15; scratchpad block_roles.py, heads_space.py,
button_blocks.py, coherence_breakdown.py):**
- The arms converged: masked loss 0.040 / 0.035 / 0.029 at 2k (blocks 1 / 2 / 3 doubled), 0.0367 / 0.0370 /
  0.0366 at 4k; with buttons at 6k 1.35 / 2.06, 1.48 / 2.09, 1.47 / 2.46 lost / morph per 100 frames. Prediction 3
  missed: none beat s3_plain's best (0.94 / 1.64). Without buttons all three break a third as often as s3_plain.
  The copies were used (each adds 0.07-0.30 of the stream) and drifted apart (7-9% of their weights at 2k, 15-19%
  at 4k).
- Roles (the loss with a block skipped, on pixels that stay and pixels that change): the first block does most of
  the motion (s3_plain: changed 4.7x, static 1.9x), the last is the readout that copies the board (static 25.6x).
  A doubled block's copies share its job; none took a new one. Temporal attention reaches back 4-5 frames in every
  block; spatial attention is close to uniform over the screen in every block and arm (mean distance about 8
  patches; the most local heads put 0.10-0.22 of their weight in the 3 x 3 patches around, uniform 0.035).
- **The buttons are read in the first block only.** With block 1's actnorm switched off the shift's direction is
  gone (p gap +0.58 -> +0.02) in every model but the one with block 1 doubled (there split between the copies);
  the later blocks' actnorm only sharpens the move (exact shifts 62-67% with 4 layers, 10-56% with one off). The
  buttons added at the input carry no direction alone. No block carries a turn's direction (p gap about +0.007).
- Breaks by situation (32 long dreams with the bot's buttons, per 100 quiet frames, lost / morph): within 2 frames
  of a turn 15-22 / 23-29, of a shift 8-9 / 14-15, no button 2-3 / 3-4. The fourth layer helped soft drops (morph
  3.0 -> 1.8-2.3), not turns or shifts.

## The buttons into the last block only (`world_model_tetris_s3_late`, 2026-10-15)

**Why:** every model reads the buttons once, in its first block, before any piece's shape exists: a direction
for the whole picture, which is a shift but cannot be a turn (a turn depends on the piece's orientation). The
user's option A, at 3 layers: the buttons enter only the last block's norms (`action_blocks`), not the input
tokens, so the model must read them where the piece's features are built. From scratch, s3_plain's settings
otherwise; s3_plain (every block, and the input) is the baseline at every 2k.

**Predictions (before launch):** copying and the board as s3_plain (they do not need the buttons); shifts slower
to come (the readout block must learn the move as well); the turn p gap above +0.05 by 16k, where every model so
far stayed under +0.03. If turns stay at 0 with the buttons in the last block, where they read a piece's features,
the next suspects are the near-global spatial attention (option B) and the turn pixels' small weight in the loss
(option C).

**To 30k (2026-10-15, still training):** the turn prediction failed: turn p gap +0.0004 at 16k, +0.002 at 30k
(s3_plain +0.002 at 16k). Shifts came at about half s3_plain's pace (p gap 10k / 16k / 30k: +0.08 / +0.23 / +0.42,
s3_plain +0.19 / +0.46 at 10k / 16k); with buttons shape trails s3_plain by 6-8k steps (late 0.163 at 30k, plain
0.167 at 20k), spawn by about 4k, and pieces fall 11-14% too fast (fall 1.11-1.14, s3_plain 1.01-1.03). No
buttons: level with s3_plain on shape (0.64 at 22-26k), flat since 22k. Buttons in the last block only cost
speed and bought nothing. The swap check's no-press variant never draws a turn (every model), so the tapped
press is read from the frame before; A and B stay apart where the buttons enter (distance A-B relative to
from-none: embedding 0.63, block modulations 0.29-0.34 in s3_plain 16k, against L-R 0.86 / 0.76-0.87): the
direction reaches the blocks and is not used, which points at the loss (option C) more than the lag or the path.

## Change-weighted loss, post-trained from s3_plain 16k (`world_model_tetris_s3_changew10`, 2026-10-15)

**Why:** the A/B direction reaches every block (above) and is not used, and a hedge between the two turns costs
the loss almost nothing: the cells that tell them apart are a few of the changed game pixels, which are 0.16% of
the masked pixels. Weighting changed game pixels 10x (`--change-weight 10`, Dynamics.loss_weights: game pixels
whose colour differs from the frame before's; the border is not weighted) gives the moves the loss's attention.
A target-dependent weight leans the trained probabilities to change by exactly the weight's odds, so every
decode (FrameDecoder: dreams, previews, scored rollouts, the swap check) takes the odds of each game pixel's
change back down by 1/10 (Dynamics.change_undo), which gives back the probabilities an unweighted loss trains
to; without that undo, the 2026-10-05 decode-time check (w = 2-3 on srr 20k) turned over-filled pixels into
ghosts. Never tried on a pixel model before: only that decode-time check, and the latent generators'
`--moved-weight 10` mixed with other changes.

**Setup:** s3_plain's 16k checkpoint (weights, optimizer, averaged weights, step) resumed in a new folder with
only the weight added; the same constant learning rate (3e-4). s3_plain's own 16k -> 26k is the control, step for
step. The logged total loss is the weighted mean (not comparable); game and changed losses are unweighted.

**Predictions (before launch):**
1. The turn p gap above +0.05 by 26k (s3_plain +0.002 at 16k, +0.004 at 22.8k); the turn gap (real-move share)
   above +0.05.
2. Shifts no worse: p gap at or above s3_plain's at the same steps (+0.46 at 16k, +0.55 at 22.8k).
3. With buttons: lost and morph per 100 frames at or below s3_plain's trend at the same steps, shape at or
   above it (0.17-0.20 at 18-26k).
4. Calibrated (the undo works): fall speed 0.95-1.05, no-button wrong +128 within 1.2x s3_plain's (2.2-2.7%),
   no rise in unsure pixels.
If the turn p gap is under +0.02 at 26k, the loss's weighting is not what keeps turns blended, and the next
suspect is the model's structure (local attention, option B).

**Result at 26k (stopped 2026-10-15):** prediction 1 failed: the turn p gap +0.003 / +0.004 / +0.005 / +0.006
/ +0.005 at 18 / 20 / 22 / 24 / 26k, the real-move gap +0.00 to +0.02 (s3_plain +0.002 at 16k, +0.004 at
22.8k). Prediction 2 held: shift p gap +0.45 -> +0.58 (s3_plain +0.46 at 16k, +0.55 at 22.8k). Prediction 3 split:
with buttons over 18-26k (level / slope per 2k) morph 2.09 / +0.13 against 2.51 / +0.09 and lost 1.38 / +0.06
against 1.55 / +0.09 (fewer breaks), but shape 0.167 / +0.004 against 0.183 / +0.015 and spawn 0.63 against
0.73 (26k: morph 2.50 / 2.58, lost 1.53 / 1.61, shape 0.177 / 0.206, spawn 0.75 / 0.75): it draws fewer of the
changes s3_plain gets wrong rather than drawing them right. Prediction 4 held: fall 1.04 (s3_plain 1.02),
wrong +128 3.6% / 3.3% with buttons, 2.6% / 2.3% without; no rise in unsure pixels; the undo works. No buttons:
shape 0.668 against 0.645, spawn 0.33 against 0.51.

Why turns stayed blended, most likely: the weight is per pixel and temporal, so in a turn it lands on the cells
the real turn fills (they change) and not on the cells only the wrong turn would fill (empty before and after):
half of what tells the two turns apart kept the old tiny weight. The next try weighs whole frames instead, the
ones a press reaches (their incoming button or the one before: the press shows a frame late), every pixel of
them, wrong turn's cells included; the buttons are inputs, so that weight leaves the trained probabilities as
they are and needs no undo.

## Frames a press reaches weighted, post-trained from s3_plain 16k (`world_model_tetris_s3_pressw10`, 2026-10-15)

**Why:** the per-pixel change weight never weighted the cells only a wrong turn fills (above). Here whole frames
weigh 10x (`--press-weight 10`, train_dynamics_ui.press_weights): the frames an A, B, Left or Right press
reaches (its incoming button or the frame before's), every pixel of them. 17% of training frames; with every
button it would be 45%, most of them held soft drops (Down 23%), which have no direction to learn. Weighted, they
are about two thirds of the loss. The weight depends only on the buttons, which the model sees, so the trained
probabilities stay as they are: no decode change. The pixel weight is off (kept for a later combination; the
user's call). Same setup as s3_changew10: s3_plain's 16k checkpoint resumed in a new folder, constant lr 3e-4,
s3_plain's 16k -> 26k as the control. The logged total loss is the weighted mean.

**Predictions (before launch):**
1. The turn p gap above +0.05 by 26k (s3_plain +0.002 / +0.004 at 16k / 22.8k; s3_changew10 +0.005 at 26k), the
   turn gap above +0.05.
2. Shifts no worse: p gap at or above s3_plain's (+0.46 at 16k, +0.55 at 22.8k).
3. With buttons: lost and morph at or below s3_plain's trend at the same steps, shape at or above it.
4. The cost, if any, on the frames left at weight 1 (a third of the loss now, from 83%): gravity and spawns.
   No-button spawn and wrong +128 within 1.2x s3_plain's, fall 0.95-1.05.
If the turn p gap is under +0.02 at 26k with whole turn frames weighted, the loss is not what keeps turns
blended, and the next suspect is the model's structure (local attention, option B).

**Result at 26k (stopped 2026-10-15):** prediction 1 failed by its own bar, though this is the only run whose
turn p gap rose steadily: +0.002 / +0.005 / +0.008 / +0.008 / +0.012 at 18-26k (about +0.0024 per 2k, three times
s3_plain's and s3_changew10's), turn gap +0.03 at 26k; under +0.02, so by the rule above the loss's weighting is
not the main thing that keeps turns blended, though it moves them. Prediction 2 held, the best of any run: shift p
gap +0.54 -> +0.65 (s3_plain +0.55 at 22.8k). Prediction 3 mostly held: with buttons over 18-26k lost 1.23 / +0.09
against 1.55 / +0.09 (21% fewer), morph 2.19 against 2.51, shape level (0.180 against 0.183), spawn level (0.72
against 0.73), board level (3.3%). Prediction 4, the cost: no-button spawn 0.35 against 0.51 over 18-26k, closing
(0.58 against 0.63 at 26k), board level (2.4% / 2.3%), fall 1.08 (s3_plain 1.08).

Next (the user, 2026-10-15): motion as a channel in the stream, VideoJAM-style (examples/papers): first the
motion itself, made from the RAM and checked against the pictures (data/tetris_motion.py,
scripts/motion_check.py), then the UI for it, then the border off as its own run (its fall timer and autorepeat
counter are hidden state the picture shows a frame later).

## Motion in and out, from scratch (`world_model_tetris_s3_motion`, 2026-10-15)

**Why:** VideoJAM (examples/papers/videojam_2502.02492.pdf): a pixel objective is nearly blind to motion; their
fix predicts motion with the pixels from one representation, and gives it back as input. In pixels a turn is a
flip whose A / B difference is a few cells of one frame; as motion it is the whole answer on those cells: each new
cell's source, which A and B give opposite ways. The motion is made from the RAM and checked against the pictures
(data/tetris_motion.py, scripts/motion_check.py: 13,871 of 13,877 piece moves match pixel for pixel, every turn;
line-clear drops from the pictures, 30 of 30): per 8 x 8 cell still / moved from (dy, dx) / new; UNKNOWN for a
window's first two frames. It is training data only: dreams make their own (the user: nothing but the model at
inference).

**What changes (s3_plain's recipe otherwise, from scratch, `--motion`):** each visible token gets its cells'
motion codes through a zero-initialised embedding (VideoJAM's added input rows); a head predicts every masked
token's cells' motion, cross-entropy on the picture's cells (not the border's rows), summed with the pixel loss
(VideoJAM: no weighting). The motion input is withheld (UNKNOWN) in 20% of windows (VideoJAM's 20% dropout) and
before a random cut in another 20% (a dream starts with none, from a screenshot, then has its own). Decoding:
each generated frame's own motion (the head's most likely code per cell) goes into the history with it; a dream's
start has none. Reported: the motion loss, the share of moved and of turned cells (moved along both axes: only a
turn's cells do) right on masked tokens, and the same for the previews' own dreamed frames.

**Predictions (before launch), against s3_plain at the same steps:**
1. Turns: the previews' turned cells right over 0.5 by 16k (their own frames, real context); the swap check's
   turn p gap over +0.05 by 16k (s3_plain +0.002), with the real context's motion as in training.
2. Copying and the board as s3_plain: no-button wrong +128 within 1.2x at the same steps.
3. With buttons: lost and morph at or below s3_plain's trend, shape at or above it.
4. Speed within 10% of s3_plain's 3.9 steps/s (a tiny head; the motion is made in the data workers).
If the turn p gap is under +0.02 at 16k while the dreamed turned cells are right, the motion is learned but not
used by the pixels: next, decode motion first and the pixels given it (VideoJAM's inner guidance, two passes).

**Restarted as `world_model_tetris_s3_motion2` at 8.8k (2026-10-15 21:00), from scratch.** Two changes (the user:
"we cannot see the input or the output motion vectors and the cover screen on loss probably needs motion
vectors"): (1) the top-out curtain now has motion: it covers one more row of the well every 4 frames, each new
row the row above it in the frame before, so it slides down a row (data/tetris_motion.py; it was NEW); (2) a dream's
drawn motion keeps the border's token rows STILL, as the data does (Dynamics.drawn_motion): the border is never
scored, so its logits were untrained, and the first run's dreams wrote junk border motion into their histories.
The previews now draw REAL MOTION and DRAWN MOTION (diagnostics/motion_pictures.py) and the Curves tab the motion
scores. The first run (`world_model_tetris_s3_motion`, kept) to 8k against s3_plain: pieces fell at real speed
from 6k (s3_plain 0.24 at 6k), half the pressed breaks (morph 0.37 against 0.67 at 6k), first spawns at 6k,
board drift 1.3x at 6k; shift p gap +0.010 / +0.052 / +0.186 at 4 / 6 / 8k (s3_plain +0.10 at 8.2k), turn p gap
0.000; turned cells right 0 / 0 / 0.05 masked at 4 / 6 / 8k, 0.22 in its own previews at 8k.

**Overnight (the user, 2026-10-15 ~23:00: "let it train overnight instead of just 16k"):** s3_motion2 has no step
limit and keeps training until stopped; the turn prediction is still judged at 16k. A report-only loop runs the
swap check at every 2k checkpoint from 16k (button_swap.csv). At 12k against s3_plain: shift p gap +0.39 / +0.31,
turn p gap 0.000 (turned cells right 0.19 masked, 0.11 in its own previews); no buttons shape 0.62 / 0.34, morph /
lost 0.28 / 0.16 against 0.68 / 0.43; with buttons shape 0.101 / 0.070, morph 1.35 / 1.64, spawn 0.21 / 0.39,
presence 0.58 / 0.79; board level (3.3-4.0% / 3.4-4.2%).

**The two motion runs at 6k on the same 16 trials (levels 0-19, scripts/long_dream_check.py --per-band 4, the
fixed decoder for both):** the first run still had more piece pixels, ghosts and breaks than motion2 (buttons:
wrong +128 9.5% / 8.0%, piece at +32 3.95 / 2.96, ghost 3.72 / 2.84; no buttons morph 0.19 / 0.03), so the runs
differed by 6k themselves, not by the border-motion decoding fix. Against s3_plain (3.8%, piece 0.32, ghost 0.22)
both motion models smeared pressed pieces along their motion at 6k (the user saw it in the previews: "smearing in
the direction of the motion ... a different learning pattern"), most at fast levels; on the training test's 32
trials it was milder (5.0% / 4.6%) and gone by 12-14k (shape above s3_plain's, board level).

**Overnight to 81k (2026-10-16 05:20, still training).**
- Prediction 1 (turns) failed at 16k: turn p gap 0.000, the previews' turned cells right 0.30 (band 16-24k).
  Overnight the pixel turn p gap crept to +0.015 (+0.005 at 20-30k, +0.012-0.018 at 54-80k; s3_plain +0.004 at
  22.8k; actnorm 72k +0.026 at 8 layers).
- The motion learned turns where the pixels did not: turned cells right 0.65-0.72 masked and about 0.5 in its own
  previews from 40k (0.71 at 80k). A swap test of the drawn motion (scratchpad motion_swap.py: the swap check's
  instances, one frame ahead with the real context's motion) shows how much of that is direction: turns, the real
  move's cells drawn 0.38 with the real press and 0.27 swapped, the opposite move's 0.22 and 0.35 (gap +0.12 at
  80k, +0.09 at 40k); shifts 0.98 / 0.00 against 0.00 / 0.98. The first turn-direction signal of any model, in
  motion only, and weak; the pixels, drawn in the same pass, take little of it.
- Shifts: p gap +0.52 at 16k (s3_plain +0.46), +0.64 at 26k, +0.72 at 70-80k.
- Long dreams with buttons (level / slope per 2k): 18-26k morph 2.05 / +0.12 against s3_plain 2.51 / +0.09, lost
  1.27 against 1.55, shape 0.194 against 0.183, spawn 0.65 against 0.73, presence 0.86 against 0.94, board 3.3%
  both. Then on its own: 56-80k morph 1.78, lost 1.10, shape 0.34, spawn 0.84, presence 0.95, board 2.8%, fall
  0.99 (actnorm, 8 layers 2.45M, on the same trials: 40k 1.31 / 0.87 / 0.45 / 0.96, 72k 0.78 / 0.48 / 0.55 /
  1.00). No buttons 56-80k: morph 0.42, lost 0.22, shape 0.71, spawn 0.79, board 2.1%.
- The 6k smear is gone: on the same 16 trials (levels 0-19, --per-band 4), with buttons at 24k motion2 / s3_plain:
  piece at +32 1.10 / 1.11, ghost 0.32 / 0.29, wrong +128 3.2% / 2.8%, morph 2.21 / 2.09, lost 1.29 / 1.17, shape
  0.34 / 0.27, spawn 0.64 / 0.82, full rows per 100 frames 11.5 / 5.1; motion2 at 80k: ghost 0.12, wrong +128
  2.0%, morph 1.68, lost 1.19, shape 0.41, spawn 1.00, full 0. No buttons at 24k level (shape 0.69 both).

## Motion first, then the pixels given it (`world_model_tetris_s3_motionfirst`, from s3_motion2, 2026-10-16)

**Why:** s3_motion2's drawn motion follows A vs B a little (motion swap gap +0.12 at 80k,
scripts/motion_swap_check.py) but its pixels, drawn in the same pass, almost not (p gap +0.015): the motion is an
output beside the pixels, not something they follow. VideoJAM's other half guides the video by its own motion. Here,
as two passes (`--motion-first`, Dynamics.motion_first): a generated frame's motion is drawn first, then its pixels
with that motion given on its tokens, then the frame goes into the history with it (3 forwards a frame, not 2:
dreams about 1.5x slower). Training gives half the frames' masked tokens their real motion (MOTION_FIRST_SHARE;
those tokens' motion is not scored), so the pixels learn to follow motion; the motion embedding now goes on any
token whose motion the caller gives (the trainer withholds the rest).

**Setup:** s3_motion2's last checkpoint (weights, optimizer, step) resumed in a new folder with `--motion-first`;
the same lr; s3_motion2's 56-80k is the reference (it plateaued there). A fork, to learn what the second pass adds
fast; a single-stage run from scratch follows if it works.

**Predictions (before launch):**
1. The pixels follow the motion: the turn p gap rises toward the drawn motion's own gap, above +0.08 within 10k
   (s3_motion2 +0.015); the motion's own swap gap stays near +0.12 (the second pass cannot make the motion better
   at turns, only make the pixels follow it).
2. Shifts no worse: p gap 0.72 or more.
3. With buttons: lost and morph at or below s3_motion2's 56-80k (1.10 / 1.78), shape at or above 0.34; board
   2.8% or less.
If the turn p gap stays near +0.015 while the motion's gap holds, the pixels do not follow given motion: check the
share given in training.

**First tests (2026-10-16 morning).** At 98k (642 steps after the switch) the dreams broke: no buttons presence
0.53, fall 0.10, shape 0.04 (s3_motion2 90-97k: 0.98 / 1.00 / 0.78); by 100k they were back (0.95 / 1.01, spawn
0.89) and at 102k the best of any small model with no buttons (wrong +128 1.7%, presence 0.98, fall 1.05, spawn
1.00, lost / morph 0.06 / 0.07 per 100 frames). The break was the model learning to follow given motion, not the
decoding: I first blamed the most likely motion (a drop on an unsure frame is never the most likely code) and
switched to sampling it with one number a frame, but sampled dreams at 98k broke the same (fall 0.15), and on the
same 16 dreams at 100k sampling did worse with buttons (wrong +128 2.2% against 1.5%, lost / morph 0.91 / 1.20
against 0.50 / 0.83, spawn 0.83 against 1.00; no buttons full rows 2.0 against 0, spawn 1.00 against 0.75), so
the most likely code stays (Dynamics.drawn_motion) and the sampling is gone. Gravity is not unsure here: the
border shows the fall timer. Drawn motion's turn gap (most likely): +0.19 at 98k. The swap rows for 100k and 102k
were first taken with sampled motion (pixel turn p gap +0.045 / +0.012, real-move gap +0.02 / +0.07; motion +0.12 /
+0.10) and are taken again with the most likely; the later rows are the ones that count.

**What holds turns back now (104k, scratchpad follow_motion.py and motion_coherence.py, the swap check's 150
turns).** Pass 2 follows the motion it is given: with the real press and the real turn's motion it draws the real
turn 81% (p(real) where the turns differ 0.91), given the opposite turn's motion the opposite 73% (0.36), given
none 21% / 17% / 60% mix, given its own drawn motion 21% / 21% / 57% mix: its own motion helps no more than none.
The drawn motion is the problem: per turn its cells are exactly the real turn's 7% (swapped 1%), exactly the
opposite's 1% (9%), some of each 46% (51%), part of the real 37% (16%), part of the opposite 8% (21%). It leans
the right way with the press, but draws each cell on its own, so a turn's field is half one way, half the other,
and the pixels given it mix. Drawn-motion turn gap +0.25 / +0.22 / +0.20 at 100 / 102 / 104k (s3_motion2 +0.12 at
80k); pixel turn p gap +0.033 / +0.017 / +0.027, real-move gap +0.05 / +0.06 / +0.03. Dreams over 100-104k against
s3_motion2's 84-96k: with buttons morph / lost 1.07 / 0.81 against 1.73 / 1.08, shape 0.34 / 0.39, spawn 0.88 /
0.94, wrong +128 2.8% / 2.6%; no buttons morph / lost 0.13 / 0.10 against 0.40 / 0.24.

## Motion in two steps, MaskGIT by cell (`world_model_tetris_s3_motion_mg`, from s3_motionfirst, 2026-10-16)

**Why:** s3_motionfirst's pixels follow clean motion (81% / 73%) but its drawn turn fields mix the two turns (46%),
each cell drawn on its own. MaskGIT's confident-first decoding, by motion cell (the user: "let's try the maskgit"):
step 1 predicts every cell's motion and commits the ones at least 0.9 sure (the still background and stack) plus
the single most confident other cell, usually one of the turning piece's; step 2 draws the rest given those, so
the piece's other cells follow the committed one's turn; then the pixels are drawn given the whole motion, and the
frame goes into the history (4 forwards a frame, from 3). Only the motion is iterated: the pixels stay hidden
until the last pass (given clean motion they already draw the turn). By cell, not token: a token holds 4 cells and
a turning piece's cells mostly share tokens, and committing the most confident half of the tokens, as MaskGIT does,
would commit only background. Training: of the masked frames given motion (MOTION_FIRST_SHARE, half), half get all
their cells' motion (the pixel pass) and half a random share of their cells (the second motion step); the motion
loss scores every masked cell not given. The motion-first decoding is now always two steps (Dynamics.
committed_motion); s3_motionfirst's checkpoints never saw partial motion, so they are tested only up to the fork.

**Setup:** s3_motionfirst's last checkpoint resumed in a new folder, same lr; s3_motionfirst's 100-108k is the
reference.

**Predictions (before launch):**
1. Coherent turn fields: drawn motion's "some of each" from 46% to under 15%, exactly the real turn with the real
   press over 40% (now 7%).
2. Pixels: turn real-move gap above +0.15 and p gap above +0.08 within 10k (s3_motionfirst +0.03-0.06 /
   +0.02-0.03).
3. Dreams no worse than s3_motionfirst's 100-104k (with buttons morph / lost 1.07 / 0.81, wrong +128 2.8%).
If the fields come out whole but turn the wrong way a third of the time, the pixel gap stays near the motion's
own lean (+0.2): the motion's direction is then the next thing to improve.

**At 122k (10k after the fork): predictions 1 and 2 failed, 3 held.** Turn fields some of each 25 / 31 / 35 / 35 /
39% at 114-122k (under 15% predicted), exactly the real turn 2-4% (over 40% predicted); pixel turn p gap +0.022 /
+0.031 / +0.047 / +0.047 / +0.013, real-move gap +0.02-0.06; drawn-motion turn gap +0.14-0.19. Dreams better than
s3_motionfirst's 104-112k: with buttons morph / lost 0.85 / 0.69 against 1.05 / 0.83, shape 0.37 against 0.33; no
buttons 0.08 / 0.08 against 0.12 / 0.12; board level (2.8% / 1.8%).
Why (scratchpad commit_where.py, step2_follows.py, 120 turns at 122k): (1) the committed unsure cell is rarely a
turn's: elsewhere in the frame 46%, on the piece but not a turn code (a vacated cell's NEW, the centre) 36%, the
real turn's code 14%, the opposite's 2%; step 1 leaves about 5 unsure cells. (2) Step 2 hardly follows a committed
turn cell: with one real turn cell given (and step 1's sure cells), the piece's other moving cells are drawn as the
real turn 42% / the opposite 20%; with one opposite turn cell given, 35% / 23%. The cells barely condition on each
other: the spatial attention measured 2026-10-15 is near-global in every block (the most local heads 0.10-0.22 of
their weight on the 3 x 3 patches around, uniform 0.035), so a committed cell's pull on its neighbours is spread
over the screen. A better commit rule would not make the fields whole.

## Everything so far plus a local spatial bias, from scratch (`world_model_tetris_s3_motion_local`, 2026-10-16)

**Why:** the two-step motion could not make a turn's cells agree because they barely condition on each other, and
every model's spatial attention is near-global. The user: "lets try a, and try restarting with all the changes we
have made so far". The bias (`--local-attention`, SpatialAttention.local_bias): per block and head a learned
logit per (dy, dx) offset between tokens, added to the spatial attention (Swin's relative position bias), from
zero, so a head can make itself as local as it needs. The recipe: s3_plain's (96 x 3, rotary time and space,
QK-norm, actnorm, lr 3e-4) with motion in and out (the curtain's motion, the border STILL), motion first in two steps
(committed_motion; MOTION_FIRST_SHARE, cells given all or part), and the local bias; not the pixel change weight
nor the press weight (separate experiments: no help on turns; the press weight cost the clock). From scratch, one
stage, until stopped. Compared at the same steps with s3_motion2 (motion one pass, from scratch) and s3_plain.

**Predictions (before launch):**
1. The bias is used: the most local head of each block puts over half its weight on the 3 x 3 tokens around by
   10k (every model so far 0.10-0.22).
2. Turns: drawn turn fields mostly whole, some of each under 20% by 20k (the two-step fork 35-47%); pixel turn p
   gap above +0.05 by 20k and +0.1 by 40k (s3_motion2 +0.005 at 20k, +0.015 at 60-80k).
3. With buttons at least s3_motion2's long dreams at the same steps; no-button dreams too.
4. Speed within 15% of s3_motion2's (a float attention bias takes SDPA off its fastest kernel).
If the heads go local and the turn fields stay mixed, locality is not what the turn needs.

**At 24k: predictions 1 and 2 failed, 3 split.** (1) The heads stay global: at 20k the most local head of each
block puts 0.13 / 0.21 / 0.09 of its weight on the 3 x 3 tokens around (mean distance 6.8-9.0 tokens). (2) Turns
are not drawn at all yet: of the drawn turn fields 83-99% are "neither" at 12-24k (some of each 0-1%, so the 20%
bound holds only vacuously); pixel turn p gap -0.004 to +0.003 (s3_motion2 +0.004 to +0.006 at 18-24k); motion
turn gap +0.00-0.01. Shift is the best of any run from 14k: p gap +0.51 / +0.57 / +0.62 / +0.65 / +0.67 / +0.67 at
14-24k against s3_motion2's +0.47-0.63 and s3_plain's +0.41-0.55. (3) With buttons, 20-24k means (slope per 2k):
morph / lost 0.22 (-0.00) / 0.20 (+0.00) against s3_motion2's 2.14 (+0.04) / 1.31 (+0.13) and s3_plain's 2.51 /
1.59; per unit of presence about 0.29 against 2.5 and 2.7. But it has fewer pieces: presence 0.74 (+0.04) against
0.86 / 0.93, spawn on time 0.32 (+0.11) against 0.66 / 0.73, fall 0.86 against 0.97 / 1.02. Board level (wrong
+128 3.6% against 3.4 / 3.2%). No buttons: the same picture (morph / lost 0.03 / 0.03 against 0.54 / 0.38; fall
0.63 against 0.84 / 1.08). Its pieces are whole and its clock is late; spawn and presence still climb while the
other two runs' are flat.

**At 56k (with the event checks, 2026-10-16):** the breaks levelled at 0.5-0.7 from 30k (with buttons, morph /
lost 0.66 / 0.50 at 52-56k against s3_motion2's 2.07 / 1.30, and s3_motion_mg's 0.77 / 0.61 at 128-134k);
presence, spawn and fall level with s3_motion2 from 30k; no buttons the best of any run (0.04 / 0.03). Turns
start in both runs at the same rate: drawn-motion turn gap +0.05 at 38k (s3_motion2 +0.05 at 34 and 38k), pixel
turn p gap +0.009 (+0.003); with A and B swapped neither flips (local's turn fields part of the real / of the
opposite 18 / 21%, s3_motion2's 31 / 32%). Heads still global at 30k. Event checks (8 events each, no context
motion, the browser's start): play at 10k lost / morph 0.16 / 0.26 against 0.78-1.46 / 1.2-2.45; at 20k the only
run whose level menu beats copying the frame (0.351 wrong, copying 0.398), the best line-clear cells (0.073);
the curtain not yet learned by either motion run (0.18, the others 0.12-0.13).

Turns at 40-56k against s3_motion2 at the same steps: pixel turn p gap +0.019 / +0.013 / +0.020 / +0.019 /
+0.014 at 40 / 44 / 48 / 52 / 56k (s3_motion2 +0.005 / +0.005 / +0.007 / +0.008 / +0.013); drawn-motion turn gap
+0.04 / +0.05 / +0.05 / +0.07 / +0.06 (s3_motion2 at 44-56k +0.08 / +0.09 / +0.11 / +0.10); drawn turn fields
some of each 12 -> 28% as more turns are drawn (neither 41 -> 24%; s3_motion2 19-25%, neither 7-16%). With A
and B swapped, s3_motion2's fields lean a little the swapped way (part of the opposite 32-35% over part of the
real 27-31%), local's barely (23 / 20% at 56k). Heads still global at 40k.

## Turns from the button: press frames hidden, counterfactual twins (`world_model_tetris_s3_branch`, from s3_motion_local 58k, 2026-10-16)

**Why:** the model reads a turn's direction off what it can see of the frame and off the history, which
predicts the bot's choice, not off A vs B: teacher-forced it gets turned cells right on some batches (up to
0.89 at 56k) while the swap tests barely move. The user: "This has to be it training to find the lower loss stuff
and then getting stuck ... we need to figure out how to make our post training step have good loss for buttons
turning in the correct direction"; option 1 and 2 together. (1) `--hide-presses`: the frames an A, B, Left or
Right press reaches are hidden entirely whenever predicted (training_mask hide, pressed_frames), so the turn is
learned from the button, as a dream draws it; their motion is withheld but for motion_first's told share (a
dream's second and third passes). s3_hidden hid half of every frame from scratch and stalled; this is on a
trained base and only at presses. (2) `--branches 8`: 8 of the 16 workers give each window a counterfactual twin
(world_nes_tetris.TurnBranches): the emulator's saved state just before one of the window's turn press edges in
play (position 32-59), replayed with the other turn button (as long as the real one was held) or none, the other
buttons as played. The twin's frames up to the press are the window's own; only the button explains what
differs. Every batch is the 16 windows plus 8 twins (24 rows). Measured on the stream: 40% of windows have a
press to branch (the others reuse the worker's last twin), 58% of twins change the picture (O pieces and blocked
turns do not), a branching worker makes 4.0 windows a second against 9.8 (the trainer needs about 2.5).
Everything else as s3_motion_local (the local bias still weight-decayed: one change at a time); its continuation
on the same GPU is the control. Steps count on from 58,000.

**Predictions (before launch):**
1. Drawn-motion turn gap above +0.15 by 64k and +0.25 by 70k (local +0.05 at 38k).
2. Pixel turn p gap above +0.03 by 64k and +0.08 by 70k (local +0.009 at 38k).
3. With A and B swapped the turn fields flip: part of the opposite above part of the real (local 21 / 18%).
4. Dreams with buttons no worse than the control's at the same steps (breaks, presence, spawn within 0.05).
If the motion gap grows and the pixels' does not, the pixel pass does not follow its motion on the hidden
frames; if neither moves, 8 twins are too few against the history's lean (next: every worker branching).

**s3_branch stopped at 66,000 (2026-10-16 21:25)** with the control at 62,200: four GPU processes (both trainers and
two checks) filled the 24 GB card and thrashed over PCIe from 20:06; all were killed. Its 60k / 62k turn gaps were at or
below the control's (pixel +0.019 / -0.006 against +0.035 / +0.036, motion +0.05 / +0.04 against +0.07 / +0.07);
64k and 66k were not tested. From now on one GPU process at a time: a control trains in the same process.

## A critic that judges the turn with its button (`critic_ab`: s3 and s3_critic from s3_motion_local 62.2k, 2026-10-16)

**Why:** docs/guides/turn_direction.md (the user: try all three of the GAN, focal loss and a paired loss, the
GAN first). A per-pixel loss trains a model that ignores the button toward a hedge, and the hedge costs almost
nothing; a critic can make a blend, and a turn the wrong way for its button, cost what it should.
models/turn_critic.py: conditional and matching-aware (GAN-CLS): it sees the frame before a J / L / T turn and the
turned frame, 48 x 48 pixels around the piece, with the turn button. Real: the real frame, real button; fake: the
model's frame (hard colours, Gumbel straight through) with the real button, and the real frame with A and B
swapped. Hinge loss, spectral norm, a button projection; Adam 1e-4 (0.5, 0.9). The judged frames are hidden whole
when predicted and get none of their motion (a dream's view); the model's pull is the critic's score per judged
pixel in the pixel loss's units (`s3_critic`'s critic weight 1.0). In one process (memory: one-gpu-process): two
models on the same batches, `s3` (the control, s3_motion_local continued) and `s3_critic`, both from s3_motion_local's
62,200 (`--output-root output\critic_ab`, archive D:\token_world_checkpoints\critic_ab).

**Predictions (before launch):**
1. The critic tells the direction: real beats swapped (`critic_direction`) above 0.8 within 2k steps (the real
   pair and the swapped pair differ only there, and the critic sees the real data).
2. The model follows: pixel turn p gap above +0.08 and drawn-motion turn gap above +0.15 by 70k (the control's
   pixel +0.036, motion +0.07 at 62k); with A and B swapped the turn fields flip (part of the opposite above part
   of the real).
3. No grain: with buttons the dreams' breaks no worse than the control's at the same steps, and the previews'
   static pixels as the control's (the 10-01 critic's failure).
If the critic tells the direction and the model's gap does not open, the pull is too weak (raise the weight); if
the previews grain, it is too strong.

**At 70k (8k after the fork; swap checks on the CPU, the same 200 / 150 turns for both arms): the critic taught the
drawn motion the direction, the pixels hardly.**
- Drawn-motion turn gap +0.19 / +0.18 / +0.19 at 66 / 68 / 70k against the control's +0.06 / +0.07 / +0.06
  (prediction 2's motion part held: above +0.15). With A and B swapped its motion flips: the real turn's cells
  0.24, the opposite's 0.38-0.39 (control 0.22-0.25 / 0.22-0.26; prediction 3 held, in motion). With the real
  press 0.41-0.43 / 0.17-0.23 (control 0.28-0.31 / 0.20-0.21).
- Pixel turn p gap +0.020 / +0.026 / +0.017 against the control's +0.007 / -0.007 / -0.010 (prediction 2's pixel
  part failed: under +0.08); real-move gap -0.02 to +0.04 both.
- The critic tells the direction 0.69-0.87 (1k means; prediction 1 about held).
- Dreams with buttons, 64-70k (mean, slope per 2k): morph 0.72 (+0.05) against 0.60 (-0.05), lost 0.53 (+0.02)
  against 0.40 (-0.01), shape 0.33 (-0.02) against 0.39 (0.00), hit 0.77 (-0.04) against 0.82, spawn 0.80 against
  0.87 (the control's falling); board level (3%), no grain. No buttons level (morph / lost 0.02-0.03). So far it
  breaks a little more with buttons, and rising (prediction 3's "no worse" failing).
The motion drawn first now leans the right way about as much as s3_motionfirst's did (+0.20-0.25), and as there
the pixels drawn given it do not follow its direction: the turn-field check (which cells come out each way)
would tell whether its fields are whole or mixed.

**critic_ab stopped at 112,850 (2026-10-17 09:30).** The critic ran away from about 80k (its loss 0.03-0.07, the
model's frames scored about -8, the model's loss 3-5x the control's) and its button dreams broke more than the
control's (104-110k morph / lost 0.94 / 0.72 against 0.67 / 0.47). The control plateaued by the growth rule (no
button score past its best by more than a test's swing after 64k) and was not grown: a miss. Against the bests
on the same trials, single tests: control 72k morph / lost 0.70 / 0.50, shape 0.38, spawn 0.79, full rows 1.91
per 100 frames; actnorm 72k (8 layers) 0.78 / 0.48, 0.55, 1.00, 0.32; resync 18k 0.60 / 0.47, 0.54, 1.00, 0.53.
Level on breaks, behind on shape, spawns and clears; the 8-layer models are still the best.

**The critic's turn tests to 110k (CPU, the same turns for both arms), found after the stop:** the drawn motion's
direction kept growing while the critic ran away. Motion turn gap +0.33 / +0.30 / +0.41 at 86 / 98 / 110k
(control +0.08 / +0.12 / +0.10); with the real press the opposite turn's cells fell to 0.05-0.12, and with A and B
swapped the field flips cleanly (real 0.06-0.13, opposite 0.39-0.46; control 0.25-0.32 / 0.32-0.36). Pixel turn
p gap +0.019 / +0.048 / +0.071 (control -0.008 / +0.029 / +0.049): the critic arm's 110k is the largest pixel turn
gap of any run (s3_motion_mg +0.047, actnorm 72k +0.026), though the control rose too. Its button dreams broke
more all the while (above).

**Event checks, critic arm / control at 70k and 110k (CPU; the same 8 events per check as the 10k / 20k checks
above):** restart menu wrong 0.25 / 0.26 and 0.23 / 0.23 (copying 0.40; the 20k runs 0.35-0.44); back in a game
+8 0.13 / 0.18 and **0.27** / 0.13 (copying 0.08); curtain 0.13-0.16 both. Line clears: full rows left 0.25 /
0.13 and 0.13 / 0.25, cells wrong 0.064 / 0.046 and 0.039 / 0.058 (the 20k runs 0.38-0.75, 0.073-0.088). Play
(240 frames, real buttons): lost / morph 0.52 / 0.78 against 0.31 / 0.52 at 70k, **0.89 / 1.20** against 0.31 /
0.37 at 110k; shape 0.24 against 0.44 at 110k. The control kept improving its events; the critic arm got worse
at play and at starting a new game.

## The control grown a layer (`world_model_tetris_s4_motion_local`, from critic_ab's s3 at 112,850, 2026-10-17)

**Why:** the growth rule (above, the plain chain): the s3 motion-local line plateaued from 70k. `s4copy_b1`:
block 0 (the one that reads the buttons and does most of the motion) doubled, both copies' outputs halved (copy
start: zero-start blocks stayed no-ops); a fresh optimizer and warmup; steps from 0; the same flags (motion,
motion first, local bias, rotary, QK-norm, actnorm). One GPU process.

**Predictions (before launch):** by 10k, with buttons, a new best past the swing on shape (above 0.57) or morph
(under 0.40), and the turn gaps no worse than the control's (motion +0.06, pixel about 0); at the plateau, grow
again (the chain toward actnorm's 8 layers), judged against actnorm 72k and resync 18k on the same trials.

**The 4-layer growth was stopped at about 4k (2026-10-17 09:54), unjudged:** it grew one layer, copied the first
block and halved both copies' outputs, reset the optimizer with a warmup and grew at a plateau, five departures from
the published recipes (docs/guides/turn_direction.md has the papers). The user: grow an early checkpoint 2x or 3x
with the standard recipe.

## The control line stacked 3x (`world_model_tetris_s9_stack`, from s3_motion_local 48k, 2026-10-17)

**Why:** the user's choice after reading the growth papers: a checkpoint before the plateau, grown 3x by the
standard recipe. G_stack (Du et al., NeurIPS 2024): the whole model stacked in order, [b0 b1 b2] x 3, first and
last blocks included (their appendix H: whole >> middle ~ back >> front; in order beats each block twice in
place); plain copies, nothing zeroed or halved. Each copy gets its source block's AdamW state ("Growing to See",
2610.16817; Bu 2511.04981 found reset and copy alike) and the learning rate (3e-4) and step count carry on with no
new warmup (Bu: the same learning rate across the expansion). The trained weights are copied, not the averaged
ones (they go with the optimizer's state); the averaged weights restart from them. 48k: breaks flat since about
32k but shape and spawn still rising (48-56k with buttons: morph 0.65, lost 0.47, shape 0.34, spawn 0.72). 9 layers,
2.73M parameters (actnorm 2.45M); the flags of the line (motion, motion first, local bias, rotary, QK-norm,
actnorm). Steps count on from 48,000. One GPU process.

**Predictions (before launch):**
1. Within 10k (by 58k), with buttons, past the control line's best by more than a test's swing on shape (above
   0.57) or spawn (above 0.96), and breaks no worse (morph 0.6-0.7, lost 0.4-0.5).
2. By 72k (24k after the growth) level with or past actnorm 72k on the same trials (morph / lost 0.78 / 0.48,
   shape 0.55, spawn 1.00) and toward resync 18k (0.60 / 0.47, 0.54, 1.00), compared checkpoint to checkpoint.
3. A loss spike at the growth that recovers within about 1k steps (stacking breaks function preservation).
4. Turns: the motion gap no worse than the control's (about +0.06).

**At 60k (12k after the stacking), against the s3 line at the same steps and the 8-layer bests on the same trials,
checkpoint to checkpoint.** With buttons 56 / 58 / 60k: morph 0.44 / 0.54 / 0.53 (s3 line 0.73 / 0.69 / 0.66),
lost 0.32 / 0.41 / 0.37 (0.52 / 0.48 / 0.56), shape 0.42 / 0.42 / 0.44 (0.34 / 0.32 / 0.32), spawn 0.93 / 0.89 /
0.93 (0.82 / 0.79 / 0.75); fewer breaks than actnorm 72k (0.78 / 0.48) and resync 18k (0.60 / 0.47), behind them
on shape (0.55 / 0.54) and spawn (1.00); full rows that never clear 4-9 per 100 frames (actnorm 0.32, resync 0.53),
the s3 line's weakness. No buttons: shape 0.80, the best of any run. Prediction 1 failed (no shape past 0.57 or
spawn past 0.96 by 58k). Turns (CPU, 200 / 150 turns): pixel turn p gap +0.065, real-move gap +0.05 (the s3 line
+0.035 at 60k, actnorm 72k +0.026); drawn-motion turn gap +0.14 (the line +0.07 at 60-62k); with A and B swapped
the field leans the swapped way (real 0.29, opposite 0.40). actnorm 72k ends a cooldown; s9 has had none.

**To 102k (54k after the stacking), the best model so far on the button dreams.** 6k windows with buttons (morph /
lost / shape / spawn): 50-56k 0.60 / 0.42 / 0.36 / 0.84, 62-68k 0.49 / 0.34 / 0.51 / 0.96, 74-80k 0.41 / 0.29 / 0.54
/ 0.96; tests at 98-102k 0.35-0.38 / 0.26-0.27 / 0.61-0.65 / 0.96-1.00, full rows 0.2-2.0. At 102k against actnorm
72k and resync 18k, checkpoint to checkpoint: morph 0.35 against 0.78 / 0.60, lost 0.26 against 0.48 / 0.47, shape
0.65 against 0.55 / 0.54, spawn 1.00 all, full rows 0.21 against 0.32 / 0.53; no buttons shape 0.83-0.87 (the best
of any run). Not yet cooled (actnorm 72k ends a cooldown). The growth rule's per-test reading called this a plateau
from 68k (no score past its best by more than a swing) while the window trend climbed steadily: the rule is
retired (the user agreed), plateaus are judged by window trends. Training continues.

**Turns learned (CPU swap checks, the same 200 / 150 turns as every run):** pixel turn p gap +0.065 / +0.124 /
**+0.532** at 60 / 80 / 102k, real-move gap +0.05 / +0.15 / **+0.73**; drawn-motion turn gap +0.14 / +0.37 / **+0.89**:
at 102k with the real press the real turn's cells 0.92 and the opposite's 0.04, with A and B swapped 0.02 and 0.90.
The first model whose turns follow A vs B, as its shifts do (shift p gap about +0.7 in every model since actnorm).
Every earlier run stayed at +0.00 to +0.07 (actnorm 72k +0.026; the critic arm +0.071 at 110k).

**Event checks at 102k against actnorm 72k and resync 18k (CPU, the same 8 events per check):** restarts: menu
wrong 0.019 against 0.127 / 0.074 (copying 0.398), back in a game +8 0.015 against 0.038 / 0.019 (copying 0.081),
curtain 0.039 against 0.043 / 0.037. Line clears (8 singles, to +60): no full row left for any model; cells wrong
0.019 against 0.012 / 0.016; mass 0.996 against 1.00. Play (240 frames, real buttons): lost / morph per 100 frames
0.10 / 0.21 against 0.47 / 0.99 and 0.21 / 0.47; shape kept 0.82 against 0.65 / 0.53. The best of the three on
restarts and play, about level on single clears. (CPU dreams: s9 473 ms a frame, actnorm 142: four passes of nine
layers, so the CPU checks take about an hour for three models.)

**The training data's own event scores (built 2026-10-17, from the next launch on).** The user: "why are we not
gathering these stats just on the fucking training data?" The stream is fresh every step, so every batch is
unseen. The workers label each picture with the events it first shows (tetris_events.frame_events, the scenario
checks' rules and alignment); every 8th step the trainer hides one event frame per window whole (the rarest event
present), the real past visible, and scores it as the scenario checks do (exact, wrong, false change) with one
no-gradient forward, plus one with A and B swapped on the J / L / T turns (the turn gap): metrics.csv's live_
columns, summed between previews (train_dynamics_ui.live_event_scores). The 2k test keeps the long dreams, the
errors that compound on the model's own frames. Plan (the user): s9_stack trains overnight untouched; in the
morning a checkpoint is picked for its cooldown.

**Overnight to 156k (2026-10-18 morning).** Button dreams, window means of 5 tests (136k / 146k / 156k): shape
0.675 / 0.682 / 0.650, lost 0.209 / 0.171 / 0.182, morph 0.263 / 0.243 / 0.225, full 0.075 / 0.036 / 0.953 per
100 frames (single tests 150-156k: 0.30, 1.11, 1.26, 1.83), spawn 0.99-1.00, pixels wrong +128 2.0%; against
actnorm 72k 0.55, 0.48, 0.78, 0.32 and resync 18k 0.54, 0.47, 0.60, 0.53. No-button dreams flat for 20k (shape
0.84, lost / morph about 0.02, spawn 1.00). Live scores 116k-156k: turns exact 0.35 -> 0.41 with the real press,
0.06 with A and B swapped (gap +0.29 -> +0.36, 2,863 turns); gravity 96%, soft drop 98%, shift 88%, spawn 81% ->
86%, rotate 46% -> 51% exact; line clears the weakest tier and not moving (single 15%, double 3%, triple, tetris
and level up 0%; game start 19%).

## s9 cooldown from 140k (`world_model_tetris_s9_cool140`, 2026-10-18)

**The pick (the user):** 140k, the middle of the best button window (136-144k: shape about 0.68, lost about 0.16,
morph about 0.22) and before full rows climbed (its test: shape 0.679, lost 0.141, morph 0.197, full 0.09).

**Resync first? Not now.** The user asked whether resync rollouts should come before the cooldown. The trainer has
no rollouts for motion models (`rollout_losses` refuses them), so it would be new code; and the order we and the
papers use is the other way: actnorm 72k ended a cooldown, then resync fine-tuned it at 3e-5 (Self Forcing,
2025, likewise post-trains a converged model). Resync's gain there was small (morph 0.78 -> 0.60, lost level, full
0.32 -> 0.53), and s9 at 140k is already past it on all three. A resync fine-tune of the cooled model stays an
option once rollouts handle motion.

**Run:** model_step140000.pt (weights, optimizer, EMA, step) as the new folder's model_latest.pt; the same flags
as s9_stack plus `--cooldown 140000 12000` (linear from 3e-4 to zero by 152k, the length of every earlier
cooldown); s9_stack stopped at its latest checkpoint, resumable.

**Predictions (written before launch):**
1. No-button dreams stay where they are (shape about 0.84, lost / morph under 0.03, spawn 1.00).
2. With buttons, lost and morph fall 10-30% from 140k's window (to about 0.12-0.15 and 0.16-0.20) and full rows
   stay under 0.3; shape within 0.03 of 0.68. Earlier cooldowns ranged from worse (rope) to clearly better
   (layered), so a cooldown that changes nothing is possible too.
3. Live scores: rotate and spawn exact up a few points, the turn gap at least +0.36; line clears do not move
   (a schedule change cannot teach a tier the model has not started).

**Result (finished 10:33, 2026-10-18): the cooldown helped, against s9_stack at the same steps.** Button dreams,
the mean of the six tests 142-152k (cooled / not): lost 0.129 / 0.205 (-37%), morph 0.199 / 0.253 (-21%), shape
0.668 / 0.663, full 0.29 / 0.30, spawn 0.97 / 0.99, pixels wrong +128 1.9% / 2.0%; at 152k lost 0.103, morph
0.153, shape 0.677, full 0.21. No-button dreams level (shape 0.83, lost / morph 0.005 against 0.017 / 0.014).
Against actnorm 72k and resync 18k (lost 0.48 / 0.47, morph 0.78 / 0.60, shape 0.55 / 0.54): the best model on
all three. Live scores 140-152k (cooled / not): rotate 0.64 / 0.50, spawn 0.93 / 0.87, shift 0.90 / 0.88, single
clears 0.19 / 0.14, doubles 0.09 / 0.04; turns exact 0.54 / 0.41 with the real press and 0.06 with A and B swapped
(gap +0.49 against +0.35). Predictions: 1 held; 2 held for lost (-37%, beyond the 10-30% guessed) and morph
(-21%), full rows not under 0.3 at every test (1.05 at 148k); 3 held for rotate and spawn and the turn gap, and
line clears moved a little (singles 14% -> 19%), against the guess that they would not.

## Resync rollouts for s9, against its cooldown (`world_model_tetris_s9_resync140`, planned 2026-10-18)

**The plan (the user):** let the cooldown finish; then resync from the same uncooled 140k, without a cooldown, to
152k; if it beats the cooled 152k, cool it down too. Judged against s9_stack at the same steps (the user, 12:45: comparing an uncooled
run with a cooled one is not interesting): same start, same rate, the rollouts the only difference; if it
beats s9_stack, its own cooldown is compared with cool140. Last time resync started from a cooled model (actnorm 72k)
and ran at a flat 3e-5, so its small gain could not be told apart from more training at a low rate.

**The comparison at 152k, three ways from the same 140k:** cool140 (cooldown, no rollouts), s9_stack itself
(already trained 140k-156k at 3e-4 with no rollouts: the control for "more steps at this rate") and resync140
(3e-4, rollouts with resync). Resync runs s9's own schedule (lr 3e-4, weight decay 0.05, no warmup: the step
carries on), not the old fine-tune's 3e-5, so s9_stack is its exact control.

**Built (2026-10-18): rollouts for motion models.** The trainer refused them. Now `rollout` passes the window's
motion to the decoder and returns the motion each dreamed frame went into the cache with; with resync the
replaced patches take the real motion as well as the real pixels (FrameDecoder target_motion), so pixels and
motion agree. train_step gives the dreamed history that drawn motion (what a dream sees) and the frames from the
boundary on their real motion, masked and scored as in teacher forcing. Tests: test_dynamics
`test_a_resynced_motion_rollout_gives_the_replaced_patches_their_real_motion`,
`test_a_motion_model_trains_on_resynced_rollouts_with_the_motion_they_were_drawn_with`.

**Run:** model_step140000.pt as the new folder's model_latest.pt; s9's flags plus `--rollouts 16 16 1 --resync 8
--refresh 2000 --steps 152000`. Memory checked in the first minutes (s9 teacher-forced 18.5 GB; the drawer's
cache adds a few GB): over 23.5 GB, stop and shrink before going on.

**Predictions (written before launch):**
1. Speed 0.8-1.0 steps/s (four forwards per dreamed frame with motion first), about 12k steps in 3.5-4 h.
2. Button dreams at 152k against s9_stack's 152k: morph 10-25% lower (resync took actnorm's 0.78 to 0.60), lost
   level, full rows no better (resync raised them, 0.32 to 0.53); no-button dreams level.
3. Against the cooled 152k: not better without its own cooldown, unless rollouts matter more than the schedule;
   the cooldown of resync decides.
4. The live turn gap holds (+0.36 or more): the rollouts do not touch how a press is read.

**Result (152k, finished 14:33): small gains on breaks, losses elsewhere; not better than s9 overall.** Against
s9_stack at the same steps, the mean of the six tests 142-152k with buttons (resync / s9): lost 0.178 / 0.205
(-13%), morph 0.234 / 0.253 (-8%), shape 0.671 / 0.663, full rows 0.52 / 0.30 (worse at four of six tests: 0.55,
0.19, 0.79, 0.90, 0.68), spawn and hit level. No buttons: level but for spawn 0.95 at 148-152k (s9 1.00). Live
scores 140-152k: rotate 0.42 / 0.50, spawn 0.81 / 0.87, shift 0.82 / 0.88, gravity 0.94 / 0.96, clears level;
turns 0.32 / 0.41 exact with the real press, 0.06 swapped (gap +0.26 / +0.35). Rollouts replaced under 0.1% of
patches (resync almost never fired: the model's own frames trained it). As with actnorm (morph 0.78 -> 0.60, full
0.32 -> 0.53), resync trades a little coherence for full rows that never clear, and here also for turns.
Predictions: 1 failed the other way (1.06 steps/s, faster); 2 held for morph (-8%, at the bottom of 10-25%) and
full rows, lost a little better than level; 4 failed (the turn gap fell).

## One motion step instead of two (planned 2026-10-18)

**Why (the user):** "a post training where we only use 1 step for motion and 1 step for the image". Motion first
decodes a frame in 4 forwards: the motion in two steps (committed_motion: the cells at least 0.9 sure plus the
single most confident other, then the rest given those), the pixels given it, the cache write. The second motion
step came in with s3_motion_mg (2026-10-16) because one-pass motion drew turn fields cell by cell, 46% of turns
half one way and half the other; it only took that to 25-39%, and s9 learned turns later by growing, not by it.
One step is 3 forwards a frame: dreams about 1.3x faster, which matters in the browser (which has no motion path
yet: onnx_export and web/dreamer.js still need one before s9 can ship).

**Measure first, no training:** one-step decoding is already in training's distribution (the first pass is the
plain masked prediction with no motion given, scored by the motion loss; the pixel pass is the "given all"
share). `Dynamics.motion_steps` (1 or 2, a decoding setting, default 2) and `--motion-steps` on
button_swap_check.py and motion_swap_check.py. When the GPU is free (after resync140): s9_cool140's 152k with 1
and with 2 steps on the trainer's 32 + 32 long-dream trials, the pixel and motion swap checks, and dream speed.
A post-training for one step only if one step is clearly worse; the loser of the two decodings is then deleted.

**Predictions (written before measuring):**
1. Speed: 1.25-1.35x the frames per second.
2. Drawn motion's turn fields mix more with one step, and the pixel turn p gap falls by 0.05-0.15.
3. Button dreams: morph and lost up 0-30% (turn fields that mix break pieces); no-button dreams level (still
   pieces' motion is sure in one pass).

**Result (2026-10-18, s9_cool140 152k, the trainer's 32 + 32 trials): one step is as good as two.** Button
dreams, 2 steps / 1 step: lost 0.103 / 0.116, morph 0.169 / 0.194, full 0.45 / 0.00, shape 0.736 / 0.716, spawn
1.00 / 1.00, wrong +128 1.7% / 1.7%; no buttons level (shape 0.732 / 0.721, lost and morph 0). Drawn motion's
turn gap +0.98 both ways (shifts +0.99); pixels draw the real turn 98% with the real press and 0% swapped, p gap
+0.748 both (it was +0.53 at 102k: the cooldown sharpened turns). Dream speed on the 3090, batch 1: 13.5 / 12.4
and 22.5 / 18.6 ms a frame (two runs; launch-bound at this size, so 8-17% here; the work is 3 forwards against
4). Predictions: 1 failed (smaller here, as launches bound it), 2 failed (no change: s9's turn fields do not
mix), 3 held. No post-training needed; the browser decodes with one step (models/onnx_export.py). Whether the
two-step decoding and training's partial-motion share are deleted is the user's call.

**On the website (2026-10-18):** s9_cool140 152k exported (models/onnx_export.py rewritten for the pixel model
with motion first; the layered export it replaces is gone): 96 frames of the page's start, 0 pixels differ from
PyTorch in onnxruntime (Python and node) and within 2 per colour value in Chrome on WebGPU. Speed on the 3090:
41 ms a step in the check, 44-76 ms while playing (12-18 frames/s); a step is about 2,400 kernels (two passes
of nine layers), onnxruntime-web's dispatch the limit. Files 37.7 MB (the local bias folded into a constant
table in each graph). The image start now builds model frames (web/screenshot.js), with no piece marking.
