# Turn direction: what we tried, why it fails, what's next (2026-10-16)

Every pixel model since actnorm shifts pieces on the button (left vs right) but none turns them the way the
button says (A vs B). It knows *that* a turn button was pressed (no press: never a turn) but not *which*. Two
thirds of all pieces breaking in dreams with buttons happen within two frames of a turn (recipe_attempts.md,
"Where the pixel model breaks pieces: rotations"): the model draws a blend of the two turns, and the blend goes
back into the history. This page keeps the discussion of 2026-10-16: everything tried, what we now think the
cause is, and the next three experiments.

## How it is measured

- **Pixel swap test** (`scripts/button_swap_check.py`): J / L / T turns where the opposite turn would also
  fit; the same real history, one frame ahead, with the real press, the press swapped A <-> B, and none. The
  **p gap** is the probability of the real picture on the cells the two turns disagree on, real press minus
  swapped. +0.00: the button is ignored. Shifts reach +0.7.
- **Motion swap test** (`scripts/motion_swap_check.py`): the same for the drawn motion (motion models).
- **Turn fields** (scratchpad `motion_coherence.py`): each drawn turn's motion classified as exactly the real
  turn, the opposite, some of each, part of the real, part of the opposite, or neither.
- Reports compare trends at the same steps, button dreams first (memory: judge-by-buttons).

## What we think the cause is

1. **The hedge is the right answer for a model that ignores the button.** Any per-pixel loss hedges then:
   cross-entropy trains toward 50 / 50, L2 toward the average (both turns blended, the classic blur), L1 toward
   the per-pixel median (each cell its own way: our "some of each"). Changing the per-pixel loss cannot make
   the model use an input it ignores.
2. **The hedge costs almost nothing.** Only J, L and T show a direction (I, S and Z have two orientations, so A
   and B give the same picture; O does not turn). The cells that tell the two turns apart are a few of the
   changed game pixels, themselves 0.16% of the masked pixels.
3. **There are cheaper ways to the answer than the button.** Teacher-forced, part of the predicted frame is
   visible, and the direction can be read off it: s3_motion_local gets turned cells right 0.86 teacher-forced
   while its swap gap is +0.02; with press frames hidden (s3_branch) that falls to 0.35. And the bot turns the
   short way to its planned spot, so the board predicts the button for planned turns.
4. **Not the button's path.** A and B are far apart where they enter (embedding distance 0.63, block
   modulations 0.29-0.34, against left/right 0.86 / 0.76-0.87); the direction reaches every block and is not used.
   Buttons in the last block only (s3_late) did not help.
5. **A turn is harder to represent than a shift.** A shift is one motion code (dx = +-1) on every cell, so the
   button maps straight onto it. A turn moves each cell by a different offset, depending on where it sits around
   the centre, so the button has to combine with each cell's place.

The data is not the excuse: about half the J / L / T turn frames already have a direction the board cannot
predict (50k frames of the training stream: perfect brain planned 480, holding 288; perfect brain **spinning**
259, random brain 386, reckless 274, button-down frames, one sample). The spin brain is the perfect brain's
`spin` option (`data/tetris_bot.py` PerfectBrain; BrainSwitcher.SPIN = 0.5: half of the patient pieces, 30% of
its pieces, over stacks of 8 rows or less, press A or B at random at 1/8 a frame while they wait).

## Everything tried

Turn results are the pixel swap test's p gap unless said; details in recipe_attempts.md, latent_world_model.md,
video_vae.md and layered_tokens.md.

### Token schemes

| what | when | result |
|---|---|---|
| palette patch tokenizer, recurrent temporal model, history compressor, metasprite splat decoder, per-cell exact memory | Sep | replaced by the Genie-style pixel MaskGIT (09-27) |
| per-token codes (codes, codes_nolevels, modern) | 10-01 | erased pieces: probability spread over many piece codes, the one background code wins |
| copy codes (stay / move down one cell) | 10-02 | proposed, never built |
| layered slots: 64 OAM sprite slots (x, y, tile) | 10-06 | the piece's 4 sprites each chose their own move and the piece came apart; hit 0.2, ghost 0.9 |
| slots + rotary time | 10-07 | the same failure (the clock was not it) |
| slots + actnorm (`slots_actnorm_small`) | 10-09 | failed, not written up before: 14k hit 0.23, ghost 0.91 (pixel actnorm 1.00 / 0.01) |
| layered pixels: background canvas + a sprite-picture layer | 10-06/07 | best model then (58k), learned buttons; turns 12 / 12 / 68% mix, the same swapped |
| video VAE latents (n1, c4/c6, n1s, n1sn sprite numbers) + flow / EDM / fast / context-noise / moved-weight generators | 10-10 to 13 | latents hold motion cleanly; generators break pieces far more (n1sn 3.9 lost / 4.4 morph per 100 frames without buttons, pixel 0.03 / 0.08); sprite numbers stayed on the piece but did not keep it whole |

### Recipes

| what | result |
|---|---|
| original small: own guesses, swaps, rollouts 1-32, tossing | 28 / 32 intact; the only recipe that kept pieces for a week |
| standard (teacher forcing + context noise); + tossing / Self Forcing / GAN critic | 17 / 32 best; Self Forcing 29 erased; the GAN (10-01) grained the frames |
| noborder, noborder_soft, stage1 (soft sets), noswap, blend | lost pieces; the blend brought them back |
| HorizonDrive two stages (base, srr) | srr 20k shipped |
| random latent (DreamerV3 choice) x4 | carried the timer's phase, not the random event |
| rollouts on the rope model; resync rollouts; onestage | rollouts hurt until resync; onestage turn +0.002 |

### Architecture

| what | result |
|---|---|
| rotary time | the big jump; learned per-slot time broke the clock |
| 2D rotary space, QK-norm, learned place kept | slower to start, kept |
| **actnorm** (buttons on every block's norms) | buttons work, hedging everywhere stops; turn +0.026 at 72k |
| depth chains (hidden, plain), grown and copied layers | grown layers were near no-ops |
| buttons in the last block only (s3_late) | turn +0.002 |
| local spatial bias (s3_motion_local) | never used: heads stayed global |

### Aimed at turns

| what | turn result |
|---|---|
| bot: short-way turns (A / B balanced), spin brain always on | no gap on their own |
| half of masked frames hidden (s3_hidden and chain) | +0.000 at every depth |
| change-weighted loss | +0.005 (weighted the real turn's cells, not the wrong one's) |
| press frames weighted | +0.012, the only steady rise; cost the clock |
| motion in and out (motion2) | pixels +0.015; drawn motion +0.12, the first direction signal |
| motion first (motionfirst) | motion +0.20-0.25, pixels +0.03; pixels follow given motion, fields mix (46%) |
| two-step MaskGIT motion (motion_mg) | motion +0.14-0.19, pixels +0.01-0.05 |
| everything + local bias (motion_local, to 62.2k) | fewest breaks of any run; turns +0.014-0.036 pixels, +0.06-0.07 motion |
| press frames hidden + counterfactual twins (s3_branch, 58k to 66k) | at or below its control: 60k / 62k pixel +0.019 / -0.006 (control +0.035 / +0.036), motion +0.05 / +0.04 (+0.07 / +0.07); teacher-forced turned cells fell from 0.86 to 0.35 |

### Decoding only

Two-step confident-first (halved 512-frame error); soft context; committed vs soft; sprites one per pass;
next-piece decodings; decode-time change weighting (ghosts); sampled vs most likely motion (most likely wins).

## The standard fixes, by what they fix

| problem | standard fix | tried |
|---|---|---|
| the loss averages possible outcomes into a blend | a generative loss that commits to one: GAN, diffusion / flow, discrete codes | GAN critic 10-01 (grain), flow / EDM on latents (broke pieces), codes (erased) |
| the hard pixels are a tiny share of the loss | **focal loss**: cross-entropy x (1 - p)^gamma (Lin et al. 2017) | no |
| the model does not depend on an input | a **contrastive / paired loss** on that input | no (the twins are an unpaired version) |

## The next three experiments, in this order

The user, 2026-10-16: try all three; the GAN first ("we tried the gan VERY early and may have improved it"). Each is
a fork of s3_motion_local, compared with s3_motion_local's own steps.

### 1. A critic that judges the turn with its button

The 10-01 critic judged whole generated frames over 8 frames with no button: at factor 0.1 nothing, at 1.0 the
frames turned to grain (its adaptive weight balanced the pulls only at the pixel head). Since then the model
learned the buttons (actnorm), rotary time and motion; and the critic can be made to judge exactly the thing we
want:
- **Conditional and matching-aware** (Reed et al. 2016, GAN-CLS): the critic sees the frame before, the
  frame, and the button. Real: the real frame with the real button. Fake: the model's frame with the real
  button, and the real frame with the swapped button. A blend looks wrong, and a clean turn the wrong way looks
  wrong *for that button*: the critic can only tell real from swapped by the direction.
- **Only at turns:** J / L / T turn frames, cropped around the piece (from the real motion's moved cells), and
  those frames hidden whole when predicted (as a dream draws them; otherwise the model reads the turn off the
  visible part).
- **Hard samples for the fake** (Gumbel-softmax, straight through): the model's frame goes to the critic as
  one colour per pixel, like the real one, so the critic cannot win on softness alone (the likely cause of the
  10-01 grain).
- Hinge loss, a small fixed weight, the critic trained alongside on the same batches.

### 2. Focal loss

Cross-entropy times (1 - p)^gamma: a pixel the model is sure of counts for almost nothing, a 50 / 50 one in
full, the wrong turn's cells included (the change weight missed those). It keeps each pixel's most likely
colour but softens confidence, so the soft frames fed back shift a little. It makes a hedge expensive; it cannot
make the model use the button.

### 3. The paired loss

The same window, the same mask, the press frame hidden in both passes; one pass with the real button and one
with the other. Forms, weakest to strongest: unlikelihood on the real outcome under the swapped button; a margin
on log p(real | real button) - log p(real | swapped); and a positive target for the swapped button (the opposite
turn's motion, from RAM: `opposite_codes` in the motion swap check). Only on J / L / T turns the game accepted, on
the cells where the two turns disagree (elsewhere A and B give the same picture, and punishing it would be
false). One extra forward on the windows with such a turn.

### Also listed, not scheduled

Contrastive button guidance at decode time (no training; amplifies whatever gap exists, two passes on turn
frames in the browser); one motion code per turning piece (turning clockwise / counter-clockwise on all its
cells, as a shift is one code); more random-direction turns in the bot; hard-local attention on half the heads.

## Running them

**One GPU process at a time** (memory: one-gpu-process). On 2026-10-16 two trainers and two checks filled the 24
GB card and every process thrashed over PCIe from 20:06 until they were killed at 21:25. A control runs inside
the same trainer on the same batches, and checks run between training, never beside it. Under 500 MB of free
VRAM means it is thrashing.

Where the runs stopped (all killed 21:25, checkpoints fine): s3_motion_local at 62,200 (model_latest.pt);
s3_branch at 66,000 (its 64k and 66k swap checks not taken).
