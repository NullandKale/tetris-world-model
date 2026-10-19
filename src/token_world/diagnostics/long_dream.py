"""Long dreams, with and without buttons: does the world model keep a falling piece intact, let it fall,
and move it as the buttons say?

A trial is a real game paused at a fresh piece: its last CONTEXT real frames and the buttons that made
them, then FUTURE (1,000) real frames, either with nobody pressing anything (gravity alone) or with the bot
playing on (pressed: its presses, Trial.buttons). A model dreams the same FUTURE frames from the context,
given the same buttons. The scores below look at the first FIRST (128) frames, as every test did before
2026-10-09 (FUTURE was 128); over all FUTURE frames, where a dream long since follows its own pieces (random
ones differ from the real game's), the play check's integrity scores (diagnostics/play.py integrity: cells
lost outside a clear, pieces changing their block count, full rows that never clear), which need nothing of
the real game: lost_long, morph_long, full_long and size_long (pieces born too big or too small), per 100
frames. No-button dreams test the clock and the pieces on their own (a model that ignores the buttons
can pass them); pressed dreams test that it moves, rotates and drops the piece when the real one does. Dreams are soft (models/dynamics.py): each pixel is a colour
distribution. Scored on the playfield: pixels expected wrong at HORIZONS (an expectation), and the block
mass at the last frame against the real one (1.0 = as many filled pixels; pieces that melt or grow move
it), where a pixel is filled when the dream gives it a block at least SURE likely: a faint haze of maybe
over the whole well is not a piece (summed as an expectation, a 2% haze over the empty upper playfield
outweighed a whole piece). Trials are live games (no stored set). Used by scripts/long_dream_check.py,
and by training every LONG_DREAM_EVERY steps (scripts/train_dynamics_ui.py).

The falling piece is scored four ways (score): kept (its pixel count, anywhere in the upper playfield),
hit (the share of the real piece's pixels the dream fills, where they are: a piece frozen at the top
scores 0 once the real one has fallen), ghost (pixels the dream fills where the real frame has none,
stuck or doubled pieces), and fall (how far the dream's piece has fallen against the real one). spawn
asks whether the next piece appears when the real one does, after the dream's first piece has left the
spawn area; timer_wrong scores the fall timer drawn in the border (the game's gravity clock).

presence and activation watch for the stall a model trained toward fewer wrong pixels falls into: it
erases the pieces it is unsure of (an empty well is mostly right), and once nothing is drawn where pieces
fall it has no wrong piece to correct toward a right one. presence is the share of frames with a real
piece in which the dream draws any piece at all (anywhere in the upper playfield, right or wrong);
activation the dream's expected block pixels there over the real piece's, however faint (a soft model's
haze counts, a committed model's pixels are 0 or 1). Both falling toward 0 is the warning.

shape asks whether the piece stays the piece it is (an S does not turn into an O): in the frames where the
real upper playfield holds one whole piece (4 cells), the share in which the dream's holds the same cells up to
where they are (so a pressed dream must rotate it when the real one rotates).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset, get_worker_info

from token_world.data.model_frames import BAND, border_codes
from token_world.data.real_tetris import CONTEXT, RealTetris
from token_world.diagnostics.line_clears import WELL, well_cells
from token_world.diagnostics.play import integrity
from token_world.models.dynamics import Dreamer

FUTURE = 1000                                    # frames a trial dreams (128 before 2026-10-09)
FIRST = 128                                      # the frames the piece scores, HORIZONS and the bias test look at
EVERY = 8                                        # the pictures (sheet, animation) show every this many frames
BATCH = 16                                       # trials dreamed side by side (dreams)
PLAYFIELD = (slice(52, 212), slice(92, 172))
UPPER = (slice(52, 140), slice(92, 172))         # the playfield above any stack a fresh-piece trial has
PIECE_HORIZONS = (16, 32)
CROP = (slice(44, 212), slice(88, 176))
SURE = 0.5                       # a dreamed pixel is filled when its probability of a block is at least this
HORIZONS = (16, 32, 64, 128)
BANDS = ((0, 4), (5, 9), (10, 14), (15, 19))
LEVEL, PIECE_Y = 0x44, 0x41
SPAWN = (slice(52, 68), slice(108, 156))         # the top two rows of cells, around where pieces spawn
SPAWN_SLACK = 16                                 # frames a dreamed spawn may be early or late
FALL_FRAMES = 64                                 # the fall is measured over at most this many frames
CELL = 8                                         # the well's cells are 8 x 8 pixels at x = 96 + 8 col, y = 56 + 8 row
CELL_FILLED = 24                                 # a cell holds a block with this many filled pixels (a block: 7 x 7)
UPPER_CELLS = (slice(56, 144), slice(96, 176))   # the well's top 11 rows of cells, all 10 columns


def _timer_pixels() -> np.ndarray:
    """[256, 256] bool: the border pixels that show the fall timer (the first state byte, $0045 in Tetris;
    data/model_frames.py), found as the pixels whose shade changes with it."""
    codes = border_codes(np.zeros(256, np.int64), np.stack([np.arange(256), np.zeros(256, np.int64)], 1))
    varies = (codes != codes[:1]).any(0)                                # [2 * BAND, 256]
    out = np.zeros((256, 256), bool)
    out[:BAND], out[256 - BAND:] = varies[:BAND], varies[BAND:]
    return out


TIMER = _timer_pixels()


@dataclass
class Trial:
    context: np.ndarray           # [CONTEXT, 256, 256] uint8 model frames
    actions: np.ndarray           # [CONTEXT] the button byte that produced each
    future: np.ndarray            # [FIRST, 256, 256] the first real frames (the scores')
    level: int
    layers: dict | None = None    # the context's layered frames, each key [CONTEXT, ...] (data/nes_layers.py)
    buttons: np.ndarray | None = None   # [FUTURE] the button byte that produced each future frame; None: none
    pictures: np.ndarray | None = None  # [FUTURE // EVERY, 256, 256] every EVERY-th real frame (sheet,
                                        # animation); None: from future

    def __post_init__(self):
        if self.buttons is None:
            self.buttons = np.zeros(FUTURE, np.int64)
        if self.pictures is None:
            self.pictures = self.future[EVERY - 1::EVERY]

    @property
    def pressed(self) -> bool:
        return bool(self.buttons.any())

    @property
    def what(self) -> str:
        frames = len(self.buttons)
        return f"the bot's buttons for {frames} frames" if self.pressed else f"no buttons for {frames} frames"


def trials(seed: int, per_band: int | None = None, layered: bool = False, pressed: bool = False) -> Iterator[Trial]:
    """Real games paused at a fresh piece, then FUTURE frames with no buttons (pressed: the bot plays on, and
    its presses are the trial's buttons). per_band: that many from each level band in BANDS, then stop; None:
    endless, whatever levels the bot's games reach. layered: the context's layers too (Trial.layers)."""
    found = {band: 0 for band in BANDS}
    while per_band is None or any(n < per_band for n in found.values()):
        real = RealTetris(seed, layered)
        seed += 1
        for played in range(20_000):
            real.step(None)
            ram = real.observation.ram
            if played > 700 and real.playing() and int(ram[PIECE_Y]) <= 1 and len(real.frames) == CONTEXT:
                level = int(ram[LEVEL])
                band = next((b for b in BANDS if b[0] <= level <= b[1]), None)
                if per_band is not None and (band is None or found[band] >= per_band):
                    break
                if band is not None:
                    found[band] += 1
                context, actions = np.stack(real.frames), np.array(real.actions, np.int64)
                layers = {k: np.stack([f[k] for f in real.layers]) for k in real.layers[0]} if layered else None
                future, buttons = zip(*(real.step(None if pressed else 0) for _ in range(FUTURE)))
                yield Trial(context, actions, np.stack(future[:FIRST]), level, layers, np.array(buttons, np.int64),
                            np.stack(future[EVERY - 1::EVERY]))
                break


class TrialStream(IterableDataset):
    """Endless live trials, one console per DataLoader worker (training keeps one worker filling it)."""

    def __init__(self, seed: int, layered: bool = False, pressed: bool = False):
        self.seed, self.layered, self.pressed = seed, layered, pressed

    def __iter__(self):
        info = get_worker_info()
        yield from trials(self.seed * 1000 + (info.id if info else 0), layered=self.layered, pressed=self.pressed)


@dataclass
class Dream:
    """A dream, kept as what the scores and pictures need of its colour distributions: its first FIRST frames
    whole, its well's blocks every frame, and every EVERY-th frame's picture when asked for."""
    rgb: np.ndarray               # [FIRST, 256, 256, 3] uint8: each pixel's expected colour
    right: dict[int, np.ndarray]  # frame h (HORIZONS) -> [256, 256] each pixel's probability of the real colour
    filled: np.ndarray            # [FIRST, 256, 256] float16: each pixel's probability of a block, every frame
    likely: np.ndarray            # [FIRST, 256, 256] uint8: each pixel's most likely colour (coherence scores)
    sure: np.ndarray              # [FIRST, 256, 256] float16: its probability
    cells: np.ndarray             # [FUTURE, 20, 10] bool: the well's blocks every frame (integrity scores)
    seen: np.ndarray | None = None  # [FUTURE // EVERY, 256, 256, 3] uint8: every EVERY-th frame's expected colours

    @classmethod
    def of_frames(cls, trial: "Trial", frames: np.ndarray, palette: np.ndarray) -> "Dream":
        """Exact frames [T >= FIRST, 256, 256] as a dream that is sure of every pixel (tests, the real game,
        the layered models' committed dreams)."""
        empty = empty_colour(trial)
        first = frames[:FIRST]
        return cls(palette[first], {h: (first[h - 1] == trial.future[h - 1]).astype(np.float32) for h in HORIZONS},
                   (first != empty).astype(np.float16), first, np.ones(first.shape, np.float16),
                   np.stack([well_cells(f, empty) for f in frames]), palette[frames[EVERY - 1::EVERY]])


def empty_colour(trial: "Trial") -> int:
    """The empty well's colour."""
    return int(np.bincount(trial.future[0][UPPER].ravel()).argmax())


@torch.no_grad()
def dreams(model, trials: list[Trial], palette: np.ndarray, keep: int | None = None, change_weight: float = 1.0,
           pictures: int = 0, batch: int = BATCH) -> Iterator[Dream]:
    """Each trial's FUTURE frames from its context, given its buttons (change_weight: Dreamer's), BATCH trials
    dreamed side by side (models/dynamics.py Dreamer) -> its Dream, in order; the first `pictures` trials keep
    their pictures (Dream.seen)."""
    device = next(model.parameters()).device
    colours = torch.as_tensor(palette, dtype=torch.float32, device=device)
    for lo in range(0, len(trials), batch):
        group = trials[lo:lo + batch]
        b = len(group)
        empty = torch.tensor([empty_colour(t) for t in group], device=device)
        buttons = torch.from_numpy(np.stack([t.buttons for t in group])).to(device)
        shown = torch.tensor([i for i in range(b) if lo + i < pictures], dtype=torch.long, device=device)
        rgb, filled, likely, sure, cells, seen, right = [], [], [], [], [], [], {}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            make = getattr(model, "Dreamer", Dreamer)    # the latent model's own (models/latent_dynamics.py)
            dreamer = make(model, torch.from_numpy(np.stack([t.context for t in group])).to(device),
                           torch.from_numpy(np.stack([t.actions for t in group])).to(device), keep, change_weight)
            for h in range(1, FUTURE + 1):
                probs = dreamer.step(buttons[:, h - 1]).float()                  # [b, 256, 256, COLOURS]
                top = probs.max(-1)
                expected = None
                if h <= FIRST:
                    expected = (probs @ colours).round().clamp(0, 255).byte()
                    rgb.append(expected.cpu().numpy())
                    gone = probs.gather(-1, empty.view(b, 1, 1, 1).expand(b, 256, 256, 1))[..., 0]
                    filled.append((1 - gone).half().cpu().numpy())
                    likely.append(top.indices.byte().cpu().numpy())
                    sure.append(top.values.half().cpu().numpy())
                if h in HORIZONS:
                    real = torch.from_numpy(np.stack([t.future[h - 1] for t in group])).to(device).long()
                    right[h] = probs.gather(-1, real[..., None])[..., 0].cpu().numpy()
                block = top.indices[:, WELL[0], WELL[1]] != empty.view(b, 1, 1)               # well_cells
                cells.append((block.reshape(b, 20, CELL, 10, CELL).sum((2, 4)) >= CELL_FILLED).cpu().numpy())
                if h % EVERY == 0 and len(shown):
                    expected = (probs[shown] @ colours).round().clamp(0, 255).byte() if expected is None \
                        else expected[shown]
                    seen.append(expected.cpu().numpy())
        del dreamer                                  # its CUDA graph's memory goes back only after a collection
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for i in range(b):
            kept = int((shown == i).nonzero()[0, 0]) if (shown == i).any() else None
            yield Dream(np.stack([f[i] for f in rgb]), {h: right[h][i] for h in HORIZONS},
                        np.stack([f[i] for f in filled]), np.stack([f[i] for f in likely]),
                        np.stack([f[i] for f in sure]), np.stack([f[i] for f in cells]),
                        np.stack([f[kept] for f in seen]) if kept is not None else None)


def dream(model, trial: Trial, palette: np.ndarray, keep: int | None = None, change_weight: float = 1.0) -> Dream:
    """One trial's dream (dreams), with its pictures."""
    return next(dreams(model, [trial], palette, keep, change_weight, pictures=1))


def score(trial: Trial, dreamed: Dream) -> dict[str, float]:
    """A dream's scores against the real game (NaN where a score does not apply to this trial):
    - wrong_h: playfield pixels expected wrong at each of HORIZONS;
    - timer_wrong_h: the border's fall-timer pixels expected wrong (the gravity clock, TIMER);
    - mass: the dream's filled pixels (SURE) in the playfield at the last frame over the real game's;
    - for the falling piece at PIECE_HORIZONS, in the upper playfield without what never changes there
      (NaN once the real piece has left it): piece_h, its filled pixels over the real piece's, wherever
      they are (1.0 the whole piece, 0 a lost one; wrong pixels instead reward erasing an unsure piece);
      piece_hit_h, the share of the real piece's pixels the dream fills (a piece in the wrong place,
      frozen or lagging, scores 0); piece_ghost_h, the dream's filled pixels where the real frame has
      none, over the real piece's (stuck, doubled or smeared pieces);
    - fall: how far the dream's piece has fallen against the real one (centres of their pixels), over
      the real piece's time in the upper playfield up to FALL_FRAMES (1.0 at the real speed, 0 frozen;
      NaN if the real one moved under a row or either piece is gone);
    - spawn: 1.0 if, after the dream's first piece has left the spawn area, the next piece appears
      within SPAWN_SLACK frames of the real one's; 0 otherwise (NaN if the real game spawns none);
      spawn_lag: its frames late (negative: early);
    - presence: over the frames whose upper playfield holds a real piece, the share in which the dream's
      holds a piece's worth of filled pixels, right or wrong; activation: the dream's expected block
      pixels there (each pixel's probability, however faint) over the real piece's (NaN with no piece);
    - shape: over the frames whose upper playfield holds one whole real piece (4 cells), the share in which
      the dream's holds the same cells, wherever they are (an erased, garbled or swapped piece scores 0);
    - lost_long, morph_long, full_long, size_long: the integrity scores (play.py) over all FUTURE frames.
    Every score but the last three looks at the first FIRST frames."""
    out = {f"wrong_{h}": float(1 - dreamed.right[h][PLAYFIELD].mean()) for h in HORIZONS}
    out |= {f"timer_wrong_{h}": float(1 - dreamed.right[h][TIMER].mean()) for h in HORIZONS}
    empty = empty_colour(trial)
    real_frames = trial.future != empty                                          # [FUTURE, 256, 256]
    sure = dreamed.filled >= SURE
    out["mass"] = float(sure[-1][PLAYFIELD].sum() / max(real_frames[-1][PLAYFIELD].sum(), 1))
    static = np.all(np.concatenate([trial.context, trial.future])[:, UPPER[0], UPPER[1]] != empty, 0)
    real_upper = real_frames[:, UPPER[0], UPPER[1]] & ~static                    # [FUTURE, h, w]
    dream_upper = sure[:, UPPER[0], UPPER[1]] & ~static
    for h in PIECE_HORIZONS:
        real_piece, mine = real_upper[h - 1], dream_upper[h - 1]
        size = int(real_piece.sum())
        ok = size > 50
        out[f"piece_{h}"] = float(mine.sum()) / size if ok else float("nan")
        out[f"piece_hit_{h}"] = float((mine & real_piece).sum()) / size if ok else float("nan")
        out[f"piece_ghost_{h}"] = float((mine & ~real_piece).sum()) / size if ok else float("nan")
    pieces = real_upper.reshape(len(real_upper), -1).sum(1)
    with_piece = pieces > 50
    if with_piece.any():
        out["presence"] = float((dream_upper.reshape(len(dream_upper), -1).sum(1)[with_piece] > 50).mean())
        haze = (dreamed.filled[:, UPPER[0], UPPER[1]].astype(np.float32) * ~static).reshape(len(pieces), -1).sum(1)
        out["activation"] = float((haze[with_piece] / pieces[with_piece]).mean())
    else:
        out["presence"] = out["activation"] = float("nan")
    still = np.all(np.concatenate([trial.context, trial.future])[:, UPPER_CELLS[0], UPPER_CELLS[1]] != empty, 0)
    out["shape"] = _shape(cells(real_frames[:, UPPER_CELLS[0], UPPER_CELLS[1]] & ~still),
                          cells(sure[:, UPPER_CELLS[0], UPPER_CELLS[1]] & ~still))
    out["fall"] = _fall(real_upper, dream_upper)
    out["spawn"], out["spawn_lag"] = _spawn(real_frames, sure, static, empty)
    start = well_cells(trial.context[-1], empty)[None]
    out |= {f"{k}_long": v for k, v in integrity(np.concatenate([start, dreamed.cells])).items()}
    return out


def cells(pixels: np.ndarray) -> np.ndarray:
    """Pixel masks [..., h, w] cut on the cell grid (UPPER_CELLS) -> cell masks [..., h // CELL, w // CELL]:
    a block."""
    h, w = pixels.shape[-2] // CELL, pixels.shape[-1] // CELL
    blocks = pixels[..., :h * CELL, :w * CELL].reshape(*pixels.shape[:-2], h, CELL, w, CELL)
    return blocks.sum((-3, -1)) >= CELL_FILLED


def _trimmed(mask: np.ndarray) -> bytes | None:
    """A cell mask's cells, cut to their bounding box (where they are does not matter), as a key."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return bytes([*box.shape]) + np.packbits(box).tobytes()


def _shape(real: np.ndarray, mine: np.ndarray) -> float:
    """shape (score), from the real and dreamed cells [FUTURE, rows, cols] that change."""
    whole = [t for t in range(len(real)) if real[t].sum() == 4]
    if not whole:
        return float("nan")
    return float(np.mean([_trimmed(real[t]) == _trimmed(mine[t]) for t in whole]))


def _centre_row(pixels: np.ndarray) -> float:
    """The mean row of a [h, w] mask's pixels, NaN with under 50 of them (no piece)."""
    rows = np.nonzero(pixels)[0]
    return float(rows.mean()) if len(rows) > 50 else float("nan")


def _fall(real_upper: np.ndarray, dream_upper: np.ndarray) -> float:
    """fall (score): the dream piece's drop over the real one's, from the first frame to the real piece's
    last in the upper playfield (at most FALL_FRAMES)."""
    present = np.nonzero(real_upper.reshape(len(real_upper), -1).sum(1) > 50)[0]
    if len(present) == 0 or present[0] != 0:
        return float("nan")
    last = min(int(np.nonzero(np.diff(present) != 1)[0][0]) if (np.diff(present) != 1).any() else
               int(present[-1]), FALL_FRAMES - 1)
    real = _centre_row(real_upper[last]) - _centre_row(real_upper[0])
    mine = _centre_row(dream_upper[last]) - _centre_row(dream_upper[0])
    return mine / real if real >= 8 and not np.isnan(mine) else float("nan")


def _spawn(real_frames: np.ndarray, sure: np.ndarray, static: np.ndarray, empty: int) -> tuple[float, float]:
    """spawn, spawn_lag (score), from when the spawn area holds a piece (over a cell's worth of pixels)."""
    still = static[SPAWN[0].start - UPPER[0].start:SPAWN[0].stop - UPPER[0].start,
                   SPAWN[1].start - UPPER[1].start:SPAWN[1].stop - UPPER[1].start]
    held = lambda frames: (frames[:, SPAWN[0], SPAWN[1]] & ~still).reshape(len(frames), -1).sum(1) > 64

    def next_piece(inside: np.ndarray) -> int | None:
        gone = np.nonzero(~inside)[0]
        back = np.nonzero(inside[gone[0]:])[0] if len(gone) else []
        return int(gone[0] + back[0]) if len(back) else None

    real = next_piece(held(real_frames))
    if real is None:
        return float("nan"), float("nan")
    mine = next_piece(held(sure))
    if mine is None:
        return 0.0, float("nan")
    return float(abs(mine - real) <= SPAWN_SLACK), float(mine - real)


def sheet(trial: Trial, dreams: dict[str, Dream], palette: np.ndarray, every: int = 64) -> Image.Image:
    """Rows: the real game, then each dream (expected colours; its pictures, Dream.seen); columns: the playfield
    every `every` frames (a multiple of EVERY), scaled 2x."""
    h, w = trial.future[0][CROP].shape
    rows = {"real": palette[trial.pictures]} | {name: d.seen for name, d in dreams.items()}
    columns = range(every // EVERY - 1, len(trial.pictures), every // EVERY)
    out = Image.new("RGB", (220 + len(columns) * (w * 2 + 4), 20 + len(rows) * (h * 2 + 6)), "#181820")
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), f"level {trial.level}, {trial.what}; playfield every {every} frames",
              fill="white")
    for r, (name, frames) in enumerate(rows.items()):
        y = 20 + r * (h * 2 + 6)
        draw.text((4, y + h), name, fill="white")
        for c, f in enumerate(columns):
            out.paste(Image.fromarray(frames[f][CROP]).resize((w * 2, h * 2), Image.NEAREST),
                      (220 + c * (w * 2 + 4), y))
    return out


def animation(trial: Trial, dreams: dict[str, Dream], palette: np.ndarray) -> list[Image.Image]:
    """GIF frames: the real playfield next to each dream's (expected colours), every EVERY frames (their
    pictures), scaled 2x, labelled."""
    h, w = trial.future[0][CROP].shape
    panels = {"real": palette[trial.pictures]} | {name: d.seen for name, d in dreams.items()}
    width = len(panels) * (w * 2 + 8)
    frames = []
    for i in range(len(trial.pictures)):
        f = (i + 1) * EVERY - 1                                                 # the frame this picture shows
        image = Image.new("RGB", (width, h * 2 + 34), "#181820")
        draw = ImageDraw.Draw(image)
        for n, (name, seq) in enumerate(panels.items()):
            x = n * (w * 2 + 8)
            draw.text((x + 2, 2), name, fill="white")
            image.paste(Image.fromarray(seq[i][CROP]).resize((w * 2, h * 2), Image.NEAREST), (x, 16))
        draw.text((2, h * 2 + 20), f"level {trial.level}, frame +{f + 1}, "
                  + (f"buttons {trial.buttons[f]:#04x}" if trial.pressed else "no buttons"), fill="#a8a8b8")
        frames.append(image)
    return frames


def stacked(*pictures):
    """Pictures (sheet images, or animation frame lists of the same length) one above the other: a test's
    variants (no buttons, then the bot's buttons) in one long_dream.png / .gif."""
    if isinstance(pictures[0], list):
        return [stacked(*frames) for frames in zip(*pictures)]
    out = Image.new("RGB", (max(p.width for p in pictures), sum(p.height for p in pictures)), "#181820")
    y = 0
    for p in pictures:
        out.paste(p, (0, y))
        y += p.height
    return out


def dream_world(world, trial: Trial, palette: np.ndarray, keep: int | None = None, seed: int = 0) -> Dream:
    """Any model's (diagnostics/worlds.py World) FUTURE frames from the trial's context and buttons: the pixel
    model's dream (dream), or a layered model's (dream_layered: the trial needs layers)."""
    if world.layered:
        return dream_layered(world.model, trial, palette, seed=seed, steps=world.saved["args"].get("dream_steps", 1))
    return dream(world.model, trial, palette, keep)


@torch.no_grad()
@torch.no_grad()
def dreams_layered(model, trials: list[Trial], palette: np.ndarray, temperature: float = 1.0, seed: int = 0,
                   steps: int = 1, batch: int = BATCH, objects: bool = False) -> Iterator[Dream]:
    """A layered model's (models/layered.py) FUTURE frames for each trial from its layered context and buttons,
    BATCH trials dreamed side by side (its Dreamer takes a batch of games), composed into model frames
    (data/nes_layers.py) and scored as committed frames (Dream.of_frames) -> its Dream, in order. objects: the
    slot model's decisions per object (models/layered_slots.py)."""
    from token_world.data.nes_layers import TRANSPARENT, compose, model_frame
    device = next(model.parameters()).device
    for lo in range(0, len(trials), batch):
        group = trials[lo:lo + batch]
        layers = {k: torch.from_numpy(np.stack([np.asarray(t.layers[k]) for t in group])).to(device)
                  for k in group[0].layers}
        buttons = torch.from_numpy(np.stack([t.buttons for t in group])).to(device)
        generator = torch.Generator(device=device).manual_seed(seed + lo)
        frames = [[] for _ in group]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dreamer = model.Dreamer(model, layers, torch.from_numpy(np.stack([t.actions for t in group])).to(device),
                                    temperature=temperature, generator=generator, steps=steps, objects=objects)
            for h in range(FUTURE):
                out = {k: v.cpu().numpy() for k, v in dreamer.step(buttons[:, h]).items() if k != "probs"}
                for i, made in enumerate(frames):
                    made.append(model_frame(compose({k: v[i] for k, v in out.items()}), out["border"][i]))
        del dreamer                                  # as dreams(): its memory back before the next batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for trial, made in zip(group, frames):
            made = np.stack(made)
            yield Dream.of_frames(trial, np.where(made == TRANSPARENT, 0, made), palette)


def dream_layered(model, trial: Trial, palette: np.ndarray, temperature: float = 1.0, seed: int = 0,
                  steps: int = 1) -> Dream:
    """One trial's layered dream (dreams_layered)."""
    return next(dreams_layered(model, [trial], palette, temperature, seed, steps))

