"""A game next to its world models: the real game and every model's dream, frame by frame in lockstep.

    python scripts/play_world_model.py [--runs output/world_model_tetris_base D:/.../model_step58000.pt ...]
    python scripts/play_world_model.py --game smb [--stage 1-1] [--runs output/scale_ab/world_model_smb_s6_stairs]

--game smb (2026-10-25): Super Mario Bros. from a level's start (--stage), the training bot (or you) playing
WARMUP frames before the dreams take over from the last frames' two layers, buttons and motion codes (a two-game
model gets them in its own vocabulary); X jumps (A), Z runs (B).

The real game runs in World NES. Every frame one controller byte (the bot's, by
default, or your keys) goes to the console and to every model alike, any model:
the pixel model or a layered one (diagnostics/worlds.py). The dreams start as
the browser's does: from the first 3 frames of a level-0 game
(data/real_tetris.py game_start; --seed 0, the default, is the very start the
browser ships), the real game running on from the same frames; after that each
dream sees only the buttons. --runs names the models (run folders or archived
checkpoints, any steps); without it, the latest long-trained run.

A worker thread owns the console, the bot and the models, steps every dream
once per real frame, and composes the real game and each dream (or each
dream's difference from the real game) into one image per frame; the UI copies
that image to its canvas once per frame. Frames advance at up to the speed
cap (60 = the NES), or as fast as all the models generate together.

The pixel model's dream is soft (models/dynamics.py): each pixel is a colour
distribution, shown as its expected colour, and its difference shows how likely
each pixel is to be wrong. A layered model's dream is committed: its layers
composed, and its difference shows the wrong pixels. The toolbar sets the pixel
models' change weight (the odds of every pixel change times it,
models/dynamics.py weigh_changes, so an unsure piece is drawn with too many
cells instead of thin ones) and --keep how many frames a layered or latent
model keeps when its window slides (default three quarters of each model's
window; the pixel model's history is a ring, never re-encoded:
models/dynamics.py TemporalCache).

Inference (the toolbar's second row; a change starts the game and the dreams again): the pixel models'
settings measured by the 2026-10-30 tests (models/decoding.py): 4 Halton passes a frame (slow, accurate: lost and
morphed pieces cut 3-4x) and unsure patches (< 0.99) hidden from the history. --policy: a policy head (models/policy_head.py, scripts/policy_probe.py) trained
on the --runs model; the controller Head then has each dream play itself, from the head reading that dream's
newest frame: a chunk head's chunk, its first presses played before the next; a per-frame head's press each frame
(the real game goes on with the bot, for reference).

Keys: arrows move and soft-drop, X / Z rotate, Enter is Start, Right Shift is
Select (with the controller on You). S taps Start into the dreams alone: a dream that has topped out before
the real game waits on the covered board for Start, as the real game does (it never leaves without one), and
the real game's Start only comes when the real game ends. R starts every dream again from the game's start
(the same one every time), D shows dreams or differences, Space pauses, Esc quits.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.model_frames import BORDER_VERSION, GAME_ROWS, border_slots
from token_world.data.tetris_bot import A, B, DOWN, LEFT, RIGHT, SELECT, START, UP
from token_world.data.nes_palette import tetris_palette
from token_world.data.real_tetris import game_start, start_context
from token_world.diagnostics.worlds import World, load
from token_world.models.policy_head import Player, load_head
from token_world.ui.check_window import crash, tk_catch_callbacks

KEYS = {"Left": LEFT, "Right": RIGHT, "Down": DOWN, "Up": UP, "x": A, "X": A, "z": B, "Z": B,
        "Return": START, "Shift_R": SELECT}
BUTTON_NAMES = (("A", A), ("B", B), ("Select", SELECT), ("Start", START), ("Up", UP), ("Down", DOWN),
                ("Left", LEFT), ("Right", RIGHT))
BG = "#15151c"


MIN_STEPS = 20_000                              # long-trained runs only
WARMUP = 64                                     # SMB: frames the bot plays from the level's start before dreaming


def runs_with_checkpoints(game: str = "tetris") -> list[Path]:
    """Long-trained world-model runs of `game` (train_dynamics_ui.py, train_layered.py) on the current frame
    layout: a checkpoint, a run config naming the game and BORDER_VERSION (or a layered model's), and at
    least MIN_STEPS steps; the most recently saved last."""
    runs = []
    for checkpoint in [*(ROOT / "output").glob("*/model_latest.pt"), *(ROOT / "output").glob("*/*/model_latest.pt")]:
        run = checkpoint.parent
        config, metrics = run / "run_config.json", run / "metrics.csv"
        if not (config.is_file() and metrics.is_file()):
            continue
        c = json.loads(config.read_text())
        if (c.get("border") != BORDER_VERSION and c.get("kind") != "layered") or game not in c.get("games", []):
            continue
        last = metrics.read_text().strip().splitlines()[-1].split(",")[0]
        if last.isdigit() and int(last) >= MIN_STEPS:
            runs.append(run)
    return sorted(runs, key=lambda r: (r / "model_latest.pt").stat().st_mtime)


class RealSmb:
    """Super Mario Bros. in World NES for the app: a game of the training bot (or you) from a level's start, with
    the last frames' pictures, two layers, incoming buttons and motion codes (what a dream starts from)."""

    def __init__(self, seed: int, stage: tuple[int, int], context: int):
        from collections import deque
        from token_world.data.world_nes_smb import SmbGame, smb_palette
        self.game = SmbGame(seed, sprites=True)
        self.game.start_level(*stage)
        self.slots = border_slots(smb_palette())
        self.pictures, self.layers, self.actions, self.motion = (deque(maxlen=context) for _ in range(4))
        self.game.step()                                  # loads the level's start
        self._observe(0)
        for _ in range(WARMUP):
            self.step(None)

    def _observe(self, action: int) -> np.ndarray:
        from token_world.data.nes_layers import TWO_LAYER_REGIONS, two_layers
        from token_world.data.smb_motion import GRID, UNKNOWN, frame_motion
        from token_world.data.world_nes_smb import smb_model_frame
        o = self.game.observation
        x = smb_model_frame(o, self.game.tick, self.slots)
        self.motion.append(frame_motion(self.pictures[-1], x) if self.pictures
                           else np.full((GRID, GRID), UNKNOWN, np.uint8))
        self.pictures.append(x)
        self.layers.append(two_layers(x, {k: o.regions[k] for k in TWO_LAYER_REGIONS}))
        self.actions.append(action)
        return x

    def step(self, action: int | None) -> tuple[np.ndarray, int]:
        """One frame with `action` (None: the bot's) -> (model frame [256, 256], the action)."""
        action, jumped = self.game.step(action)
        if jumped:                                        # a retry or level start: a new history
            for history in (self.pictures, self.layers, self.actions, self.motion):
                history.clear()
        return self._observe(action), action

    def new_bot(self) -> None:
        """The bot takes over again (its session carries on)."""

    def context(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The last frames: pictures [T, 256, 256], layers [T, 2, 256, 256], incoming buttons [T], motion [T, 32, 32]
        (SMB's codes)."""
        return (np.stack(self.pictures), np.stack(self.layers), np.array(self.actions, np.int64),
                np.stack(self.motion))


def run_label(run: Path) -> str:
    """A run folder's name, or an archived checkpoint's run and step (D:/.../<run>/model_step58000.pt)."""
    return run.name if run.is_dir() else f"{run.parent.name} @ {run.stem.removeprefix('model_step')}"


def run_args(run: Path) -> dict:
    """A run's (folder or checkpoint file) saved args."""
    if run.is_dir():
        return json.loads((run / "run_config.json").read_text())
    return torch.load(run, map_location="cpu", weights_only=False)["args"]


def window(run: Path) -> int:
    """The frames a run (folder or checkpoint file) was trained on."""
    return run_args(run)["frames"]


def load_model(run: Path, motion_steps: int) -> tuple[World, str]:
    world = load(run)
    world.model.motion_steps = motion_steps               # a motion-first model's decoding (as the browser: 1)
    params = sum(p.numel() for p in world.model.parameters()) / 1e6
    kind = "layered, committed" if world.layered else "latent, sampled" if world.latent else "soft"
    where = f"{run.name} @ {world.step}" if run.is_dir() else run_label(run)
    return world, f"{where} ({params:.1f}M, EMA, {kind})"


class Composer:
    """The real game and each dream (or its difference), scaled, side by side in one preallocated RGB image."""

    def __init__(self, palette: np.ndarray, scale: int, panels: int, gap: int = 12):
        self.palette, self.scale, self.gap = palette, scale, gap
        self.panel = 256 * scale
        self.image = np.empty((self.panel, panels * self.panel + (panels - 1) * gap, 3), np.uint8)
        self.image[:] = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)

    def x(self, i: int) -> int:
        return i * (self.panel + self.gap)

    def put(self, i: int, rgb: np.ndarray) -> None:
        s = self.scale
        self.image[:, self.x(i):self.x(i) + self.panel] = rgb.repeat(s, 0).repeat(s, 1)

    def compose(self, real: np.ndarray, dreams: list[tuple[np.ndarray, np.ndarray]], differences: bool) -> np.ndarray:
        """real [256, 256] palette indices; dreams: each model's expected colours [256, 256, 3] and each pixel's
        probability of anything but the real colour [256, 256]; differences: show those (magenta = wrong)."""
        real_rgb = self.palette[real]
        self.put(0, real_rgb)
        for i, (dream, wrong) in enumerate(dreams, 1):
            if differences:
                w = wrong[..., None]
                dream = (real_rgb * 0.3 * (1 - w) + np.array([255, 0, 200]) * w).astype(np.uint8)
            self.put(i, dream)
        return self.image


class Dream:
    """One model's live dream in the app, with its running score."""

    def __init__(self, run: Path, world: World, name: str, keep: int):
        self.run, self.world, self.name, self.keep = run, world, name, keep
        self.dreamer, self.dreamed, self.wrong, self.wrong_sum = None, 0, 0.0, 0.0
        self.pressed = 0                                   # its last press (the Head controller)
        self.player = None                                 # the head's presses for this dream (policy_head.Player)


class Engine(threading.Thread):
    """Runs the real game and every dream in lockstep; publishes one composed image per frame."""

    def __init__(self, runs: list[Path], keep: int | None, palette: np.ndarray, scale: int, seed: int,
                 motion_steps: int = 1, game: str = "tetris", stage: tuple[int, int] = (0, 0),
                 policy: Path | None = None):
        super().__init__(daemon=True)
        self.policy, self.head = policy, None
        self.options = {"steps": 1, "past": None}                       # models/decoding.py OptionDreamer
        decode = run_args(runs[0]).get("decode") if runs else None
        if decode:                                     # the first model's own (Dynamics.decode): its tests' way
            self.options = {"steps": decode.get("steps", 1), "past": decode.get("past")}
        self.choices = torch.Generator(device="cuda").manual_seed(seed)
        self.runs, self.keep, self.seed, self.motion_steps = runs, keep, seed, motion_steps
        self.game, self.stage = game, stage
        self.palette = palette
        self.colours = torch.as_tensor(palette, dtype=torch.float32, device="cuda")
        self.composer = Composer(palette, scale, 1 + len(runs))
        self.commands: queue.Queue = queue.Queue()
        self.keys: set[int] = set()
        self.tapped: set[int] = set()                      # pressed since the last frame read the keys
        self.keys_lock = threading.Lock()
        self.lock = threading.Lock()
        self.version, self.image, self.stats = 0, None, {"status": "starting"}
        self.controller, self.paused, self.fps_cap, self.stopping = "bot", False, 60, False
        self.change_weight, self.differences = 1.0, False
        self.dream_start = 0                               # frames left of a Start tap into the dreams alone
        self.dreams: list[Dream] = []

    # -- called from the UI thread ------------------------------------------------------------
    def buttons(self) -> int:
        """The next frame's byte: the keys held, and any pressed since the last frame read them (2026-11-01: a
        tap shorter than a frame never reached the game or the model; now it gets one frame)."""
        with self.keys_lock:
            byte = sum(self.keys | self.tapped)
            self.tapped.clear()
            return byte

    def press(self, button: int, down: bool) -> None:
        with self.keys_lock:
            (self.keys.add if down else self.keys.discard)(button)
            if down:
                self.tapped.add(button)

    def latest(self):
        with self.lock:
            return self.version, self.image, dict(self.stats)

    # -- worker thread ------------------------------------------------------------------------
    def publish(self, image: np.ndarray | None, **stats) -> None:
        with self.lock:
            if image is not None:
                self.image = image.copy()
                self.version += 1
            self.stats.update(stats)

    def load(self) -> None:
        """Every model in --runs (one that fails to load crashes the app, check_window.crash)."""
        for run in self.runs:
            self.publish(None, status=f"loading {run_label(run)} ...")
            world, name = load_model(run, self.motion_steps)
            self.dreams.append(Dream(run, world, name, self.keep or window(run) * 3 // 4))
        if self.policy is not None:
            self.head = load_head(self.policy, "cuda")

    def restart(self) -> None:
        """The real game back at its start (game_start, the same one every time) and every dream anew from its
        frames, as the browser starts."""
        for d in self.dreams:
            d.dreamer = None
        torch.cuda.empty_cache()
        if self.game == "smb":
            from token_world.data.motion_vocab import to_vocabulary
            w, l = self.stage
            self.publish(None, status=f"starting {w + 1}-{l + 1} (the bot plays {WARMUP} frames) ...")
            self.real = RealSmb(self.seed, self.stage, max(d.keep for d in self.dreams))
            pictures, layers, actions, motion = self.real.context()
            for d in self.dreams:
                model = d.world.model
                codes = (to_vocabulary(torch.from_numpy(motion), "smb", model.motion_codes).numpy()
                         if getattr(model, "motion", False) else None)
                d.dreamer = d.world.dreaming(layers if getattr(model, "two_layers", False) else pictures, actions,
                                             None, d.keep, change_weight=self.change_weight, motion=codes,
                                             options=self.options)
                d.dreamed, d.wrong, d.wrong_sum = 0, 0.0, 0.0
            self.publish(None, status="dreaming")
            return
        self.publish(None, status="starting a new game (the bot plays the menus) ...")
        self.real = game_start(self.seed)
        frames, actions, layers, twos = start_context(self.real)
        for d in self.dreams:                              # a two-layer model starts from the two layers
            start = twos if getattr(d.world.model, "two_layers", False) else frames
            d.dreamer = d.world.dreaming(start, actions, layers if d.world.layered else None, d.keep,
                                         change_weight=self.change_weight, options=self.options)
            d.pressed = int(actions[-1])                   # the press that made the start's last frame
            d.player = Player(self.head, generator=self.choices) if self.head is not None else None
            d.dreamed, d.wrong, d.wrong_sum = 0, 0.0, 0.0
        self.publish(None, status="dreaming")

    def command(self, name: str, value=None) -> None:
        if name == "change_weight":
            self.change_weight = value
            for d in self.dreams:
                d.dreamer.set_change_weight(value)
        elif name == "controller":
            self.controller = value
            if value == "bot":
                self.real.new_bot()
        elif name == "options":                           # the inference settings: the game starts again
            self.options = dict(value)
            self.restart()
        elif name == "differences":
            self.differences = not self.differences
        elif name == "dream_start":
            self.dream_start = 2
        elif name == "restart":
            self.restart()
        elif name == "pause":
            self.paused = not self.paused
        elif name == "fps":
            self.fps_cap = value
        elif name == "quit":
            self.stopping = True

    def step(self, d: Dream, frame: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        """One dream's next frame -> its expected colours and each pixel's chance of being wrong."""
        likely, probs = d.dreamer.step(action)
        if probs is None:                                  # committed: wrong or not
            miss = (likely != frame).astype(np.float32)
            dream = self.palette[likely]
        else:
            real = torch.from_numpy(frame).cuda().long()
            miss = (1 - probs.gather(-1, real[..., None])[..., 0]).cpu().numpy()
            dream = (probs @ self.colours).round().clamp(0, 255).byte().cpu().numpy()
        d.wrong = float(miss[GAME_ROWS].mean())
        d.dreamed += 1
        d.wrong_sum += d.wrong
        return dream, miss

    def run(self) -> None:
        try:
            self.load()
            self.restart()
            fps_frames, fps_since, fps = 0, time.monotonic(), 0.0
            while not self.stopping:
                while not self.commands.empty():
                    self.command(*self.commands.get())
                if self.paused:
                    time.sleep(0.02)
                    continue
                started = time.monotonic()
                frame, action = self.real.step(None if self.controller in ("bot", "head") else self.buttons())
                start = START if self.dream_start else 0  # the dreams get Start, the real game does not
                self.dream_start = max(self.dream_start - 1, 0)
                for d in self.dreams:                      # each dream's press: the real game's, or its head's
                    d.pressed = (int(d.player.press(d.dreamer.newest(), torch.tensor([d.pressed], device="cuda"))[0])
                                 if self.controller == "head" else action) | start
                image = self.composer.compose(frame, [self.step(d, frame, d.pressed) for d in self.dreams],
                                              self.differences)
                fps_frames += 1
                if time.monotonic() - fps_since >= 1:
                    fps, fps_frames, fps_since = fps_frames / (time.monotonic() - fps_since), 0, time.monotonic()
                self.publish(image, fps=fps, action=action, controller=self.controller,
                             scores=[(d.dreamed, d.wrong, d.wrong_sum / max(d.dreamed, 1), d.pressed)
                                     for d in self.dreams])
                spare = 1 / self.fps_cap - (time.monotonic() - started)
                if spare > 0:
                    time.sleep(spare)
        except BaseException:
            crash(title="play_world_model")


class App:
    def __init__(self, root: tk.Tk, engine: Engine):
        self.root, self.engine = root, engine
        self.shown = 0
        root.title(f"{'Super Mario Bros.' if engine.game == 'smb' else 'Tetris'}: real game vs world models")
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        for name in ("TFrame", "TLabel", "TRadiobutton", "TCheckbutton"):
            style.configure(name, background=BG, foreground="#d8d8e0")
        style.configure("Status.TLabel", foreground="#a8a8b8")
        style.configure("TButton", padding=(10, 3))

        bar = ttk.Frame(root, padding=(10, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Change weight").pack(side="left")
        self.change_weight = tk.DoubleVar(value=engine.change_weight)
        ttk.Spinbox(bar, values=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0), width=4, textvariable=self.change_weight,
                    state="readonly", command=lambda: self.send("change_weight", self.change_weight.get())
                    ).pack(side="left", padx=(4, 10))
        ttk.Label(bar, text="Speed cap").pack(side="left")
        self.fps = tk.IntVar(value=engine.fps_cap)
        ttk.Spinbox(bar, values=(10, 20, 30, 45, 60), width=4, textvariable=self.fps, state="readonly",
                    command=lambda: self.send("fps", self.fps.get())).pack(side="left", padx=(6, 16))
        ttk.Label(bar, text="Controller").pack(side="left")
        self.controller = tk.StringVar(value=engine.controller)
        for text, value in (("Bot", "bot"), ("You", "you")) + ((("Head", "head"),) if engine.policy else ()):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.controller, takefocus=False,
                            command=lambda: self.send("controller", self.controller.get())).pack(side="left", padx=4)
        ttk.Button(bar, text="Restart (R)", takefocus=False,
                   command=lambda: self.send("restart")).pack(side="left", padx=(16, 4))
        ttk.Button(bar, text="Start the dreams (S)", takefocus=False,
                   command=lambda: self.send("dream_start")).pack(side="left", padx=4)
        ttk.Button(bar, text="Dreams / differences (D)", takefocus=False,
                   command=lambda: self.send("differences")).pack(side="left", padx=4)
        ttk.Button(bar, text="Pause (Space)", takefocus=False, command=lambda: self.send("pause")).pack(side="left", padx=4)

        inference = ttk.Frame(root, padding=(10, 0, 10, 6))
        inference.pack(fill="x")
        ttk.Label(inference, text="Inference (restarts the game):").pack(side="left")
        self.steps = tk.IntVar(value=engine.options["steps"])
        for text, value in (("1 pass (browser)", 1), ("4 Halton passes (slow, accurate)", 4)):
            ttk.Radiobutton(inference, text=text, value=value, variable=self.steps, takefocus=False,
                            command=self.send_options).pack(side="left", padx=4)
        self.hide = tk.BooleanVar(value=engine.options["past"] is not None)
        ttk.Checkbutton(inference, text="Hide unsure patches (< 0.99) from the history", variable=self.hide,
                        takefocus=False, command=self.send_options).pack(side="left", padx=(12, 4))

        c = engine.composer
        titles = ["Real game (World NES)"] + [run_label(r) for r in engine.runs]
        self.canvas = tk.Canvas(root, width=c.image.shape[1], height=c.image.shape[0] + 24, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(padx=10)
        for i, title in enumerate(titles):
            self.canvas.create_text(c.x(i) + c.panel // 2, 11, text=title, fill="#d8d8e0", width=c.panel,
                                    font=("Segoe UI", 10))
        self.photo = ImageTk.PhotoImage(Image.fromarray(c.image))
        self.canvas.create_image(0, 24, anchor="nw", image=self.photo)
        self.status = tk.StringVar(value="starting")
        ttk.Label(root, textvariable=self.status, style="Status.TLabel", padding=(10, 6), font=("Consolas", 10),
                  justify="left").pack(fill="x")
        ttk.Label(root, style="Status.TLabel", padding=(10, 0, 10, 8),
                  text=("Arrows move, X jump (A), Z run (B)" if engine.game == "smb" else
                        "Arrows move / soft-drop, X / Z rotate") + ", Enter = Start, Right Shift = Select (controller You)"
                       "   |   S Start into the dreams, R restart, D dreams / differences, Space pause, Esc quit").pack(fill="x")

        root.bind("<KeyPress>", lambda e: self.key(e, True))
        root.bind("<KeyRelease>", lambda e: self.key(e, False))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        tk_catch_callbacks(root)
        self.canvas.focus_set()
        self.poll()

    def send_options(self) -> None:
        self.send("options", {"steps": self.steps.get(), "past": 0.99 if self.hide.get() else None})

    def send(self, *command) -> None:
        self.engine.commands.put(command)
        self.canvas.focus_set()                                   # keys go to the game, not the toolbar

    def key(self, event, down: bool) -> None:
        if isinstance(event.widget, ttk.Spinbox):
            return
        if down and event.keysym == "Escape":
            self.quit()
        elif down and event.keysym == "space":
            self.send("pause")
        elif down and event.keysym in ("r", "R"):
            self.send("restart")
        elif down and event.keysym in ("d", "D"):
            self.send("differences")
        elif down and event.keysym in ("s", "S"):
            self.send("dream_start")
        elif event.keysym in KEYS:
            self.engine.press(KEYS[event.keysym], down)

    def poll(self) -> None:
        version, image, stats = self.engine.latest()
        if version != self.shown and image is not None:
            self.photo.paste(Image.fromarray(image))                # the one copy per frame
            self.shown = version
        if "fps" in stats:
            held = " ".join(n for n, b in BUTTON_NAMES if stats["action"] & b) or "-"
            lines = [f"{stats['fps']:.1f} fps   |   {stats['controller']}: {held}"
                     + ("   |   PAUSED" if self.engine.paused else "")
                     + (f"   |   {stats['status']}" if stats.get("status") not in (None, "dreaming") else "")]
            width = max(len(d.name) for d in self.engine.dreams)
            lines += [f"{d.name:{width}s}   {dreamed:>6,} frames   pixels wrong now {wrong:6.2%}   mean {mean:6.2%}"
                      + (f"   head presses: {' '.join(n for n, b in BUTTON_NAMES if pressed & b) or '-'}"
                         if stats["controller"] == "head" else "")
                      for d, (dreamed, wrong, mean, pressed) in zip(self.engine.dreams, stats["scores"])]
            self.status.set("\n".join(lines))
        else:
            self.status.set(stats.get("status", ""))
        self.root.after(15, self.poll)

    def quit(self) -> None:
        self.engine.commands.put(("quit",))
        self.root.after(100, self.root.destroy)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", default=None,
                   help="the models, run folders or checkpoint files, any steps; all dream at once "
                        "(default: the latest run with MIN_STEPS steps)")
    p.add_argument("--keep", type=int, default=None,
                   help="frames a layered or latent model keeps when its window slides (default: three quarters "
                        "of each model's window; the pixel model's history is a ring)")
    p.add_argument("--scale", type=int, default=None, help="pixels per NES pixel (default: what fits the screen)")
    p.add_argument("--seed", type=int, default=0, help="the game start (0: the browser's, scripts/export_onnx.py)")
    p.add_argument("--motion-steps", type=int, choices=(1, 2), default=1,
                   help="a motion-first model's motion decoding: 1 step (as the browser) or 2")
    p.add_argument("--game", choices=("tetris", "smb"), default="tetris")
    p.add_argument("--stage", default="1-1", help="smb: the level the game starts at (world-level)")
    p.add_argument("--policy", type=Path, default=None,
                   help="a policy head trained on the --runs model (scripts/policy_probe.py policy_head.pt): adds "
                        "the controller Head, each dream playing itself")
    args = p.parse_args()
    runs = [r.resolve() for r in args.runs] if args.runs else runs_with_checkpoints(args.game)[-1:]
    if not runs:
        p.error(f"no {args.game} run on this frame layout has {MIN_STEPS:,} steps yet; pass --runs")
    world, level = (int(v) - 1 for v in args.stage.split("-"))
    if args.game == "smb":
        from token_world.data.world_nes_smb import smb_palette
        palette = smb_palette().cpu().numpy().astype(np.uint8)
    else:
        palette = tetris_palette().cpu().numpy().astype(np.uint8)
    root = tk.Tk()
    panels = 1 + len(runs)
    scale = args.scale or max(1, min(3, (root.winfo_screenwidth() - 40) // (panels * (256 + 12))))
    engine = Engine(runs, args.keep, palette, scale, args.seed, args.motion_steps, args.game, (world, level),
                    args.policy)
    App(root, engine)
    engine.start()
    root.mainloop()


if __name__ == "__main__":
    main()
