"""Tetris next to its world model: the real game and one model's dream, frame by frame in lockstep.

    python scripts/play_world_model.py [--run output/world_model_tetris_base]

The real game runs in World NES. Every frame one controller byte (your keys,
or the bot) goes to both the console and the model, any model: the pixel model
or a layered one (diagnostics/worlds.py). The dream starts from the last 48
real frames (and their layers); after that it sees only the buttons. One model
generates at a time; pick it in the toolbar.

A worker thread owns the console, the bot and the model, and composes real,
dream and their difference into one image per frame; the UI copies that
image to its canvas once per frame. Frames advance at up to the speed cap
(60 = the NES), or as fast as the model generates.

The pixel model's dream is soft (models/dynamics.py): each pixel is a colour
distribution, shown as its expected colour, and the difference panel shows how
likely each pixel is to be wrong. A layered model's dream is committed: its
layers composed, and the difference panel shows the wrong pixels. The toolbar
sets how many frames are kept when the window slides (a change starts a new
dream) and the pixel model's change weight, at once: the odds of every pixel
change times it (models/dynamics.py weigh_changes), so an unsure piece is drawn
with too many cells instead of thin ones.

Keys: arrows move and soft-drop, X / Z rotate, Enter is Start, Right Shift is
Select. R restarts the dream from the real game, Space pauses, Esc quits.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from token_world.data.model_frames import BORDER_VERSION, GAME_ROWS
from token_world.data.tetris_bot import A, B, DOWN, LEFT, RIGHT, SELECT, START, UP
from token_world.data.nes_palette import tetris_palette
from token_world.data.real_tetris import RealTetris
from token_world.diagnostics.worlds import World, load

KEYS = {"Left": LEFT, "Right": RIGHT, "Down": DOWN, "Up": UP, "x": A, "X": A, "z": B, "Z": B,
        "Return": START, "Shift_R": SELECT}
BUTTON_NAMES = (("A", A), ("B", B), ("Select", SELECT), ("Start", START), ("Up", UP), ("Down", DOWN),
                ("Left", LEFT), ("Right", RIGHT))
BG = "#15151c"


MIN_STEPS = 20_000                              # long-trained runs only


def runs_with_checkpoints() -> list[Path]:
    """Long-trained Tetris world-model runs (train_dynamics_ui.py, train_layered.py) on the current frame
    layout: a checkpoint, a run config naming the game and BORDER_VERSION (or a layered model's), and at
    least MIN_STEPS steps."""
    runs = []
    for checkpoint in (ROOT / "output").glob("*/model_latest.pt"):
        run = checkpoint.parent
        config, metrics = run / "run_config.json", run / "metrics.csv"
        if not (config.is_file() and metrics.is_file()):
            continue
        c = json.loads(config.read_text())
        if (c.get("border") != BORDER_VERSION and c.get("kind") != "layered") or "tetris" not in c.get("games", []):
            continue
        last = metrics.read_text().strip().splitlines()[-1].split(",")[0]
        if last.isdigit() and int(last) >= MIN_STEPS:
            runs.append(run)
    return sorted(runs)


def load_model(run: Path) -> tuple[World, str]:
    world = load(run)
    params = sum(p.numel() for p in world.model.parameters()) / 1e6
    kind = "layered, committed" if world.layered else "soft"
    return world, f"{run.name} @ step {world.step:,} ({params:.1f}M, EMA, {kind})"


class Composer:
    """real | dream | difference, scaled, into one preallocated RGB image."""

    def __init__(self, palette: np.ndarray, scale: int, gap: int = 12):
        self.palette, self.scale, self.gap = palette, scale, gap
        self.panel = 256 * scale
        self.image = np.empty((self.panel, 3 * self.panel + 2 * gap, 3), np.uint8)
        self.image[:] = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)

    def x(self, i: int) -> int:
        return i * (self.panel + self.gap)

    def put(self, i: int, rgb: np.ndarray) -> None:
        s = self.scale
        self.image[:, self.x(i):self.x(i) + self.panel] = rgb.repeat(s, 0).repeat(s, 1)

    def compose(self, real: np.ndarray, dream: np.ndarray, wrong: np.ndarray) -> np.ndarray:
        """real [256, 256] palette indices, dream [256, 256, 3] expected colours, wrong [256, 256] each
        pixel's probability of anything but the real colour."""
        real_rgb = self.palette[real]
        self.put(0, real_rgb)
        self.put(1, dream)
        w = wrong[..., None]
        self.put(2, (real_rgb * 0.3 * (1 - w) + np.array([255, 0, 200]) * w).astype(np.uint8))
        return self.image


class Engine(threading.Thread):
    """Runs the real game and the dream in lockstep; publishes one composed image per frame."""

    def __init__(self, run: Path, keep: int, palette: np.ndarray, scale: int, seed: int):
        super().__init__(daemon=True)
        self.run_path, self.keep, self.seed = run, keep, seed
        self.palette = palette
        self.colours = torch.as_tensor(palette, dtype=torch.float32, device="cuda")
        self.composer = Composer(palette, scale)
        self.commands: queue.Queue = queue.Queue()
        self.keys: set[int] = set()
        self.keys_lock = threading.Lock()
        self.lock = threading.Lock()
        self.version, self.image, self.stats = 0, None, {"status": "starting"}
        self.controller, self.paused, self.fps_cap, self.stopping = "you", False, 60, False
        self.change_weight = 1.0
        self.world = self.dreamer = None
        self.model_name = ""

    # -- called from the UI thread ------------------------------------------------------------
    def buttons(self) -> int:
        with self.keys_lock:
            return sum(self.keys)

    def press(self, button: int, down: bool) -> None:
        with self.keys_lock:
            (self.keys.add if down else self.keys.discard)(button)

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

    def load(self, run: Path) -> None:
        """Swap in another model; on failure keep the current one and say why."""
        self.publish(None, status=f"loading {run.name} ...")
        try:
            world, name = load_model(run)
        except Exception as exc:
            self.publish(None, status=f"could not load {run.name}: {exc}")
            return
        self.world = self.dreamer = None
        torch.cuda.empty_cache()
        self.world, self.model_name, self.run_path = world, name, run
        self.restart()

    def restart(self) -> None:
        """A new dream from the last CONTEXT real frames (and their layers)."""
        frames, actions = np.stack(self.real.frames), np.array(self.real.actions, np.int64)
        layers = ({k: np.stack([f[k] for f in self.real.layers]) for k in self.real.layers[0]}
                  if self.world.layered else None)
        self.dreamer = self.world.dreaming(frames, actions, layers, self.keep, change_weight=self.change_weight)
        self.dreamed, self.wrong_sum = 0, 0.0
        self.publish(None, status="dreaming", model=self.model_name)

    def command(self, name: str, value=None) -> None:
        if name == "model":
            self.load(value)
        elif name == "decoding":
            self.keep = value
            self.restart()
        elif name == "change_weight":
            self.change_weight = value
            if self.dreamer is not None:
                self.dreamer.set_change_weight(value)
        elif name == "controller":
            self.controller = value
            if value == "bot":
                self.real.new_bot()
        elif name == "restart":
            self.restart()
        elif name == "pause":
            self.paused = not self.paused
        elif name == "fps":
            self.fps_cap = value
        elif name == "quit":
            self.stopping = True

    def run(self) -> None:
        try:
            self.publish(None, status="booting the game (the bot plays until a piece is falling) ...")
            self.real = RealTetris(self.seed, layered=True)
            played = 0
            while not (played > 600 and self.real.playing()):
                self.real.step(None)
                played += 1
            self.load(self.run_path)
            fps_frames, fps_since, fps = 0, time.monotonic(), 0.0
            while not self.stopping:
                while not self.commands.empty():
                    self.command(*self.commands.get())
                if self.paused:
                    time.sleep(0.02)
                    continue
                started = time.monotonic()
                frame, action = self.real.step(None if self.controller == "bot" else self.buttons())
                likely, probs = self.dreamer.step(action)
                if probs is None:                                  # committed: wrong or not
                    miss = (likely != frame).astype(np.float32)
                    dream = self.palette[likely]
                else:
                    real = torch.from_numpy(frame).cuda().long()
                    miss = (1 - probs.gather(-1, real[..., None])[..., 0]).cpu().numpy()
                    dream = (probs @ self.colours).round().clamp(0, 255).byte().cpu().numpy()
                image = self.composer.compose(frame, dream, miss)
                wrong = float(miss[GAME_ROWS].mean())
                self.dreamed += 1
                self.wrong_sum += wrong
                fps_frames += 1
                if time.monotonic() - fps_since >= 1:
                    fps, fps_frames, fps_since = fps_frames / (time.monotonic() - fps_since), 0, time.monotonic()
                self.publish(image, fps=fps, action=action, wrong=wrong, mean_wrong=self.wrong_sum / self.dreamed,
                             dreamed=self.dreamed, controller=self.controller)
                spare = 1 / self.fps_cap - (time.monotonic() - started)
                if spare > 0:
                    time.sleep(spare)
        except Exception as exc:
            traceback.print_exc()
            self.publish(None, status=f"error: {exc}")


class App:
    def __init__(self, root: tk.Tk, engine: Engine, runs: list[Path]):
        self.root, self.engine, self.runs = root, engine, runs
        self.shown = 0
        root.title("Tetris: real game vs world model")
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        for name in ("TFrame", "TLabel", "TRadiobutton", "TCheckbutton"):
            style.configure(name, background=BG, foreground="#d8d8e0")
        style.configure("Status.TLabel", foreground="#a8a8b8")
        style.configure("TButton", padding=(10, 3))

        bar = ttk.Frame(root, padding=(10, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="Model").pack(side="left")
        self.model = ttk.Combobox(bar, state="readonly", width=34, values=[r.name for r in runs])
        self.model.set(engine.run_path.name)
        self.model.bind("<<ComboboxSelected>>", lambda e: self.send("model", runs[self.model.current()]))
        self.model.pack(side="left", padx=(6, 16))
        self.keep = tk.IntVar(value=engine.keep)
        ttk.Label(bar, text="Keep").pack(side="left")
        ttk.Spinbox(bar, values=tuple(sorted({engine.keep // 2, engine.keep * 2 // 3, engine.keep,
                                              engine.keep * 4 // 3 - 1})),
                    width=4, textvariable=self.keep, state="readonly",
                    command=self.decoding).pack(side="left", padx=(4, 10))
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
        for label, value in (("You", "you"), ("Bot", "bot")):
            ttk.Radiobutton(bar, text=label, value=value, variable=self.controller, takefocus=False,
                            command=lambda: self.send("controller", self.controller.get())).pack(side="left", padx=4)
        ttk.Button(bar, text="Restart dream (R)", takefocus=False,
                   command=lambda: self.send("restart")).pack(side="left", padx=(16, 4))
        self.pause_button = ttk.Button(bar, text="Pause (Space)", takefocus=False, command=lambda: self.send("pause"))
        self.pause_button.pack(side="left", padx=4)

        c = engine.composer
        titles = ("Real game (World NES)", "World model dream", "Difference (magenta = wrong pixels)")
        self.canvas = tk.Canvas(root, width=c.image.shape[1], height=c.image.shape[0] + 24, bg=BG,
                                highlightthickness=0)
        self.canvas.pack(padx=10)
        for i, title in enumerate(titles):
            self.canvas.create_text(c.x(i) + c.panel // 2, 11, text=title, fill="#d8d8e0", font=("Segoe UI", 11))
        self.photo = ImageTk.PhotoImage(Image.fromarray(c.image))
        self.canvas.create_image(0, 24, anchor="nw", image=self.photo)
        self.status = tk.StringVar(value="starting")
        ttk.Label(root, textvariable=self.status, style="Status.TLabel", padding=(10, 6)).pack(fill="x")
        ttk.Label(root, style="Status.TLabel", padding=(10, 0, 10, 8),
                  text="Arrows move / soft-drop, X / Z rotate, Enter = Start, Right Shift = Select   "
                       "|   R restart dream, Space pause, Esc quit").pack(fill="x")

        root.bind("<KeyPress>", lambda e: self.key(e, True))
        root.bind("<KeyRelease>", lambda e: self.key(e, False))
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self.canvas.focus_set()
        self.poll()

    def decoding(self) -> None:
        """Toolbar decoding setting changed: a new dream with it (Dreamer keep)."""
        self.send("decoding", self.keep.get())

    def send(self, *command) -> None:
        self.engine.commands.put(command)
        self.canvas.focus_set()                                   # keys go to the game, not the toolbar

    def key(self, event, down: bool) -> None:
        if isinstance(event.widget, (ttk.Combobox, ttk.Spinbox)):
            return
        if down and event.keysym == "Escape":
            self.quit()
        elif down and event.keysym == "space":
            self.send("pause")
        elif down and event.keysym in ("r", "R"):
            self.send("restart")
        elif event.keysym in KEYS:
            self.engine.press(KEYS[event.keysym], down)

    def poll(self) -> None:
        version, image, stats = self.engine.latest()
        if version != self.shown and image is not None:
            self.photo.paste(Image.fromarray(image))                # the one copy per frame
            self.shown = version
        if "fps" in stats:
            held = " ".join(n for n, b in BUTTON_NAMES if stats["action"] & b) or "-"
            self.status.set(
                f"{stats.get('model', '')}   |   {stats['fps']:.1f} fps   |   "
                f"{stats['dreamed']:,} frames dreamed, pixels wrong now {stats['wrong']:.2%}, "
                f"mean {stats['mean_wrong']:.2%}   |   {stats['controller']}: {held}"
                + ("   |   PAUSED" if self.engine.paused else "")
                + (f"   |   {stats['status']}" if stats.get("status") not in (None, "dreaming") else ""))
        else:
            self.status.set(stats.get("status", ""))
        self.root.after(15, self.poll)

    def quit(self) -> None:
        self.engine.commands.put(("quit",))
        self.root.after(100, self.root.destroy)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs = runs_with_checkpoints()
    p.add_argument("--run", type=Path, default=None,
                   help="a run folder, any length (default: the latest run with MIN_STEPS steps)")
    p.add_argument("--keep", type=int, default=None,
                   help="frames kept when the window slides (default: three quarters of the run's window)")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--seed", type=int, default=int(time.time()) % 100_000)
    args = p.parse_args()
    if args.run is None and not runs:
        p.error(f"no run on this frame layout has {MIN_STEPS:,} steps yet; pass --run")
    run = args.run.resolve() if args.run else runs[-1]
    if run not in runs:
        runs.append(run)
    palette = tetris_palette().cpu().numpy().astype(np.uint8)
    keep = args.keep or json.loads((run / "run_config.json").read_text())["frames"] * 3 // 4
    engine = Engine(run, keep, palette, args.scale, args.seed)
    root = tk.Tk()
    App(root, engine, runs)
    engine.start()
    root.mainloop()


if __name__ == "__main__":
    main()
