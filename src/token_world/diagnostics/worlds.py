"""Any trained world model behind one interface, so every test and app works for every model.

A World generates frames from real ones: given real context (model frames, data/model_frames.py, and for
the layered models their layered frames, data/nes_layers.py) and the incoming buttons, it continues frame
by frame. Its frames are always 256 x 256 model frames, so every score made for model frames applies:

- the pixel model (models/dynamics.py) decodes soft: each frame comes with each pixel's colour
  probabilities, and scores are expectations;
- the layered models (models/layered_pixels.py, layered_slots.py) decode their layers and compose them
  (data/nes_layers.py compose); their frames are committed, and scores count pixels.

load(run) reads a run folder's (or a checkpoint's) model with its averaged weights. Tests: the scenario
checks (diagnostics/tetris_scenarios.py), long dreams (scripts/long_dream_check.py), the event checks
(scripts/event_checks.py: diagnostics/restart.py, line_clears.py), the play app (scripts/play_world_model.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from token_world.data.nes_layers import TRANSPARENT, compose, model_frame

ROOT = Path(__file__).resolve().parents[3]


class World:
    """A loaded model: name, step, layered (it needs layered context frames)."""

    def __init__(self, model, saved: dict, name: str, device):
        self.model, self.saved, self.name, self.device = model, saved, name, torch.device(device)
        self.step = int(saved["step"])
        self.layered = saved["args"].get("kind") == "layered"
        self.palette = (saved["palettes"]["tetris"] if isinstance(saved["palettes"], dict) else
                        saved["palettes"]).cpu().numpy().astype(np.uint8)

    def __repr__(self) -> str:
        return f"World({self.name} @ {self.step:,})"

    @torch.no_grad()
    def rollout(self, frames: np.ndarray, actions: np.ndarray, start: int, layers: dict | None = None,
                temperature: float = 1.0, seed: int = 0, batch: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """B windows: real model frames [B, T, 256, 256], incoming actions [B, T], and (layered models) their
        layered frames (each key [B, T, ...]) -> frames start..T-1 generated one after another from the real
        frames before `start` with the real buttons: (the frames, committed [B, T - start, 256, 256] uint8;
        each pixel's probability of the real frame's colour, and of the last context frame's colour, [B,
        T - start, 256, 256] float16: the pixel model's own, 0 or 1 for the committed models)."""
        made, right, same = [], [], []
        for i in range(0, len(frames), batch):
            rows = slice(i, i + batch)
            m, r, s = self._rollout(frames[rows], actions[rows], start,
                                    None if layers is None else {k: v[rows] for k, v in layers.items()},
                                    temperature, seed)
            if r is None:                                    # committed: the frame is right or it is not
                r = (m == frames[rows, start:]).astype(np.float16)
                s = (m == frames[rows, start - 1:start]).astype(np.float16)
            made.append(m)
            right.append(r)
            same.append(s)
        return np.concatenate(made), np.concatenate(right), np.concatenate(same)

    def _rollout(self, frames, actions, start, layers, temperature, seed):
        t = frames.shape[1]
        acts = torch.as_tensor(actions, device=self.device).long()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if not self.layered:
                x = torch.as_tensor(frames, device=self.device)
                decoder = self.model.decoder(len(x), self.device)
                decoder.prefill(x[:, :start], acts[:, :start])
                made, right, same = [], [], []
                for at in range(start, t):
                    probs = decoder.next(acts[:, at:at + 1], at)["probs"].float()
                    made.append(probs.argmax(-1).byte().cpu().numpy())
                    right.append(probs.gather(-1, x[:, at, ..., None].long())[..., 0].half().cpu().numpy())
                    same.append(probs.gather(-1, x[:, start - 1, ..., None].long())[..., 0].half().cpu().numpy())
                return np.stack(made, 1), np.stack(right, 1), np.stack(same, 1)
            if layers is None:
                raise ValueError(f"{self.name} is a layered model: it needs layered context frames")
            lay = {k: torch.as_tensor(np.asarray(v), device=self.device) for k, v in layers.items()}
            dreamer = self.model.Dreamer(self.model, {k: v[:, :start] for k, v in lay.items()}, acts[:, :start],
                                         temperature=temperature,
                                         generator=torch.Generator(device=self.device).manual_seed(seed),
                                         steps=self.saved["args"].get("dream_steps", 1))
            made = []
            for at in range(start, t):
                step = {k: v.cpu().numpy() for k, v in dreamer.step(acts[:, at]).items() if k != "probs"}
                made.append(np.stack([composed({k: v[b] for k, v in step.items()}) for b in range(len(frames))]))
            return np.stack(made, 1), None, None

    def dreaming(self, frames: np.ndarray, actions: np.ndarray, layers: dict | None = None, keep: int | None = None,
                 temperature: float = 1.0, seed: int = 0, change_weight: float = 1.0) -> "Dreaming":
        """One game, played live: from real context (model frames [T0, 256, 256], incoming actions [T0],
        layered frames for the layered models) -> a Dreaming whose step(action) makes the next frame.
        keep: frames kept when the window slides; change_weight: the pixel model's (models/dynamics.py
        weigh_changes)."""
        return Dreaming(self, frames, actions, layers, keep, temperature, seed, change_weight)

    def follow(self, frames: np.ndarray, actions: np.ndarray, layers: dict | None, future_actions: np.ndarray,
               seed: int = 0) -> np.ndarray:
        """One game's future dreamed from real context, fed the real buttons (the event checks: restarts, line
        clears) -> [len(future_actions), 256, 256] uint8, each pixel's most likely colour."""
        live = self.dreaming(frames, actions, layers, seed=seed)
        return np.stack([live.step(int(a))[0] for a in future_actions])


class Dreaming:
    """One live dream (the play app, the restart check): step(action) -> the next frame [256, 256] uint8 (each
    pixel's most likely colour) and, for the pixel model, its colour probabilities [256, 256, C] float32 on the
    World's device (None for the committed models)."""

    def __init__(self, world: World, frames, actions, layers, keep, temperature, seed, change_weight):
        self.world = world
        device = world.device
        acts = torch.as_tensor(np.asarray(actions), device=device).long()
        # the history is encoded as the steps run (bf16 on the GPU), so its cache holds what the steps write
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if not world.layered:
                from token_world.models.dynamics import Dreamer
                self.dreamer = Dreamer(world.model, torch.as_tensor(np.asarray(frames), device=device), acts,
                                       keep=keep, change_weight=change_weight)
            else:
                lay = {k: torch.as_tensor(np.asarray(v), device=device)[None] for k, v in layers.items()}
                self.dreamer = world.model.Dreamer(world.model, lay, acts[None], keep=keep, temperature=temperature,
                                                   generator=torch.Generator(device=device).manual_seed(seed),
                                                   steps=world.saved["args"].get("dream_steps", 1))

    def set_change_weight(self, weight: float) -> None:
        """The pixel model's change weight from the next frame on (models/dynamics.py Dreamer); the committed
        models have none."""
        if not self.world.layered:
            self.dreamer.set_change_weight(weight)

    @torch.no_grad()
    def step(self, action: int) -> tuple[np.ndarray, torch.Tensor | None]:
        device = self.world.device
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if not self.world.layered:
                probs = self.dreamer.step(action).float()
                return probs.argmax(-1).byte().cpu().numpy(), probs
            frame = self.dreamer.step(torch.tensor([action], device=device))
            return composed({k: v[0].cpu().numpy() for k, v in frame.items() if k != "probs"}), None


def composed(frame: dict) -> np.ndarray:
    """A layered frame -> its model frame [256, 256] (TRANSPARENT where nothing at all is drawn: colour 0)."""
    out = model_frame(compose(frame), frame["border"])
    return np.where(out == TRANSPARENT, 0, out).astype(np.uint8)


def run_folder(run: str | Path) -> Path:
    """A run folder, or a checkpoint's run, where the tests' summaries go (scenarios.csv, restart.csv): an
    archived checkpoint (<archive>/<run>/model_stepN.pt) belongs to output/<run>."""
    run = Path(run)
    if run.is_dir():
        return run
    return ROOT / "output" / run.parent.name if (ROOT / "output" / run.parent.name).is_dir() else run.parent


def load(run: str | Path, device: str = "cuda") -> World:
    """A run folder (its model_latest.pt) or a checkpoint file -> its World, with averaged weights."""
    path = Path(run)
    checkpoint = path / "model_latest.pt" if path.is_dir() else path
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    name = path.name if path.is_dir() else f"{path.parent.name}/{path.stem}"
    if saved["args"].get("kind") == "layered":
        from token_world.models.layered import build
        model = build(saved["args"]).to(device).eval()
    else:
        from token_world.data.model_frames import BORDER_VERSION
        from token_world.models.dynamics import build
        if saved["args"].get("border") != BORDER_VERSION:
            raise ValueError(f"{run} was trained on another frame layout than {BORDER_VERSION}")
        model = build(saved["args"]).to(device).eval()
    model.load_state_dict(saved["ema"])
    return World(model, saved, name, device)
