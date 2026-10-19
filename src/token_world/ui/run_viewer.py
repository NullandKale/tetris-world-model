"""World-model run viewer: every chart, preview and table read back from run folders, refreshed as they change.

The trainers (scripts/train_dynamics_ui.py, train_latent_ui.py, train_layered.py, train_vae_ui.py) write
everything a run has to its folder: metrics.csv (and metrics_schema_*.csv, earlier columns), previews/*.gif
(animated: REAL | GENERATED | WRONG, and SPRITES for a VAE with a sprite output), long_dream.csv/.gif/.png (the
long-dream test every 2,000 steps: no buttons, and the bot's buttons), scenarios.csv and event_checks.csv (checks
run by hand). The viewer reads only those files, so the same window serves a training run (ui/dynamics_app.py adds
the status line and Stop Training) and any finished or running run beside it (scripts/view_runs.py); the phone
page (scripts/web_runs.py) draws the same charts and tables. Runs given as `compare` are drawn dashed, by step.

Tabs, each with a paragraph on what it shows (EXPLAIN): Curves, Previews, Long dream (animation or the sheet,
and the scores with the plateau check), Compare, Scenarios, Data, Horizons, Speed, Stats. A tokenizer's runs
(run_kind 'vae') hide the tabs only world models have. Images fit their space and tables scroll
(ui/viewer_widgets.py); charts lay themselves out again at every size.
"""
from __future__ import annotations

import csv
import io
import json
import math
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter

from token_world.diagnostics import plateau
from token_world.ui.viewer_widgets import Explainer, FitGrid, FitImage, TextPanel, split

BG, AXES, GRID, TEXT = "#131316", "#181820", "#333342", "#b0b0c0"
COLOURS = ("#72dce0", "#ff9e00", "#e0aaff", "#88d990", "#ff4d8d", "#f4d35e")
DASHES = ("-", "--", ":", "-.")
GAME = "tetris"


# ---------------------------------------------------------------------------------------------- data
def _read_rows(path: Path) -> list[dict]:
    """A CSV as dicts, read in one go and closed (the trainer may be appending or replacing it)."""
    try:
        text = path.read_bytes().decode("utf-8", "replace")
    except OSError:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def load_metrics(run: Path) -> dict[str, np.ndarray]:
    """Every metrics*.csv of a run (archived column sets too), merged by step -> column: values over the
    sorted steps, NaN where a file lacks the column."""
    by_step: dict[int, dict] = {}
    for path in sorted(run.glob("metrics*.csv"), key=lambda p: p.stat().st_mtime):
        for row in _read_rows(path):
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                continue
            by_step.setdefault(step, {}).update({k: v for k, v in row.items() if v not in ("", None)})
    steps = sorted(by_step)
    columns = sorted({k for row in by_step.values() for k in row})
    out = {}
    for c in columns:
        values = []
        for s in steps:
            try:
                values.append(float(by_step[s].get(c, "nan")))
            except ValueError:
                values.append(math.nan)
        out[c] = np.array(values)
    for c in [c for c in out if c.endswith("_wrong_changed") and c[:-len("wrong_changed")] + "changed" in out]:
        out[c][out[c[:-len("wrong_changed")] + "changed"] == 0] = math.nan    # nothing changed: older rows say 0
    return out


def load_long_dream(run: Path) -> dict[str, dict[str, np.ndarray]]:
    """Every long_dream*.csv (archived column sets too) -> variant -> column -> per-test means over the
    tested steps (with "step"); NaN where a test lacks the column."""
    rows = [r for path in sorted(run.glob("long_dream*.csv")) for r in _read_rows(path)]
    value = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else math.nan
    out: dict[str, dict[str, np.ndarray]] = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        mine = [r for r in rows if r["variant"] == variant]
        steps = sorted({int(r["step"]) for r in mine})
        keys = list(dict.fromkeys(k for r in mine for k in r if k not in ("step", "variant", "level")))
        out[variant] = {"step": np.array(steps, float)} | {
            k: np.array([_nanmean([value(r, k) for r in mine if int(r["step"]) == s]) for s in steps]) for k in keys}
    # the bias tests (diagnostics/bias.py, one row a test) as bias_<column>, at the same tested steps
    tests = [r for r in _read_rows(run / "bias.csv")] if (run / "bias.csv").exists() else []
    for variant, d in out.items():
        mine = {int(r["step"]): r for r in tests if r["variant"] == variant}
        for key in (k for k in (tests[0] if tests else {}) if k not in ("step", "variant")):
            d[f"bias_{key}"] = np.array([value(mine[int(s)], key) if int(s) in mine else math.nan for s in d["step"]])
    return out


def pressed(variant: str) -> bool:
    """A long-dream variant dreamed with the bot's buttons (train_dynamics_ui.py PRESSED_VARIANT,
    train_layered.py variants), not with none."""
    return variant.startswith("buttons")


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    """A trailing mean over the last `window` checkpoints, NaNs ignored."""
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1):i + 1]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = chunk.mean()
    return out


class RunData:
    """One run folder's files, reloaded when any of them changes."""

    def __init__(self, path: Path):
        self.path, self.name, self.stamp = path, path.name, None
        self.metrics: dict[str, np.ndarray] = {}
        self.long_dream: dict[str, dict[str, np.ndarray]] = {}

    def refresh(self) -> bool:
        files = [*self.path.glob("metrics*.csv"), *self.path.glob("long_dream*.csv"), *self.path.glob("bias.csv"),
                 *self.path.glob("scenarios.csv"), *self.path.glob("event_checks.csv")]
        stamp = tuple(sorted((f.name, f.stat().st_mtime) for f in files if f.exists()))
        if stamp == self.stamp:
            return False
        self.stamp, self.metrics, self.long_dream = stamp, load_metrics(self.path), load_long_dream(self.path)
        return True


# ---------------------------------------------------------------------------------------------- charts
def _style(ax, title: str, ylabel: str = "", log: bool = False) -> None:
    ax.set_facecolor(AXES)
    ax.set_title(textwrap.fill(title, 64), color="#e2e2ec", fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=GRID, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())             # labels at the decades only


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(facecolor="#1f1f28", labelcolor="#e2e2ec", fontsize=6, loc="best", framealpha=0.55)


def _plot(ax, run: RunData, key: str, label: str | None, colour: str, dash: str, raw: bool = True) -> None:
    """Smoothed (bold) over raw (faint, solid runs only); only the first run's lines carry a label."""
    m = run.metrics
    if key not in m or "step" not in m or not np.isfinite(m[key]).any():
        return
    if raw and dash == "-":
        ax.plot(m["step"], m[key], color=colour, alpha=0.18, linewidth=0.7)
    ax.plot(m["step"], smooth(m[key]), color=colour, linestyle=dash, linewidth=1.4, label=label)


CURVE_VIEWS = {"dreams": "Long dreams", "live": "Live events", "training": "Training"}   # draw_curves' views
DREAM_TREND = 5                     # long-dream tests in a trend point: the last 10,000 steps (one every 2,000)
LIVE_TREND = 10                     # metrics rows in a live trend point: the last 2,000 steps (one every 200)
DREAM_SCORES = (("wrong_128", "pixels wrong at +128 (lower better)"),
                ("shape", "piece shape (1 = real)"),
                ("lost_long", "breaks: cells lost / 100 frames (lower better)"),
                ("morph_long", "shape changes / 100 frames (lower better)"),
                ("full_long", "full rows never cleared / 100 frames"),
                ("spawn", "spawns on time (1 = real)"),
                ("piece_hit_32", "piece on the real one at +32 (1 = real)"),
                ("mass", "block mass at +128 (1 = real)"),
                ("presence", "a piece present (0 = stalling)"),
                ("fall", "fall speed (1 = real)"))
BASELINES = Path(__file__).resolve().parents[3] / "output" / "trends" / "baseline_newbot.csv"
REFERENCES = (("actnorm 72k", "world_model_tetris_rope_actnorm/model_step72000.pt", "#f4d35e"),
              ("resync 18k", "world_model_tetris_actnorm_resync/model_step18000.pt", "#ff4d8d"))
RUN_COLOURS = ("#72dce0", "#ff9e00", "#e0aaff", "#88d990")      # the references keep the other two


def load_references(path: Path = BASELINES) -> dict[bool, list[tuple[str, str, dict[str, float]]]]:
    """The best models' long-dream scores on the trainer's trials (baseline_newbot.csv: actnorm 72k, resync 18k)
    -> pressed or not -> (label, colour, score -> value), for the flat reference lines."""
    out: dict[bool, list] = {False: [], True: []}
    rows = _read_rows(path) if path.exists() else []
    for label, ending, colour in REFERENCES:
        for r in rows:
            if r.get("ckpt", "").replace("\\", "/").endswith(ending):
                scores = {}
                for k, v in r.items():
                    try:
                        scores[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                out[pressed(r["variant"])].append((label, colour, scores))
    return out


def weighted_trend(values: np.ndarray, counts: np.ndarray, window: int = LIVE_TREND) -> np.ndarray:
    """A trailing mean over the last `window` rows, each row weighted by its count (rows with none skipped)."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        v, c = values[max(0, i - window + 1):i + 1], counts[max(0, i - window + 1):i + 1]
        ok = np.isfinite(v) & np.isfinite(c) & (c > 0)
        if ok.any():
            out[i] = float((v[ok] * c[ok]).sum() / c[ok].sum())
    return out


def _trend(ax, steps, values, trend, colour: str, dash: str, label: str | None) -> None:
    """Each test or row as a faint dot, its trend as the line."""
    ok = np.isfinite(values)
    if not ok.any():
        return
    ax.plot(steps[ok], values[ok], linestyle="none", marker=".", markersize=3, color=colour, alpha=0.3)
    ax.plot(steps, trend, color=colour, linestyle=dash, linewidth=1.5, label=label)


def _dream_panels(fig: Figure, every: list[tuple[RunData, str]]) -> None:
    """One pair of panels per long-dream score, no buttons then the bot's buttons on one y scale; each run's
    trend over its last DREAM_TREND tests, every test a faint dot; the best models flat."""
    references = load_references()
    axes = fig.subplots(math.ceil(len(DREAM_SCORES) / 2), 4)
    for s, (key, title) in enumerate(DREAM_SCORES):
        pair = axes[s // 2, 2 * (s % 2):2 * (s % 2) + 2]
        pair[1].sharey(pair[0])
        for ax, buttons in zip(pair, (False, True)):
            for n, (run, dash) in enumerate(every):
                for variant, d in run.long_dream.items():
                    if pressed(variant) == buttons and key in d:
                        label = run.name if len([v for v in run.long_dream if pressed(v) == buttons]) == 1 \
                            else f"{run.name} {variant}"
                        _trend(ax, d["step"], d[key], smooth(d[key], DREAM_TREND), RUN_COLOURS[n % len(RUN_COLOURS)],
                               dash, label if s == 0 else None)
            for label, colour, scores in references[buttons]:
                if math.isfinite(scores.get(key, math.nan)):
                    ax.axhline(scores[key], color=colour, linestyle=(0, (3, 2)), linewidth=1.0,
                               label=label if s == 0 else None)
            _style(ax, f"{title}: {'buttons' if buttons else 'no buttons'}")
            ax.title.set_fontsize(8)
    for ax in axes.flat[2 * len(DREAM_SCORES):]:
        ax.set_visible(False)
    _legend(axes[0, 0])


def _live_panels(fig: Figure, every: list[tuple[RunData, str]]) -> None:
    """The training batches' own event scores (train_dynamics_ui.py live_event_scores): one panel per event, the
    share of its frames drawn exactly and its changed pixels' wrong share; then the J / L / T turns with the real
    press and with A and B swapped, and the gap between them."""
    from token_world.data.tetris_events import EVENTS, event_slug
    axes = fig.subplots(4, 4)
    for e, name in enumerate(EVENTS):
        ax = axes.flat[e]
        total = 0.0
        for run, dash in every:
            m = run.metrics
            counts = m.get(f"live_{event_slug(name)}_n")
            if counts is None or "step" not in m:
                continue
            total += float(np.nansum(counts)) if run is every[0][0] else 0.0
            for key, text, colour in (("exact", "frames exact", COLOURS[0]), ("wrong", "changed pixels wrong",
                                                                              COLOURS[1])):
                values = m.get(f"live_{event_slug(name)}_{key}")
                if values is not None:
                    _trend(ax, m["step"], values, weighted_trend(values, counts), colour, dash,
                           text if e == 0 and run is every[0][0] else None)
        _style(ax, f"{name} (n = {total:,.0f})")
        ax.title.set_fontsize(8)
    turn, gap = axes.flat[len(EVENTS)], axes.flat[len(EVENTS) + 1]
    for run, dash in every:
        m = run.metrics
        counts = m.get("live_turn_n")
        if counts is None or "step" not in m:
            continue
        first = run is every[0][0]
        real, swapped = m["live_turn_exact"], m["live_turn_exact_swapped"]
        _trend(turn, m["step"], real, weighted_trend(real, counts), COLOURS[0], dash, "real press" if first else None)
        _trend(turn, m["step"], swapped, weighted_trend(swapped, counts), COLOURS[4], dash,
               "A and B swapped" if first else None)
        _trend(gap, m["step"], m["live_turn_gap"], weighted_trend(real, counts) - weighted_trend(swapped, counts),
               COLOURS[3], dash, None)
    gap.axhline(0.0, color=TEXT, linewidth=0.8, linestyle=":")
    _style(turn, "J / L / T turns: frames exact")
    _style(gap, "turn gap: real press less swapped (0 = ignores A vs B)")
    for ax in (turn, gap):
        ax.title.set_fontsize(8)
    for ax in axes.flat[len(EVENTS) + 2:]:
        ax.set_visible(False)
    _legend(axes.flat[0])
    _legend(turn)


def _training_panels(fig: Figure, every: list[tuple[RunData, str]], runs: list[RunData]) -> None:
    """The training loss, the previews' wrong pixels and border, and the motion head's moved / turned cells."""
    axes = fig.subplots(2, 2)
    moving = any(np.isfinite(r.metrics.get("train_motion_moved", np.array([np.nan]))).any() for r in runs)
    for n, (run, dash) in enumerate(every):
        label = (lambda text: text) if n == 0 else (lambda text: None)
        _plot(axes[0, 0], run, "train_loss_changed", label("changed pixels"), COLOURS[0], dash)
        _plot(axes[0, 0], run, "train_loss_border", label("border"), COLOURS[1], dash)
        _plot(axes[0, 0], run, "train_loss_motion", label("motion"), COLOURS[2], dash)
        for i, part in enumerate(("cells", "sprites", "sprite_move", "sprite_place", "sprite_pixels", "camera")):
            _plot(axes[0, 0], run, f"train_loss_{part}", label(part.replace("_", " ")), COLOURS[i], dash)
        for i, h in enumerate((1, 4, 16)):
            _plot(axes[0, 1], run, f"{GAME}_h{h}_wrong_changed", label(f"+{h}"), COLOURS[i], dash)
        for i, h in enumerate((1, 16)):
            _plot(axes[1, 0], run, f"{GAME}_h{h}_border_wrong", label(f"+{h}"), COLOURS[i], dash)
            if n < len(runs):
                _plot(axes[1, 0], run, f"{GAME}_h{h}_border_copy", label(f"+{h} copying"), COLOURS[i], ":",
                      raw=False)
        if moving:                            # a motion model: the share of moved / turned cells right
            for i, (key, text) in enumerate((("train_motion_moved", "moved, masked"), ("train_motion_turned", "turned, masked"),
                                             ("preview_motion_moved", "moved, own frames"),
                                             ("preview_motion_turned", "turned, own frames"))):
                _plot(axes[1, 1], run, key, label(text), COLOURS[i], dash)
        else:
            _plot(axes[1, 1], run, f"{GAME}_h16_wrong_static", label("static +16"), COLOURS[3], dash)
            _plot(axes[1, 1], run, "blend_frames", label("blend radius"), COLOURS[4], dash)
    _style(axes[0, 0], "Train loss (masked pixels)", "cross-entropy", log=True)
    _style(axes[0, 1], "Preview: changed game pixels wrong (copying = 100%)", "fraction")
    _style(axes[1, 0], "Preview: border pixels wrong vs copying the last frame (thin dotted)", "fraction", log=True)
    if moving:
        _style(axes[1, 1], "Motion: the share of real moved / turned cells drawn right, masked in training and in "
                           "its own preview frames (a turned cell moves along both axes)", "fraction")
    else:
        _style(axes[1, 1], "Static pixels wrong at +16; blend radius (frames)", "fraction", log=True)
    for ax in axes.flat:
        _legend(ax)


def draw_curves(fig: Figure, runs: list[RunData], compare: list[RunData], view: str = "dreams") -> None:
    """One of CURVE_VIEWS over training: the long-dream scores (dreams), the training batches' own event scores
    (live) or the loss, previews and motion (training). Runs solid, compared runs dashed."""
    fig.clear()
    fig.set_facecolor(BG)
    every = [(r, "-") for r in runs] + [(r, DASHES[1 + i % 3]) for i, r in enumerate(compare)]
    if view == "dreams":
        _dream_panels(fig, every)
        how = (f"line: mean of the last {DREAM_TREND} tests, dots: each test; dashed flat lines: "
               + ", ".join(label for label, _, _ in REFERENCES))
    elif view == "live":
        _live_panels(fig, every)
        how = f"line: the last {LIVE_TREND} rows' mean (weighted by events), dots: each row (200 steps)"
    else:
        _training_panels(fig, every, runs)
        how = "faint: each row, bold: mean of the last 9"
    names = "   ".join(f"{'solid' if dash == '-' else {'--': 'dashed', ':': 'dotted', '-.': 'dash-dot'}[dash]}: "
                       f"{run.name}" for run, dash in every)
    fig.suptitle(f"{names}     ({how})", color=TEXT, fontsize=9)
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))


def fit_steps(fig: Figure, runs: list[RunData], view: str = "dreams", own: bool = True) -> None:
    """Every panel's steps from the runs' first (for the live view: their first live scores; a grown run's
    start, not its compared runs' early training) to their last (`own`; compared runs often reach far
    further) or every line's last, and its values scaled to what is left in view (over the panels sharing its
    y scale)."""
    steps = [r.metrics["step"] for r in runs if len(r.metrics.get("step", ()))]
    if not steps:
        return
    if view == "live":
        live = [s[np.nansum([v for k, v in r.metrics.items() if k.startswith("live_") and k.endswith("_n")],
                            axis=0) > 0] for r, s in zip(runs, steps) if any(k.startswith("live_") for k in r.metrics)]
        steps = [s for s in live if len(s)] or steps
    left, last = min(float(s.min()) for s in steps), max(float(s.max()) for s in steps)
    if not own:
        last = max([last] + [float(np.nanmax(line.get_xdata())) for ax in fig.axes for line in ax.get_lines()
                             if line.get_transform() == line.axes.transData and len(line.get_xdata())])
    pad = 0.03 * max(last - left, 1.0)
    left, right = left - pad, last + pad
    for ax in fig.axes:
        ax.set_xlim(left, right)
    for ax in fig.axes:
        log = ax.get_yscale() == "log"
        low, high = np.inf, -np.inf
        for line in (line for other in ax.get_shared_y_axes().get_siblings(ax) for line in other.get_lines()):
            x, y = (np.asarray(v, dtype=float) for v in line.get_data())
            if line.get_transform() != line.axes.transData:   # a flat reference: x in axes units
                x = np.full_like(y, left)
            keep = (x >= left) & (x <= right) & np.isfinite(y) & ((y > 0) if log else True)
            if keep.any():
                low, high = min(low, y[keep].min()), max(high, y[keep].max())
        if np.isfinite(low) and high > low:
            if log:
                pad = (high / low) ** 0.05
                ax.set_ylim(low / pad, high * pad)
            else:
                pad = 0.05 * (high - low)
                ax.set_ylim(low - pad, high + pad)


def is_vae(run: RunData) -> bool:
    """A video VAE's run (scripts/train_vae_ui.py), not a world model's."""
    return "recon_wrong" in run.metrics


def draw_vae_curves(fig: Figure, runs: list[RunData], compare: list[RunData]) -> None:
    """Six panels for video VAEs over training, two more for a sprite output; runs solid (raw faint, smoothed
    bold), compared runs dashed."""
    fig.clear()
    fig.set_facecolor(BG)
    every = [(r, "-") for r in runs] + [(r, DASHES[1 + i % 3]) for i, r in enumerate(compare)]
    sprites = any("sprite_missed" in r.metrics for r, _ in every)
    axes = fig.subplots(4 if sprites else 3, 2)
    for n, (run, dash) in enumerate(every):
        c = COLOURS[n % len(COLOURS)]
        if sprites:
            _plot(axes[3, 0], run, "sprite_missed", f"{run.name} missed", c, dash)
            _plot(axes[3, 0], run, "sprite_false", f"{run.name} called sprite wrongly", c, ":", raw=False)
            _plot(axes[3, 0], run, "sprite_missed_well", f"{run.name} missed in the well", COLOURS[(n + 2) % len(COLOURS)],
                  dash)
            _plot(axes[3, 0], run, "sprite_missed_well_alone", f"{run.name} in the well, no history",
                  COLOURS[(n + 2) % len(COLOURS)], "-.", raw=False)
            _plot(axes[3, 1], run, "train_sprite_loss", f"{run.name}", c, dash)
        _plot(axes[0, 0], run, "train_loss", f"{run.name} loss", c, dash)
        _plot(axes[0, 1], run, "recon_wrong", f"{run.name} game", c, dash)
        _plot(axes[1, 0], run, "recon_wrong_changed", f"{run.name} changed", c, dash)
        _plot(axes[1, 0], run, "recon_wrong_border", f"{run.name} border", c, ":", raw=False)
        _plot(axes[1, 1], run, "sample_wrong", f"{run.name} training's noise", c, dash)
        _plot(axes[1, 1], run, "noisy_wrong", f"{run.name} twice it", c, ":", raw=False)
        _plot(axes[1, 1], run, "alone_wrong", f"{run.name} alone", c, "-.", raw=False)
        _plot(axes[2, 0], run, "frames_exact", f"{run.name}", c, dash)
        _plot(axes[2, 1], run, "latent_spread", f"{run.name} spread", c, dash)
    _style(axes[0, 0], "Train loss: pixel cross-entropy of decoding noisy latents", "nats", log=True)
    _style(axes[0, 1], "Reconstructed from the latents: game pixels wrong", "fraction", log=True)
    _style(axes[1, 0], "Wrong on pixels that changed since the frame before; border (dotted)", "fraction", log=True)
    _style(axes[1, 1], "With training's noise on the latents; twice it (dotted); each frame alone (dash-dot)",
           "fraction", log=True)
    _style(axes[2, 0], "Frames with no pixel wrong", "fraction")
    _style(axes[2, 1], "Latents: spread per number (unit RMS: about 1)", "")
    if sprites:
        _style(axes[3, 0], "Sprite pixels missed: all, in the well (the falling piece), in the well with no "
                           "history (dash-dot); other pixels called sprite (dotted)", "fraction", log=True)
        _style(axes[3, 1], "Train sprite loss (per-pixel cross-entropy, sprite or not)", "nats", log=True)
    for ax in axes.flat:
        _legend(ax)


def vae_stats(run: RunData) -> str:
    last = {k: v[-1] for k, v in run.metrics.items()}
    get = lambda k: last.get(k, math.nan)
    lines = [f"{run.name}: step {int(last['step']):,}   learning rate {get('lr'):.2e}",
             f"  train loss {last['train_loss']:.5f} (pixel cross-entropy of decoding noisy latents"
             + (", plus the sprite loss)" if math.isfinite(get("train_sprite_loss")) else ")"),
             f"  reconstructed from the latents, wrong: game {last['recon_wrong']:.4%}, on changed pixels "
             f"{get('recon_wrong_changed'):.3%} (changed {get('changed_fraction'):.2%}), border "
             f"{get('recon_wrong_border'):.4%}; frames exact {get('frames_exact'):.1%}",
             f"  with training's noise {get('sample_wrong'):.4%} (each frame alone: {get('alone_wrong'):.4%}); "
             f"twice it {get('noisy_wrong'):.4%}; spread {get('latent_spread'):.3f}"]
    if math.isfinite(get("sprite_missed")):
        lines.append(f"  sprites: missed {get('sprite_missed'):.2%} of sprite pixels, called {get('sprite_false'):.3%} "
                     f"of the others sprite; sprite loss {get('train_sprite_loss'):.4f}")
        lines.append(f"  the falling piece (sprite pixels in the well): missed {get('sprite_missed_well'):.2%}, "
                     f"with no history {get('sprite_missed_well_alone'):.2%}")
    lines.append(f"  data wait {get('data_wait_ms_mean'):.0f} ms mean, {get('data_wait_ms_max'):.0f} ms max; "
                 f"elapsed this launch {get('seconds'):.0f} s")
    return "\n".join(lines)


DATA_SKIP = ("data_windows", "data_kept", "data_wait_ms_mean", "data_wait_ms_max")


def data_columns(m: dict) -> tuple[list[str], list[str]]:
    """A run's training-data columns (live_batches.data_fields) -> (events, screens)."""
    from token_world.data.live_batches import data_field
    from token_world.data.world_nes_tetris import SCREENS
    shown = {data_field(s) for s in SCREENS}
    keys = [k for k in m if k.startswith("data_") and k not in DATA_SKIP]
    return [k for k in keys if k not in shown], [k for k in keys if k in shown]


def draw_data(fig: Figure, runs: list[RunData]) -> None:
    """What the training windows held, every checkpoint (the windows since the one before): the share with
    each event, on each screen, and the share of the stream's windows kept (tossing)."""
    fig.clear()
    fig.set_facecolor(BG)
    events_ax, screens_ax, novel_ax, kept_ax = fig.subplots(4, 1, gridspec_kw={"height_ratios": (2, 2, 2, 1)})
    for r, (run, dash) in enumerate(zip(runs, DASHES)):
        events, shown = data_columns(run.metrics)
        for ax, keys in ((events_ax, events), (screens_ax, shown)):
            for i, key in enumerate(keys):
                _plot(ax, run, key, key[5:].replace("_", " ") if r == 0 else None, COLOURS[i % len(COLOURS)], dash,
                      raw=False)
        _plot(novel_ax, run, "novel_boards", run.name, COLOURS[r % len(COLOURS)], dash, raw=False)
        _plot(kept_ax, run, "data_kept", run.name, COLOURS[r % len(COLOURS)], dash, raw=False)
    _style(events_ax, "Training windows holding each event (smoothed; one line style per run)", "share", log=True)
    _style(screens_ax, "Training windows showing each screen other than play", "share", log=True)
    _style(novel_ax, "Piece spawns on a board never seen before in the run", "share")
    _style(kept_ax, "Share of the stream's windows kept (tossing; 1 = every window)", "share")
    for ax in (events_ax, screens_ax, novel_ax, kept_ax):
        _legend(ax)


def data_table(runs: list[RunData]) -> str:
    """Each run's training data so far: windows trained on, and the share holding each event and screen."""
    lines = []
    for run in runs:
        m = run.metrics
        if "data_windows" not in m:
            continue
        n = np.nan_to_num(m["data_windows"])
        total = n.sum()
        events, shown = data_columns(m)
        share = lambda k: float((np.nan_to_num(m[k]) * n).sum() / max(total, 1))
        kept = float((np.nan_to_num(m["data_kept"]) * n).sum() / max(total, 1)) if "data_kept" in m else math.nan
        lines.append(f"{run.name}: {total:,.0f} windows trained on so far (kept {kept:.0%} of the stream's)")
        boards = m["distinct_boards"][np.isfinite(m["distinct_boards"])] if "distinct_boards" in m else []
        if len(boards):
            lines.append(f"    {int(boards[-1]):,} distinct boards at spawns so far")
        for keys in (events, shown):
            lines.append("    " + "   ".join(f"{k[5:].replace('_', ' ')} {share(k):.2%}" for k in keys))
    return "\n".join(lines) or "No training-data columns yet (runs from 2026-10-10 on log them)."


def draw_speed(fig: Figure, runs: list[RunData]) -> None:
    fig.clear()
    fig.set_facecolor(BG)
    rate_ax, wait_ax = fig.subplots(2, 1)
    waits = []
    for i, run in enumerate(runs):
        m = run.metrics
        if "seconds" not in m:
            continue
        steps, seconds = m["step"], m["seconds"]
        same_launch = np.diff(seconds) > 0                                  # seconds restart at each launch
        rate = np.where(same_launch, np.diff(steps) / np.maximum(np.diff(seconds), 1e-9), np.nan)
        rate_ax.plot(steps[1:], smooth(rate), color=COLOURS[i], label=run.name)
        if "data_wait_ms_mean" in m:
            wait_ax.plot(steps, smooth(m["data_wait_ms_mean"]), color=COLOURS[i], label=f"{run.name} mean")
            waits.append(smooth(m["data_wait_ms_mean"]))
    title = "Waiting for data per step"
    waits = np.concatenate(waits) if waits else np.array([])
    waits = waits[np.isfinite(waits)]
    if len(waits) and waits.max() > 4 * max(np.percentile(waits, 95), 1):         # emulators starting up
        wait_ax.set_ylim(0, 1.3 * max(np.percentile(waits, 95), 1))
        title += f" (the start, up to {waits.max():,.0f} ms while the emulators boot, cut off)"
    _style(rate_ax, "Training speed", "steps/s")
    _style(wait_ax, title, "ms")
    for ax in (rate_ax, wait_ax):
        _legend(ax)


def draw_horizons(fig: Figure, run: RunData) -> None:
    """Changed game pixels wrong by horizon (rows) over training (columns), and the latest curve."""
    fig.clear()
    fig.set_facecolor(BG)
    heat_ax, curve_ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": (2, 1)})
    m = run.metrics
    horizons = [h for h in (1, 2, 4, 8, 16) if f"{GAME}_h{h}_wrong_changed" in m]
    if horizons:
        grid = np.stack([smooth(m[f"{GAME}_h{h}_wrong_changed"]) for h in horizons])
        mesh = heat_ax.pcolormesh(m["step"], np.arange(len(horizons)), grid, cmap="magma", vmin=0, vmax=1,
                                  shading="nearest")
        heat_ax.set_yticks(range(len(horizons)), [f"+{h}" for h in horizons])
        fig.colorbar(mesh, ax=heat_ax).ax.tick_params(colors=TEXT, labelsize=7)
        last = [m[f"{GAME}_h{h}_{k}"][-1] for h in horizons for k in ("wrong_changed",)]
        curve_ax.plot(horizons, last, marker="o", color=COLOURS[0], label="wrong on changed")
        for kind, colour in (("wrong_static", COLOURS[3]), ("border_wrong", COLOURS[4]),
                             ("border_copy", COLOURS[1])):
            if f"{GAME}_h1_{kind}" in m:
                curve_ax.plot(horizons, [m[f"{GAME}_h{h}_{kind}"][-1] for h in horizons], marker="o",
                              color=colour, label=kind.replace("_", " "))
        curve_ax.set_xscale("log", base=2)
    _style(heat_ax, f"{run.name}: changed game pixels wrong by horizon over training (smoothed)", "horizon")
    _style(curve_ax, "Latest checkpoint", "fraction")
    _legend(curve_ax)


# ---------------------------------------------------------------------------------------------- the window
EXPLAIN = {
    "curves": ("How the model is learning, by training step, in three views. Long dreams: the tests every 2,000 "
               "steps (1,000 dreamed frames from the same 32 real game starts), one pair of panels per score, no "
               "buttons on the left and the bot's buttons on the right on one scale; the line is the mean of the "
               "last 5 tests, each dot one test, and the flat dashed lines are the best models so far on the same "
               "trials (actnorm 72k, resync 18k). Live events: the training batches' own frames (fresh every step, "
               "never trained on before), one event frame per window hidden whole and drawn from the real past: "
               "the share drawn exactly and the changed pixels' wrong share, per event, and the J / L / T turns "
               "with the real press against A and B swapped (the gap above 0: it follows A vs B). Training: the "
               "loss, the previews (changed game pixels wrong 1, 4 and 16 frames ahead; copying gets 100%), the "
               "border against copying, and the motion head. Compared runs are dashed."),
    "curves_vae": ("How the tokenizer is learning, by training step: faint lines are each checkpoint, bold the mean of "
                   "the last 9. Every 200 steps the batch about to be trained on is encoded to latents and decoded back "
                   "(it has not been trained on yet). Wrong: game pixels whose colour comes back wrong, all of them, "
                   "those that changed since the frame before, and the border; with the training noise added to the "
                   "latents (what a generator's small errors look like) and twice it; frames with no pixel wrong; the "
                   "latents' spread (unit RMS: about 1). Sprites (a model with sprite numbers): sprite pixels it "
                   "misses, all and in the well (the falling piece, told apart from the stack only by its motion), "
                   "and in the well with each frame encoded alone (no history: what looking back adds); other "
                   "pixels it calls sprite, and the sprite loss."),
    "previews": ("Four windows of the batch about to be trained on, most changing first, each an animation: the last "
                 "real frame, then 16 frames the model generated from the real buttons. REAL: what happened. "
                 "GENERATED: the model's expected colours (a blur = unsure between pictures). WRONG: the real frame "
                 "with the pixels the model gets wrong in magenta. A motion model also shows REAL MOTION (the real "
                 "frame's motion: on the context frame, what went in; then what each frame really did) and DRAWN "
                 "MOTION (what the model drew for its own frame, which goes into its history): an arrow from each "
                 "moved 8 x 8 cell's source to it, coloured by direction, white boxes for cells that are new."),
    "previews_vae": ("Four windows of the batch about to be trained on, most changing first, each an animation of its "
                     "last 8 frames, each encoded and decoded back. REAL: the frame. GENERATED: decoded from its "
                     "latents. WRONG: pixels decoded wrong, in magenta. SPRITES (a model with sprite numbers): the "
                     "real frame dimmed, with the pixels the PPU drew as sprites as the model says them: GREEN found, "
                     "RED missed, YELLOW called sprite where there is none. In Tetris the sprites are the falling "
                     "piece and the next-piece box; the stack and the statistics are background."),
    "dream": ("Every 2,000 steps the model dreams 1,000 frames from each of 32 real game starts, once with nobody "
              "pressing anything and once with the bot's buttons, and the dream is scored against the real game "
              "(averaged weights). Animation: the real playfield beside the dream. Sheet: every 64 frames side by "
              "side, to see slow drift (colours, pieces). Table, the last tests: wrong = playfield pixels wrong at "
              "+16 ... +128; mass = the dream's blocks over the real (1 = right); piece / hit +32 = the falling piece "
              "kept / in the right place; fall = its speed (1 = real); spawn = the next piece on time; presence = a "
              "piece drawn; per 100 of 1,000 frames: lost (blocks vanishing), morph (pieces changing shape), full "
              "(full rows that never clear), size (pieces born too big or small). Below: the plateau check, the "
              "measured stopping rule (the last 3 tests against the 3 before, on the same trials)."),
    "compare": ("The long-dream scores of this run and each compared run at every step both were tested, on the "
                "same 32 games (the same --seed): each cell this run / the compared run."),
    "scenarios": ("Checks run on checkpoints by hand (on the CPU while training runs): scenarios "
                  "(scripts/tetris_scenarios.py), real moments of the game (gravity, a shift, a rotation, a spawn, a "
                  "line clear, a top-out, a game start) with the model generating what follows: event pixels wrong 2 "
                  "and 16 frames on, and the share exactly right at +2; and event checks (scripts/event_checks.py): "
                  "restarts, line clears, and pieces under the bot's play. The latest check of each run."),
    "data": ("What the training windows held, from the live streams: the share holding each event and showing "
             "each screen, piece spawns on boards never seen before (novelty), and the share of the stream's windows "
             "kept (tossing). The table sums every window trained on so far."),
    "horizons": ("Preview error by horizon: changed game pixels wrong 1, 2, 4, 8 and 16 frames ahead, over training "
                 "(top, darker is better) and at the latest checkpoint (bottom)."),
    "speed": ("Training steps per second (each launch measured on its own) and how long each step waited for live "
              "data: a long wait means the emulators, not the GPU, set the pace."),
    "stats": "The latest checkpoint's numbers, as the trainer wrote them.",
}
WORLD_ONLY = ("Long dream", "Compare", "Scenarios", "Horizons")


def run_kind(run: RunData) -> str:
    """'vae' for a tokenizer's run (scripts/train_vae_ui.py), else 'world' (any world model)."""
    try:
        if json.loads((run.path / "run_config.json").read_text()).get("kind") == "vae":
            return "vae"
    except (OSError, ValueError):
        pass
    return "vae" if is_vae(run) else "world"


class RunViewer:
    """The tabs, inside `master`; refresh() reloads what changed (also on a timer). Every image fits its space and
    every table scrolls (ui/viewer_widgets.py); each tab says what it shows."""

    def __init__(self, master, runs: list[Path], compare: list[Path] = (), refresh_ms: int = 10_000):
        self.runs = [RunData(p) for p in runs]
        self.compare = [RunData(p) for p in compare if p.exists()]
        self.refresh_ms, self.master, self.kind = refresh_ms, master, None
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)
        self.tabs: dict[str, ttk.Frame] = {}

        tab = self._tab("Curves")
        self.curves_text = self._explain(tab, EXPLAIN["curves"])
        self.show_compare = tk.BooleanVar(value=True)
        self.own_steps = tk.BooleanVar(value=True)
        self.curve_view = tk.StringVar(value="dreams")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10)
        for value, text in CURVE_VIEWS.items():
            ttk.Radiobutton(bar, text=text, value=value, variable=self.curve_view,
                            command=lambda: self.refresh(force=True)).pack(side="left", padx=(0, 12))
        if self.compare:
            ttk.Checkbutton(bar, text="show compared runs (dashed): " + ", ".join(r.name for r in self.compare),
                            variable=self.show_compare, command=lambda: self.refresh(force=True)).pack(side="left")
            ttk.Checkbutton(bar, text="only up to this run's step (compare at the same step)",
                            variable=self.own_steps, command=lambda: self.refresh(force=True)).pack(side="left",
                                                                                                    padx=(16, 0))
        self.curves_fig, self.curves_canvas = self._figure(tab)

        tab = self._tab("Previews")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10, pady=(6, 0))
        ttk.Label(bar, text="Run", style="Legend.TLabel").pack(side="left", padx=(0, 6))
        self.selected = tk.StringVar(value=self.runs[0].name)
        box = ttk.Combobox(bar, state="readonly", values=[r.name for r in self.runs], textvariable=self.selected,
                           width=40)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda e: self.refresh(force=True))
        ttk.Label(bar, text="Show", style="Legend.TLabel").pack(side="left", padx=(20, 6))
        self.preview_pick = tk.StringVar(value="all")
        for text, value in (("all four", "all"), ("1", "0"), ("2", "1"), ("3", "2"), ("4", "3")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.preview_pick,
                            command=lambda: self.refresh()).pack(side="left", padx=(0, 8))
        self.previews_text = self._explain(tab, EXPLAIN["previews"])
        self.previews = FitGrid(tab, 4, empty="no preview yet (one every 200 steps)")
        self.previews.pack(fill="both", expand=True, padx=6, pady=6)

        tab = self._tab("Long dream")
        self._explain(tab, EXPLAIN["dream"])
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=10)
        self.dream_view = tk.StringVar(value="gif")
        for text, value in (("Animation", "gif"), ("Sheet: every 64 frames", "png")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.dream_view,
                            command=lambda: self.refresh(force=True)).pack(side="left", padx=(0, 12))
        panes = ttk.PanedWindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True, padx=6, pady=6)
        self.dream_image = FitImage(panes, empty="no long-dream test yet (every 2,000 steps)")
        self.dream_table = TextPanel(panes, height=12)
        panes.add(self.dream_image, weight=3)
        panes.add(self.dream_table, weight=2)
        split(panes, 0.7)

        if self.compare:
            tab = self._tab("Compare")
            self._explain(tab, EXPLAIN["compare"])
            self.compare_table = TextPanel(tab)
            self.compare_table.pack(fill="both", expand=True, padx=6, pady=6)

        tab = self._tab("Scenarios")
        self._explain(tab, EXPLAIN["scenarios"])
        self.scenario_table = TextPanel(tab)
        self.scenario_table.pack(fill="both", expand=True, padx=6, pady=6)

        tab = self._tab("Data")
        self._explain(tab, EXPLAIN["data"])
        panes = ttk.PanedWindow(tab, orient="vertical")
        panes.pack(fill="both", expand=True, padx=6, pady=6)
        self.data_table = TextPanel(panes, font=("Consolas", 9), height=6)
        panes.add(self.data_table, weight=1)
        holder = ttk.Frame(panes)
        panes.add(holder, weight=3)
        split(panes, 0.22)
        self.data_fig, self.data_canvas = self._figure(holder)

        tab = self._tab("Horizons")
        self._explain(tab, EXPLAIN["horizons"])
        self.horizon_fig, self.horizon_canvas = self._figure(tab)
        tab = self._tab("Speed")
        self._explain(tab, EXPLAIN["speed"])
        self.speed_fig, self.speed_canvas = self._figure(tab)
        tab = self._tab("Stats")
        self._explain(tab, EXPLAIN["stats"])
        self.stats_table = TextPanel(tab)
        self.stats_table.pack(fill="both", expand=True, padx=6, pady=6)
        self.job = None
        self.refresh(force=True)

    def _tab(self, name: str) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        self.tabs[name] = frame
        return frame

    @staticmethod
    def _explain(tab: ttk.Frame, text: str) -> Explainer:
        label = Explainer(tab, text)
        label.pack(fill="x", padx=10, pady=(8, 4))
        return label

    @staticmethod
    def _figure(holder: ttk.Frame) -> tuple[Figure, FigureCanvasTkAgg]:
        """A chart filling `holder`, laid out again at every size (constrained layout)."""
        fig = Figure(figsize=(12, 8), dpi=100, facecolor=BG, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.get_tk_widget().configure(background=BG, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        return fig, canvas

    def run(self) -> RunData:
        return next((r for r in self.runs if r.name == self.selected.get()), self.runs[0])

    def _set_kind(self, kind: str) -> None:
        """Show the tabs and words for a tokenizer's runs or a world model's."""
        if kind == self.kind:
            return
        self.kind = kind
        for name in WORLD_ONLY:
            if name in self.tabs:
                if kind == "vae":
                    self.notebook.hide(self.tabs[name])
                else:
                    self.notebook.add(self.tabs[name])
        self.curves_text.configure(text=EXPLAIN["curves_vae" if kind == "vae" else "curves"])
        self.previews_text.configure(text=EXPLAIN["previews_vae" if kind == "vae" else "previews"])

    def refresh(self, force: bool = False) -> None:
        if self.job is not None:
            self.master.after_cancel(self.job)
        changed = any([r.refresh() for r in self.runs + self.compare]) or force
        run = self.run()
        kind = "vae" if self.runs and all(run_kind(r) == "vae" for r in self.runs) else "world"
        self._set_kind(kind)
        pick = self.preview_pick.get()
        self.previews.show([run.path / "previews" / f"{GAME}_{i}.gif"
                            for i in (range(4) if pick == "all" else [int(pick)])])
        self.dream_image.show(run.path / f"long_dream.{self.dream_view.get()}")
        if changed:
            compare = self.compare if self.show_compare.get() else []
            if kind == "vae":
                draw_vae_curves(self.curves_fig, self.runs, compare)
            else:
                draw_curves(self.curves_fig, self.runs, compare, self.curve_view.get())
            fit_steps(self.curves_fig, self.runs, "training" if kind == "vae" else self.curve_view.get(),
                      own=self.own_steps.get())
            self.curves_canvas.draw_idle()
            draw_horizons(self.horizon_fig, run)
            self.horizon_canvas.draw_idle()
            draw_speed(self.speed_fig, self.runs)
            self.speed_canvas.draw_idle()
            draw_data(self.data_fig, self.runs)
            self.data_canvas.draw_idle()
            self.data_table.set(data_table(self.runs))
            self.dream_table.set("\n\n".join([long_dream_table(self.runs + self.compare)]
                                             + [plateau.report(r.name, plateau.read(r.path)) for r in self.runs]))
            self.stats_table.set(stats_text(run))
            self.scenario_table.set(scenario_table([run] + self.compare) + "\n\n" + event_table([run] + self.compare))
            if self.compare:
                self.compare_table.set("\n\n".join(compare_table(run, other) for other in self.compare))
        self.job = self.master.after(self.refresh_ms, self.refresh)


BIASED = (("chosen", "bias_choices_dream", "{:.0f}"), ("real chose", "bias_choices_real", "{:.0f}"),
          ("types", "bias_types_dream", "{:.0f}"), ("top share", "bias_top_share_dream", "{:.2f}"),
          ("mix distance", "bias_type_distance", "{:.2f}"), ("garbled", "bias_garbled", "{:.0%}"))
COMPARED = (("presence", "presence", "{:.2f}"), ("activation", "activation", "{:.2f}"),
            ("hit +32", "piece_hit_32", "{:.2f}"), ("ghost", "piece_ghost_32", "{:.2f}"), ("fall", "fall", "{:.2f}"),
            ("wrong +16", "wrong_16", "{:.1%}"), ("wrong +128", "wrong_128", "{:.1%}"),
            ("lost 1k", "lost_long", "{:.2f}"), ("morph 1k", "morph_long", "{:.2f}"), ("full 1k", "full_long", "{:.2f}"),
            ("size 1k", "size_long", "{:.2f}"),
            ("timer +128", "timer_wrong_128", "{:.0%}"), ("unseen change", "unseen_change", "{:.0%}"))


def scenario_table(runs: list[RunData]) -> str:
    """Each run's latest scenario check (its scenarios.csv): per event, event pixels wrong at +2 and +16 and
    exact at +2, runs side by side."""
    latest = {}
    for run in runs:
        rows = _read_rows(run.path / "scenarios.csv") if (run.path / "scenarios.csv").exists() else []
        if rows:
            step = max(int(r["step"]) for r in rows)
            latest[run.name] = (step, {r["scenario"]: r for r in rows if int(r["step"]) == step})
    if not latest:
        return "No scenario checks yet: python scripts/tetris_scenarios.py <run or checkpoint> [...] --device cpu"
    width = 27
    lines = ["".ljust(24) + "".join(f"{(name[-17:] + ' @' + format(step, ',')):>{width}}"
                                   for name, (step, _) in latest.items()),
             "event".ljust(24) + "".join(f"{'+2 / +16 / exact (n)':>{width}}" for _ in latest)]
    events = list(dict.fromkeys(e for _, rows in latest.values() for e in rows))
    for event in events:
        cells = []
        for _, rows in latest.values():
            r = rows.get(event)
            if r is None:
                cells.append(f"{'—':>{width}}")
                continue
            f = lambda k: float(r[k]) if r.get(k) not in (None, "", "nan") else math.nan
            cells.append(f"{f('event_wrong_2'):6.1%} {f('event_wrong_16'):6.1%} {f('exact'):4.0%} ({r['instances']})"
                         .replace("nan%", "  —").rjust(width))
        lines.append(event[:23].ljust(24) + "".join(cells))
    return "\n".join(lines)


def event_table(runs: list[RunData]) -> str:
    """Each run's latest event checks (its event_checks.csv, scripts/event_checks.py): per check and group (all, or
    a clear size), each score, runs side by side."""
    latest = {}
    for run in runs:
        path = run.path / "event_checks.csv"
        rows = _read_rows(path) if path.exists() else []
        for check in dict.fromkeys(r["check"] for r in rows):
            mine = [r for r in rows if r["check"] == check]
            step = max(int(r["step"]) for r in mine)
            latest.setdefault(check, {})[run.name] = (step, [r for r in mine if int(r["step"]) == step])
    if not latest:
        return "No event checks yet: python scripts/event_checks.py <run or checkpoint> [...] --device cpu"
    width, lines = 27, []
    for check, by_run in latest.items():
        lines.append(check.ljust(32) + "".join(f"{(name[-17:] + ' @' + format(step, ',')):>{width}}"
                                              for name, (step, _) in by_run.items()))
        keys = list(dict.fromkeys((r["group"], r["score"]) for _, rows in by_run.values() for r in rows))
        for group, score in keys:
            cells = []
            for _, rows in by_run.values():
                r = next((r for r in rows if (r["group"], r["score"]) == (group, score)), None)
                text = "—" if r is None else f"{float(r['value']):.3g} ({r['events']})"
                cells.append(f"{text:>{width}}")
            lines.append(f"  {group[:18]:18s} {score[:11]:11s}" + "".join(cells))
        lines.append("")
    return "\n".join(lines)


def compare_table(run: RunData, other: RunData) -> str:
    """The two runs' long-dream scores at every step both were tested, side by side (run / other), then their
    bias tests (diagnostics/bias.py: the next pieces each run's dreams chose)."""
    if not run.long_dream or not other.long_dream:
        return "No long-dream tests in both runs yet."
    parts = []
    for kind, what in ((False, "no buttons"), (True, "the bot's buttons")):
        mine = next((d for v, d in run.long_dream.items() if pressed(v) == kind), None)
        theirs = next((d for v, d in other.long_dream.items() if pressed(v) == kind), None)
        if mine is None or theirs is None:
            parts.append(f"{what}: not tested in both runs yet")
        else:
            parts.append(f"{what}:\n" + _compare_kind(mine, theirs, run.name, other.name))
    return "\n\n".join(parts)


def _compare_kind(mine: dict, theirs: dict, name: str, other: str) -> str:
    """compare_table for one kind of long dream (no buttons, or the bot's)."""
    where = {int(s): i for i, s in enumerate(theirs["step"])}
    width = 15
    lines = [f"each cell: {name} / {other}", "",
             f"{'step':>7}  " + "".join(f"{name:>{width}}" for name, _, _ in COMPARED)]
    for i, step in enumerate(mine["step"]):
        j = where.get(int(step))
        if j is None:
            continue
        cells = []
        for _, key, form in COMPARED:
            a = mine.get(key, np.full(len(mine["step"]), np.nan))[i]
            b = theirs.get(key, np.full(len(theirs["step"]), np.nan))[j]
            cells.append(f"{form.format(a)} / {form.format(b)}".replace("nan%", "—").replace("nan", "—"))
        lines.append(f"{int(step):>7,}  " + "".join(f"{c:>{width}}" for c in cells))
    if len(lines) <= 3:
        return "No step tested in both runs yet."
    lines += ["", "next pieces the dreams chose (bias test): " + name + " / " + other, "",
              f"{'step':>7}  " + "".join(f"{name:>{width}}" for name, _, _ in BIASED)]
    for i, step in enumerate(mine["step"]):
        j = where.get(int(step))
        if j is None:
            continue
        cells = []
        for _, key, form in BIASED:
            a = mine.get(key, np.full(len(mine["step"]), np.nan))[i]
            b = theirs.get(key, np.full(len(theirs["step"]), np.nan))[j]
            cells.append(f"{form.format(a)} / {form.format(b)}".replace("nan%", "—").replace("nan", "—"))
        lines.append(f"{int(step):>7,}  " + "".join(f"{c:>{width}}" for c in cells))
    return "\n".join(lines)


def long_dream_table(runs: list[RunData], last: int = 6) -> str:
    lines = ["run / variant                          step     wrong +16   +32     +64     +128    mass  piece+32"
             "  hit+32  fall  spawn  presence  unseen patch/change   per 100 of 1,000 frames: lost morph full size"]
    for run in runs:
        for variant, d in run.long_dream.items():
            for i in range(max(0, len(d["step"]) - last), len(d["step"])):
                lines.append(f"{run.name[:24]:24s} {variant:12s} {int(d['step'][i]):>7,}   "
                             + "  ".join(f"{d[f'wrong_{h}'][i]:6.2%}" for h in (16, 32, 64, 128))
                             + f"   {d['mass'][i]:.2f}  "
                             + "  ".join(f"{d.get(k, np.full(len(d['step']), np.nan))[i]:6.2f}"
                                         for k in ("piece_32", "piece_hit_32", "fall", "spawn", "presence"))
                             + "  " + "/".join(f"{d.get(k, np.full(len(d['step']), np.nan))[i]:6.2%}"
                                               for k in ("unseen_patch", "unseen_change"))
                             + "   " + "  ".join(f"{d.get(k, np.full(len(d['step']), np.nan))[i]:5.2f}"
                                                 for k in ("lost_long", "morph_long", "full_long", "size_long")))
    for run in runs:                                  # the bias tests (diagnostics/bias.py)
        for variant, d in run.long_dream.items():
            if "bias_choices_dream" not in d:
                continue
            lines.append(f"{run.name[:24]:24s} {variant:12s} next pieces chosen (types, top share, mix distance "
                         f"to real, garbled previews):")
            for i in range(max(0, len(d["step"]) - last), len(d["step"])):
                lines.append(f"{'':37s} {int(d['step'][i]):>7,}   {d['bias_choices_dream'][i]:4.0f} chosen "
                             f"({d['bias_types_dream'][i]:.0f} types, top {d['bias_top_share_dream'][i]:.2f}, "
                             f"distance {d['bias_type_distance'][i]:.2f}, garbled {d['bias_garbled'][i]:.0%}); "
                             f"real {d['bias_choices_real'][i]:.0f} chosen, {d['bias_types_real'][i]:.0f} types")
    return "\n".join(lines) if len(lines) > 1 else "No long-dream tests yet (every 1,000 steps)."


def stats_text(run: RunData) -> str:
    m = run.metrics
    if "step" not in m or not len(m["step"]):
        return f"{run.name}: no checkpoints yet"
    if is_vae(run):
        return vae_stats(run)
    last = {k: v[-1] for k, v in m.items()}
    lines = [f"{run.name}: step {int(last['step']):,}   learning rate {last.get('lr', math.nan):.2e}",
             f"  train loss {last.get('train_loss', math.nan):.4f}: game {last.get('train_loss_game', math.nan):.4f}, "
             f"border {last.get('train_loss_border', math.nan):.4f}, changed pixels "
             f"{last.get('train_loss_changed', math.nan):.4f}; masked {last.get('masked_fraction', math.nan):.0%}",
             *([("  layered: " + ", ".join(f"{p.replace('_', ' ')} {last[f'train_loss_{p}']:.4f}" for p in
                                           ("cells", "sprites", "sprite_flags", "sprite_move", "sprite_place",
                                            "sprite_pixels", "backdrop", "camera")
                                           if f"train_loss_{p}" in last))]
               if "train_loss_cells" in last else []),
             f"  blend radius {last.get('blend_frames', math.nan):.0f}; last training rollout "
             f"{last.get('rollout_frames', math.nan):.0f} frames, pixels wrong {last.get('rollout_wrong', math.nan):.3%}"]
    for h in (1, 2, 4, 8, 16):
        k = f"{GAME}_h{h}_"
        if k + "wrong" in last:
            lines.append(f"  +{h:>2}: changed {last[k + 'changed']:.2%}; wrong: all {last[k + 'wrong']:.2%}, on "
                         f"changed {last[k + 'wrong_changed']:.1%}, on static {last[k + 'wrong_static']:.3%}; border "
                         f"{last[k + 'border_wrong']:.2%} (copying {last.get(k + 'border_copy', math.nan):.2%})")
    lines.append(f"  data wait {last.get('data_wait_ms_mean', math.nan):.0f} ms mean, "
                 f"{last.get('data_wait_ms_max', math.nan):.0f} ms max; elapsed this launch "
                 f"{last.get('seconds', math.nan):.0f} s")
    return "\n".join(lines)
