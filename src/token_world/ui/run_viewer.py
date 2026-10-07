"""World-model run viewer: every chart and preview read back from run folders, refreshed as they change.

The trainer (scripts/train_dynamics_ui.py) writes everything a run has to its folder: metrics.csv (and
metrics_schema_*.csv, earlier columns), horizon_curve.csv, previews/*.gif (real | generated | wrong,
animated), long_dream.csv/.gif/.png (the no-button test every LONG_DREAM_EVERY steps). The viewer reads
only those files, so the same window serves a training run (ui/dynamics_app.py adds the Stop button)
and any finished or running run beside it (scripts/view_runs.py). Runs given as `compare` are drawn
dashed on the same axes, by step.

Tabs: Curves (losses, preview errors against copying the last context frame, the border against its
copy baseline, long-dream error and block mass), Previews (four animated windows), Long dream (the
latest test, animated, with its numbers), Horizons (preview error by horizon over training), Speed
(steps/s, data waits), Stats (the latest checkpoint's numbers).
"""
from __future__ import annotations

import csv
import io
import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageSequence, ImageTk

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
    ax.set_title(title, color="#e2e2ec", fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.grid(True, color=GRID, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if log:
        ax.set_yscale("log")


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(facecolor="#1f1f28", labelcolor="#e2e2ec", fontsize=6, loc="best")


def _plot(ax, run: RunData, key: str, label: str | None, colour: str, dash: str, raw: bool = True) -> None:
    """Smoothed (bold) over raw (faint, solid runs only); only the first run's lines carry a label."""
    m = run.metrics
    if key not in m or "step" not in m or not np.isfinite(m[key]).any():
        return
    if raw and dash == "-":
        ax.plot(m["step"], m[key], color=colour, alpha=0.18, linewidth=0.7)
    ax.plot(m["step"], smooth(m[key]), color=colour, linestyle=dash, linewidth=1.4, label=label)


def draw_curves(fig: Figure, runs: list[RunData], compare: list[RunData]) -> None:
    """Six panels over training; runs solid (raw faint, smoothed bold), compared runs dashed."""
    fig.clear()
    fig.set_facecolor(BG)
    axes = fig.subplots(3, 2)
    every = [(r, "-") for r in runs] + [(r, DASHES[1 + i % 3]) for i, r in enumerate(compare)]
    for n, (run, dash) in enumerate(every):
        label = (lambda text: text) if n == 0 else (lambda text: None)
        _plot(axes[0, 0], run, "train_loss_changed", label("changed pixels"), COLOURS[0], dash)
        _plot(axes[0, 0], run, "train_loss_border", label("border"), COLOURS[1], dash)
        for i, part in enumerate(("cells", "sprites", "sprite_move", "sprite_place", "sprite_pixels", "camera")):
            _plot(axes[0, 0], run, f"train_loss_{part}", label(part.replace("_", " ")), COLOURS[i], dash)
        for i, h in enumerate((1, 4, 16)):
            _plot(axes[0, 1], run, f"{GAME}_h{h}_wrong_changed", label(f"+{h}"), COLOURS[i], dash)
        for i, h in enumerate((1, 16)):
            _plot(axes[1, 0], run, f"{GAME}_h{h}_border_wrong", label(f"+{h}"), COLOURS[i], dash)
            if n < len(runs):
                _plot(axes[1, 0], run, f"{GAME}_h{h}_border_copy", label(f"+{h} copying"), COLOURS[i], ":",
                      raw=False)
        _plot(axes[1, 1], run, f"{GAME}_h16_wrong_static", label("static +16"), COLOURS[3], dash)
        _plot(axes[1, 1], run, "blend_frames", label("blend radius"), COLOURS[4], dash)
        for v, (variant, d) in enumerate(run.long_dream.items()):
            for i, h in enumerate((16, 128)):
                axes[2, 0].plot(d["step"], d[f"wrong_{h}"], color=COLOURS[(2 * v + i) % len(COLOURS)],
                                linestyle=dash, marker=".", label=label(f"{variant} +{h}"))
            if "timer_wrong_128" in d:
                axes[2, 0].plot(d["step"], d["timer_wrong_128"], color=COLOURS[(2 * v + 4) % len(COLOURS)],
                                linestyle=dash, marker="+", alpha=0.7, label=label(f"{variant} timer +128"))
            for i, key in enumerate(("unseen_patch", "unseen_change")):        # generic: not like real play
                if key in d:
                    axes[2, 0].plot(d["step"], d[key], color=COLOURS[(2 * v + 2 + i) % len(COLOURS)],
                                    linestyle=dash, marker="dx"[i], alpha=0.7, label=label(f"{variant} {key}"))
            axes[2, 1].plot(d["step"], d["mass"], color=COLOURS[v % len(COLOURS)], linestyle=dash, marker=".",
                            label=label(f"{variant} mass"))
            if "piece_32" in d:
                axes[2, 1].plot(d["step"], d["piece_32"], color=COLOURS[v % len(COLOURS)], linestyle=dash,
                                marker="x", alpha=0.7, label=label(f"{variant} piece +32"))
            for key, marker in (("piece_hit_32", "o"), ("fall", "^"), ("spawn", "s"), ("presence", "P")):
                if key in d:
                    axes[2, 1].plot(d["step"], d[key], color=COLOURS[(v + 2 + "o^sP".index(marker)) % len(COLOURS)],
                                    linestyle=dash, marker=marker, alpha=0.7, label=label(f"{variant} {key}"))
    fig.suptitle("   ".join(f"{'solid' if dash == '-' else {'--': 'dashed', ':': 'dotted', '-.': 'dash-dot'}[dash]}: "
                           f"{run.name}" for run, dash in every), color=TEXT, fontsize=9)
    axes[2, 1].axhline(1.0, color=TEXT, linewidth=0.8, linestyle=":")
    _style(axes[0, 0], "Train loss (masked pixels)", "cross-entropy", log=True)
    _style(axes[0, 1], "Preview: changed game pixels wrong (copying = 100%)", "fraction")
    _style(axes[1, 0], "Preview: border pixels wrong vs copying the last frame (thin dotted)", "fraction", log=True)
    _style(axes[1, 1], "Static pixels wrong at +16; blend radius (frames)", "fraction", log=True)
    _style(axes[2, 0], "Long dream: playfield pixels wrong (no buttons); patches/changes unseen in real play",
           "fraction")
    _style(axes[2, 1], "Long dream: block mass at +128, falling piece kept at +32 (1.0 = real); presence "
                       "(any piece drawn; falling toward 0 = stalling)", "ratio")
    for ax in axes.flat:
        _legend(ax)
    fig.tight_layout()


def draw_speed(fig: Figure, runs: list[RunData]) -> None:
    fig.clear()
    fig.set_facecolor(BG)
    rate_ax, wait_ax = fig.subplots(2, 1)
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
    _style(rate_ax, "Training speed", "steps/s")
    _style(wait_ax, "Waiting for data per step", "ms")
    for ax in (rate_ax, wait_ax):
        _legend(ax)
    fig.tight_layout()


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
    fig.tight_layout()


# ---------------------------------------------------------------------------------------------- widgets
class GifPlayer(ttk.Label):
    """An animated GIF from a file the trainer replaces: reloaded when it changes, scaled to `width`."""

    def __init__(self, master, width: int, **kwargs):
        super().__init__(master, **kwargs)
        self.width, self.path, self.stamp, self.frames, self.at, self.job = width, None, None, [], 0, None

    def show(self, path: Path) -> None:
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return
        if path == self.path and stamp == self.stamp:
            return
        try:
            image = Image.open(io.BytesIO(path.read_bytes()))
            frames = []
            for frame in ImageSequence.Iterator(image):
                rgb = frame.convert("RGB")
                height = round(rgb.height * self.width / rgb.width)
                frames.append((ImageTk.PhotoImage(rgb.resize((self.width, height), Image.NEAREST)),
                               frame.info.get("duration", 100)))
        except (OSError, ValueError):
            return                                                      # caught mid-replace: next refresh
        self.path, self.stamp, self.frames, self.at = path, stamp, frames, 0
        if self.job is None:
            self._next()

    def _next(self) -> None:
        if self.frames:
            photo, ms = self.frames[self.at % len(self.frames)]
            self.configure(image=photo)
            self.at += 1
        self.job = self.after(ms if self.frames else 500, self._next)


class RunViewer:
    """The tabs, inside `master`; refresh() reloads what changed (also on a timer)."""

    def __init__(self, master, runs: list[Path], compare: list[Path] = (), refresh_ms: int = 10_000):
        self.runs = [RunData(p) for p in runs]
        self.compare = [RunData(p) for p in compare if p.exists()]
        self.refresh_ms, self.master = refresh_ms, master
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True)
        curves = self._tab("Curves")
        bar = ttk.Frame(curves)
        bar.pack(fill="x", pady=2)
        self.show_compare = tk.BooleanVar(value=True)
        if self.compare:
            ttk.Checkbutton(bar, text="show compared runs (dashed): " + ", ".join(r.name for r in self.compare),
                            variable=self.show_compare, command=lambda: self.refresh(force=True)).pack(side="left",
                                                                                                    padx=6)
        self.curves_fig, self.curves_canvas = self._figure(None, (12, 8), curves)
        previews = self._tab("Previews")
        bar = ttk.Frame(previews)
        bar.pack(fill="x", pady=4)
        ttk.Label(bar, text="Run").pack(side="left", padx=6)
        self.selected = tk.StringVar(value=self.runs[0].name)
        ttk.Combobox(bar, state="readonly", values=[r.name for r in self.runs], textvariable=self.selected,
                     width=40).pack(side="left")
        ttk.Label(bar, text="  last context frame, then 16 generated from the real actions; most changing "
                            "window first", style="PanelTitle.TLabel").pack(side="left")
        grid = ttk.Frame(previews)
        grid.pack(fill="both", expand=True)
        self.previews = [GifPlayer(grid, 760) for _ in range(4)]
        for i, player in enumerate(self.previews):
            player.grid(row=i // 2, column=i % 2, padx=4, pady=4)
        dream_tab = self._tab("Long dream")
        ttk.Label(dream_tab, style="PanelTitle.TLabel", text="Latest test: real playfield next to each way of "
                  "dreaming, nobody pressing anything for 128 frames (averaged weights)").pack(pady=4)
        self.dream_gif = GifPlayer(dream_tab, 900)
        self.dream_gif.pack()
        self.dream_text = tk.StringVar()
        ttk.Label(dream_tab, textvariable=self.dream_text, style="PanelTitle.TLabel", justify="left",
                  font=("Consolas", 10)).pack(anchor="w", padx=12, pady=8)
        if self.compare:
            compare_tab = self._tab("Compare")
            ttk.Label(compare_tab, style="PanelTitle.TLabel", text="Long dreams at the steps both runs were tested "
                      "(the same 32 games for the same --seed): this run / compared run").pack(anchor="w", padx=12,
                                                                                           pady=6)
            self.compare_text = tk.StringVar()
            ttk.Label(compare_tab, textvariable=self.compare_text, style="PanelTitle.TLabel", justify="left",
                      font=("Consolas", 10)).pack(anchor="nw", padx=12)
        scenarios_tab = self._tab("Scenarios")
        ttk.Label(scenarios_tab, style="PanelTitle.TLabel", text="Game events (scripts/tetris_scenarios.py) and long events (scripts/event_checks.py), run on "
                  "checkpoints: the latest check of each run"
                  ).pack(anchor="w", padx=12, pady=6)
        self.scenario_text = tk.StringVar()
        ttk.Label(scenarios_tab, textvariable=self.scenario_text, style="PanelTitle.TLabel", justify="left",
                  font=("Consolas", 10)).pack(anchor="nw", padx=12)
        self.horizon_fig, self.horizon_canvas = self._figure("Horizons", (12, 7))
        self.speed_fig, self.speed_canvas = self._figure("Speed", (12, 6))
        stats = self._tab("Stats")
        self.stats_text = tk.StringVar()
        ttk.Label(stats, textvariable=self.stats_text, style="PanelTitle.TLabel", justify="left",
                  font=("Consolas", 10)).pack(anchor="nw", padx=12, pady=12)
        self.job = None
        self.refresh(force=True)

    def _tab(self, name: str) -> ttk.Frame:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name)
        return frame

    def _figure(self, name: str | None, size, tab: ttk.Frame | None = None) -> tuple[Figure, FigureCanvasTkAgg]:
        """A chart in a new tab `name`, or in `tab`."""
        fig = Figure(figsize=size, dpi=100, facecolor=BG)
        canvas = FigureCanvasTkAgg(fig, master=tab if tab is not None else self._tab(name))
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas

    def run(self) -> RunData:
        return next((r for r in self.runs if r.name == self.selected.get()), self.runs[0])

    def refresh(self, force: bool = False) -> None:
        if self.job is not None:
            self.master.after_cancel(self.job)
        changed = any([r.refresh() for r in self.runs + self.compare]) or force
        run = self.run()
        for i, player in enumerate(self.previews):
            player.show(run.path / "previews" / f"{GAME}_{i}.gif")
        self.dream_gif.show(run.path / "long_dream.gif")
        if changed:
            draw_curves(self.curves_fig, self.runs, self.compare if self.show_compare.get() else [])
            self.curves_canvas.draw_idle()
            draw_horizons(self.horizon_fig, run)
            self.horizon_canvas.draw_idle()
            draw_speed(self.speed_fig, self.runs)
            self.speed_canvas.draw_idle()
            self.dream_text.set(long_dream_table(self.runs + self.compare))
            self.stats_text.set(stats_text(run))
            self.scenario_text.set(scenario_table([run] + self.compare) + "\n\n" + event_table([run] + self.compare))
            if self.compare:
                self.compare_text.set("\n\n".join(compare_table(run, other) for other in self.compare))
        self.job = self.master.after(self.refresh_ms, self.refresh)


BIASED = (("chosen", "bias_choices_dream", "{:.0f}"), ("real chose", "bias_choices_real", "{:.0f}"),
          ("types", "bias_types_dream", "{:.0f}"), ("top share", "bias_top_share_dream", "{:.2f}"),
          ("mix distance", "bias_type_distance", "{:.2f}"), ("garbled", "bias_garbled", "{:.0%}"))
COMPARED = (("presence", "presence", "{:.2f}"), ("activation", "activation", "{:.2f}"),
            ("hit +32", "piece_hit_32", "{:.2f}"), ("ghost", "piece_ghost_32", "{:.2f}"), ("fall", "fall", "{:.2f}"),
            ("wrong +16", "wrong_16", "{:.1%}"), ("wrong +128", "wrong_128", "{:.1%}"),
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
    mine, theirs = next(iter(run.long_dream.values())), next(iter(other.long_dream.values()))
    where = {int(s): i for i, s in enumerate(theirs["step"])}
    width = 15
    lines = [f"each cell: {run.name} / {other.name}", "",
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
    lines += ["", "next pieces the dreams chose (bias test): " + run.name + " / " + other.name, "",
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
             "  hit+32  fall  spawn  presence  unseen patch/change"]
    for run in runs:
        for variant, d in run.long_dream.items():
            for i in range(max(0, len(d["step"]) - last), len(d["step"])):
                lines.append(f"{run.name[:24]:24s} {variant:12s} {int(d['step'][i]):>7,}   "
                             + "  ".join(f"{d[f'wrong_{h}'][i]:6.2%}" for h in (16, 32, 64, 128))
                             + f"   {d['mass'][i]:.2f}  "
                             + "  ".join(f"{d.get(k, np.full(len(d['step']), np.nan))[i]:6.2f}"
                                         for k in ("piece_32", "piece_hit_32", "fall", "spawn", "presence"))
                             + "  " + "/".join(f"{d.get(k, np.full(len(d['step']), np.nan))[i]:6.2%}"
                                               for k in ("unseen_patch", "unseen_change")))
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
