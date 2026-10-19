"""Widgets for the run window (ui/run_viewer.py) that follow the window's size (2026-10-12).

- FitImage: a still or animated image file (PNG, GIF) the trainer keeps replacing, drawn as large as its space
  allows without cropping (fit: whole-number scales from 2x, a fraction below; shrunk by averaging, so one wrong
  pixel still shows faintly instead of being dropped with its row); re-fitted when the window changes size,
  reloaded when the file changes. FitGrid: several of them, in the columns that draw each largest.
- split: a PanedWindow's first pane given a share of the height once it is drawn (pane weights only share
  later growth, so a table asking for many lines would otherwise take most of the tab).
- TextPanel: read-only monospaced text with both scrollbars (wide tables scroll instead of being cut off), its
  scroll place kept across refreshes.
- Explainer: a paragraph of muted text that wraps to its width.
"""
from __future__ import annotations

import io
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageSequence, ImageTk

from token_world.ui.theme import BG_DARK, CARD_DARK, TEXT_MUTED, TEXT_PRIMARY


def fit(width: int, height: int, room_w: int, room_h: int) -> tuple[int, int]:
    """An image's size scaled to fit the room: from twice its size up a whole-number scale (pixel art stays crisp),
    below that the fraction that fits (bigger matters more there than even pixels)."""
    scale = min(room_w / width, room_h / height)
    if scale >= 2:
        scale = int(scale)
    return max(1, round(width * scale)), max(1, round(height * scale))


def best_columns(count: int, width: int, height: int, room_w: int, room_h: int) -> int:
    """How many columns to lay `count` images of one size in, so each is drawn largest in the room."""
    def scale(columns):
        rows = -(-count // columns)
        return min(room_w / columns / width, room_h / rows / height)
    return max((c for c in range(1, count + 1) if count % c == 0 or c == count), key=scale)


class FitImage(tk.Canvas):
    """A still or animated image from a file, fitted to the canvas (module docstring). `empty`: the text shown
    until the file exists."""

    def __init__(self, master, empty: str = "", **kwargs):
        super().__init__(master, background=BG_DARK, highlightthickness=0, **kwargs)
        self.empty = empty
        self.path: Path | None = None
        self.stamp = None
        self.source: list[tuple[Image.Image, int]] = []           # frames at full size, and their milliseconds
        self.shown: list[ImageTk.PhotoImage] = []                  # the frames at the current fit
        self.size = (0, 0)
        self.at, self.job, self.resize_job = 0, None, None
        self.bind("<Configure>", self._schedule_fit)

    def show(self, path: Path) -> None:
        """Show `path`, reloaded only when it changed (or another path)."""
        path = Path(path)
        try:
            stamp = path.stat().st_mtime
        except OSError:
            if path != self.path:
                self.path, self.stamp, self.source = path, None, []
                self._draw_empty()
            return
        if path == self.path and stamp == self.stamp:
            return
        try:
            image = Image.open(io.BytesIO(path.read_bytes()))
            source = [(frame.convert("RGB"), frame.info.get("duration", 100))
                      for frame in ImageSequence.Iterator(image)]
        except (OSError, ValueError):
            return                                                  # caught mid-replace: the next refresh
        self.path, self.stamp, self.source = path, stamp, source
        self.size = (0, 0)
        self._fit()

    def _draw_empty(self) -> None:
        self.delete("all")
        self.shown = []
        if self.empty:
            self.create_text(self.winfo_width() // 2, self.winfo_height() // 2, text=self.empty, fill=TEXT_MUTED,
                             font=("Segoe UI", 10), width=max(200, self.winfo_width() - 40))

    def _schedule_fit(self, _event=None) -> None:
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(60, self._fit)

    def _fit(self) -> None:
        self.resize_job = None
        if not self.source:
            self._draw_empty()
            return
        room = (self.winfo_width(), self.winfo_height())
        if room[0] < 16 or room[1] < 16:
            return
        first = self.source[0][0]
        size = fit(first.width, first.height, *room)
        if size != self.size:
            self.size = size
            how = Image.Resampling.BOX if size[0] < first.width else Image.Resampling.NEAREST
            self.shown = [ImageTk.PhotoImage(frame.resize(size, how)) for frame, _ in self.source]
            self.at = 0
        self.delete("all")
        self.item = self.create_image(room[0] // 2, room[1] // 2, image=self.shown[0])
        if self.job is None and len(self.shown) > 1:
            self._next()

    def _next(self) -> None:
        self.job = None
        if len(self.shown) < 2 or not self.winfo_exists():
            return
        self.at = (self.at + 1) % len(self.shown)
        self.itemconfigure(self.item, image=self.shown[self.at])
        self.job = self.after(self.source[self.at][1] or 100, self._next)


class TextPanel(ttk.Frame):
    """Read-only monospaced text with both scrollbars (module docstring)."""

    def __init__(self, master, font=("Consolas", 10), height: int = 24, **kwargs):
        super().__init__(master, **kwargs)
        self.text = tk.Text(self, wrap="none", font=font, height=height, background=CARD_DARK, foreground=TEXT_PRIMARY,
                            insertbackground=TEXT_PRIMARY, relief="flat", borderwidth=0, padx=10, pady=8,
                            state="disabled", highlightthickness=0)
        down = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        across = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=down.set, xscrollcommand=across.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        down.grid(row=0, column=1, sticky="ns")
        across.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.current = None

    def set(self, text: str) -> None:
        if text == self.current:
            return
        self.current = text
        y, x = self.text.yview()[0], self.text.xview()[0]
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.configure(state="disabled")
        self.text.yview_moveto(y)
        self.text.xview_moveto(x)


class Explainer(ttk.Label):
    """A paragraph of muted text that wraps to its width."""

    def __init__(self, master, text: str = "", **kwargs):
        super().__init__(master, text=text, style="Explain.TLabel", justify="left", **kwargs)
        self.bind("<Configure>", lambda e: self.configure(wraplength=max(200, e.width - 8)))


class FitGrid(ttk.Frame):
    """Up to `most` FitImages, as many as show() is given, laid out again at every size in the columns that draw
    each largest (best_columns, for the first image's size)."""

    def __init__(self, master, most: int, empty: str = "", **kwargs):
        super().__init__(master, **kwargs)
        self.images = [FitImage(self, empty=empty) for _ in range(most)]
        self.count, self.layout = 0, None
        self.bind("<Configure>", lambda e: self.arrange())

    def show(self, paths: list[Path]) -> None:
        self.count = len(paths)
        for image, path in zip(self.images, paths):
            image.show(path)
        self.arrange()

    def arrange(self) -> None:
        source = next((i.source[0][0] for i in self.images[:self.count] if i.source), None)
        width, height = (source.width, source.height) if source is not None else (4, 1)
        columns = best_columns(self.count, width, height, max(self.winfo_width(), 1), max(self.winfo_height(), 1))
        if (self.count, columns) == self.layout:
            return
        self.layout = (self.count, columns)
        rows = -(-self.count // columns)
        for i in range(len(self.images) + 1):
            self.columnconfigure(i, weight=1 if i < columns else 0)
            self.rowconfigure(i, weight=1 if i < rows else 0)
        for i, image in enumerate(self.images):
            if i < self.count:
                image.grid(row=i // columns, column=i % columns, sticky="nsew", padx=3, pady=3)
            else:
                image.grid_remove()


def split(panes: ttk.PanedWindow, share: float) -> None:
    """Put `panes`' first sash at `share` of its height once it has one (module docstring)."""
    def place(event):
        if event.height > 50:
            panes.sashpos(0, int(event.height * share))
            panes.unbind("<Configure>", bound)
    bound = panes.bind("<Configure>", place, add="+")
