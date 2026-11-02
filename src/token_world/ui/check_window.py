"""A check's window is a Tk app; a failure in it crashes the process; a check that finishes closes it (2026-10-30).

Every window is Tk, as the trainer's is (the user, 2026-10-30: "never use anything other than tk"): a tk.Tk root
with the dark ttk theme (ui/theme.py), a status line, and the check's charts as a matplotlib Figure drawn into it
with FigureCanvasTkAgg (as ui/run_viewer.py draws the trainer's); never a pyplot window.

The user, 2026-10-25, after a traceback sat cut to 170 characters in a check's one-line status and the process
exited 0: "YOU CANNOT LET A UI SWALLOW AN ERROR". Then 2026-10-30, after a check showed its traceback in red and
sat waiting for its window to be closed: "Do not fail into the UI only that's insane"; "On a fail you must crash
always". So every error in a check's work thread, a window's refresh or a Tk callback goes to crash():

- the whole traceback to stderr, at once;
- appended to <out>/error.txt when there is an out folder;
- the process exits 1 there and then (os._exit: from any thread, with the window still open; nothing waits on
  a person).

And a check that finishes closes its own window (after one last refresh draws and saves the end) and exits 0
(the same day: "the UI doesn't exit so you wouldn't fucking know"): nothing waits on a person either way.

    fig = Figure(figsize=(16, 9))
    window = CheckWindow(fig, "SMB button check", out)
    window.thread(work)                 # work() raises; no try/except of its own
    window.every(500, refresh)          # refresh() raises; no after() of its own
    window.run()                        # returns when work() has returned (or the process has crashed)

Other Tk apps (the trainer's window, the play app): tk_catch_callbacks(root), and crash(text) for an error a
worker reports.
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path


def crash(text: str | None = None, out: Path | None = None, title: str = "") -> None:
    """The process ends here with exit code 1: the traceback (the exception being handled if no text) to stderr
    and to out/error.txt. Never returns."""
    text = text or traceback.format_exc()
    print(f"{title + ' ' if title else ''}FAILED:\n{text}", file=sys.stderr, flush=True)
    if out is not None:
        try:
            with (Path(out) / "error.txt").open("a", encoding="utf-8") as f:   # torch's messages are not cp1252
                print(text, file=f)
        except OSError as e:
            print(f"(error.txt not written: {e})", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(1)


class CheckWindow:
    """A check's Tk window: the dark theme, a status line (status()) and the Figure. When work() returns, the
    next refresh draws (and saves) the final state and the window closes, so run() returns and the process exits
    0; a failure exits 1 at once (crash)."""

    def __init__(self, fig, title: str, out: Path):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        from token_world.ui.theme import setup_dark_theme
        self.fig, self.title, self.out = fig, title, Path(out)
        self.finished = threading.Event()
        self.refreshing = False
        self.root = tk.Tk()
        self.root.title(title)
        setup_dark_theme(root := self.root)
        tk_catch_callbacks(root, self.out)
        width, height = fig.get_size_inches() * fig.dpi
        root.geometry(f"{min(int(width), root.winfo_screenwidth() - 40)}x"
                      f"{min(int(height) + 40, root.winfo_screenheight() - 80)}")
        self.line = tk.StringVar(value="starting")
        ttk.Label(root, textvariable=self.line, style="Muted.TLabel").pack(fill="x", padx=8, pady=(4, 2))
        self.canvas = FigureCanvasTkAgg(fig, master=root)          # fig.canvas is this from here on
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def status(self, text: str) -> None:
        """The status line (from the Tk thread: a refresh)."""
        self.line.set(text)

    def fail(self, text: str | None = None) -> None:
        """A failure, from any thread: the process crashes (crash)."""
        crash(text, self.out, self.title)

    def thread(self, work) -> threading.Thread:
        """Run work() on a daemon thread; whatever it raises crashes the process; when it returns, the window
        closes after one more refresh (every), or at once if there is none."""
        def run() -> None:
            try:
                work()
            except BaseException:
                self.fail()
            self.finished.set()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        self._watch()
        return t

    def _watch(self) -> None:
        """Without every(): close the window once work() has returned."""
        def check() -> None:
            if self.refreshing:                       # every() closes it after its last refresh
                return
            if self.finished.is_set():
                self._close()
            else:
                self.root.after(200, check)
        self.root.after(200, check)

    def _close(self) -> None:
        print(f"{self.title}: done, closing its window", flush=True)
        self.root.destroy()

    def every(self, ms: int, refresh) -> None:
        """Call refresh() every `ms` on the Tk loop; whatever it raises crashes the process. The refresh after
        work() returns is the last: then the window closes."""
        self.refreshing = True

        def tick() -> None:
            last = self.finished.is_set()
            try:
                refresh()
            except BaseException:
                self.fail()
            if last:
                self._close()
                return
            self.root.after(ms, tick)
        self.root.after(ms, tick)

    def run(self) -> None:
        """The Tk loop, until the check has finished (or crashed)."""
        self.root.mainloop()


def tk_catch_callbacks(root, out: Path | None = None) -> None:
    """A Tk callback's error (Tk itself only prints it and carries on) crashes the process."""
    root.report_callback_exception = lambda kind, value, tb: crash(
        "".join(traceback.format_exception(kind, value, tb)), out, root.title())
