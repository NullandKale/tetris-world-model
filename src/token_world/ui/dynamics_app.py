"""The training window for the world models (scripts/train_dynamics_ui.py).

A status line and Stop Training over the run viewer (ui/run_viewer.py), which reads everything back
from the run folders; the trainer's events only update the status and ask the viewer to refresh.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from token_world.ui.run_viewer import RunViewer
from token_world.ui.theme import setup_dark_theme


def launch(args, train_fn) -> None:
    """Open the window, run train_fn(args, events, stop) on a thread, block until closed."""
    root = tk.Tk()
    root.title(f"World models ({', '.join(args.models)}) - live {', '.join(args.games)} histories")
    root.geometry("1600x980")
    root.minsize(1100, 700)
    setup_dark_theme(root)
    status = tk.StringVar(value="Starting live histories...")
    header = ttk.Frame(root)
    header.pack(fill="x", padx=8, pady=5)
    ttk.Label(header, textvariable=status, style="Header.TLabel").pack(side="left")
    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    ttk.Button(header, text="Stop Training", command=stop.set, style="Stop.TButton").pack(side="right")
    viewer = RunViewer(root, list(args.outputs.values()), args.compare)
    worker = threading.Thread(target=train_fn, args=(args, events, stop), daemon=True, name="world-model-trainer")
    worker.start()
    poll_id = None
    latest: dict[str, str] = {}

    def close() -> None:
        stop.set()
        if poll_id is not None:
            root.after_cancel(poll_id)
        root.destroy()

    def update() -> None:
        nonlocal poll_id
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "ready":
                status.set(f"{event[2]} live histories; " + ", ".join(
                    f"{name} {n:,} parameters" for name, n in event[1].items()) + "; compiling, then training")
            elif event[0] == "preview":
                name, step, m = event[1:]
                latest[name] = (f"{name} @ {step:,}: changed pixels wrong +1/+4/+16 " + " / ".join(
                    f"{m[f'tetris_h{h}_wrong_changed']:.0%}" for h in (1, 4, 16)))
                status.set("   |   ".join(latest.values()))
                viewer.refresh()
            elif event[0] == "longdream":
                name, step, summary = event[1:]
                latest[name + " dream"] = f"{name} long dream @ {step:,}: " + "; ".join(
                    f"{v} +128 {r['wrong_128']:.1%} mass {r['mass']:.2f}" for v, r in summary.items())
                status.set("   |   ".join(latest.values()))
                viewer.refresh()
            elif event[0] == "done":
                status.set("Finished at " + ", ".join(f"{name} step {s:,}" for name, s in event[1].items()))
                if args.close_when_done:
                    root.after(3000, close)
            elif event[0] == "error":
                status.set(f"Training error: {event[1]}")
                if args.close_when_done:
                    root.after(3000, close)
        if root.winfo_exists():
            poll_id = root.after(200, update)

    root.protocol("WM_DELETE_WINDOW", close)
    poll_id = root.after(200, update)
    root.mainloop()
    stop.set()
    worker.join()
