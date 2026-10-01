"""Dark ttk theme for the training dashboard (train_ui.py).

Split out of train_ui.py: pure tkinter styling with no training/state logic,
so it can be read, changed, or reused independently of the TrainUI class.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

BG_DARK = "#131316"
CARD_DARK = "#1a1a20"
BORDER_DARK = "#2a2a36"
TEXT_PRIMARY = "#f2f2f8"
TEXT_MUTED = "#8b8b9e"
ACCENT_BLUE = "#00b4d8"
ACCENT_GREEN = "#00e676"


def setup_dark_theme(root: tk.Widget) -> ttk.Style:
    """Configures the dark ttk theme in-place on `root` and returns the ttk.Style."""
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    root.configure(bg=BG_DARK)
    style.configure(".", background=BG_DARK, foreground=TEXT_PRIMARY)
    style.configure("TFrame", background=BG_DARK)
    style.configure("TLabel", background=BG_DARK, foreground=TEXT_PRIMARY)
    style.configure("Muted.TLabel", background=BG_DARK, foreground=TEXT_MUTED, font=("Segoe UI", 9))
    style.configure("Header.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=("Segoe UI", 11, "bold"))
    style.configure("State.TLabel", background=BG_DARK, foreground=ACCENT_GREEN, font=("Segoe UI", 9, "bold"))
    style.configure("MetricVal.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=("Consolas", 9))
    style.configure("PanelTitle.TLabel", background=BG_DARK, foreground=TEXT_MUTED, font=("Segoe UI", 8))
    style.configure("RolloutTitle.TLabel", background=BG_DARK, foreground=TEXT_MUTED, font=("Segoe UI", 9, "bold"))
    style.configure("RolloutStep.TLabel", background=BG_DARK, foreground=ACCENT_BLUE, font=("Consolas", 10, "bold"))

    style.configure(
        "TButton", background="#252530", foreground="#ffffff",
        bordercolor=BORDER_DARK, focuscolor=ACCENT_BLUE, padding=[8, 4], relief="flat",
    )
    style.map("TButton",
        background=[("active", "#323242"), ("pressed", "#1c1c24")],
        foreground=[("active", "#ffffff")],
    )
    style.configure(
        "Stop.TButton", background="#3a161a", foreground="#ff7070",
        bordercolor="#542026", padding=[10, 4], relief="flat", font=("Segoe UI", 9, "bold"),
    )
    style.map("Stop.TButton",
        background=[("active", "#501c22"), ("pressed", "#2a1014")],
        foreground=[("active", "#ff9090")],
    )
    style.configure(
        "PlayBtn.TButton", background="#1a2030", foreground=ACCENT_BLUE,
        padding=[6, 2], relief="flat",
    )
    style.configure(
        "TNotebook", background=BG_DARK, bordercolor=BORDER_DARK, tabmargins=[2, 4, 2, 0],
    )
    style.configure(
        "TNotebook.Tab", background=CARD_DARK, foreground=TEXT_MUTED,
        bordercolor=BORDER_DARK, padding=[12, 6], font=("Segoe UI", 9),
    )
    style.map("TNotebook.Tab",
        background=[("selected", "#262633"), ("active", "#202028")],
        foreground=[("selected", "#ffffff"), ("active", "#dcdce8")],
        bordercolor=[("selected", ACCENT_BLUE)],
    )

    # Entry/Combobox/Checkbutton/LabelFrame -- the settings form's actual
    # input widgets (world_model/ui/settings_panel.py). The base "." rule
    # above sets foreground=TEXT_PRIMARY (near-white) globally, which
    # cascades onto these too, but "clam"'s own default fieldbackground for
    # Entry/Combobox stays its stock near-white -- so without setting
    # fieldbackground explicitly here, every text field in the settings
    # panel renders near-white text on a near-white field: unreadable.
    style.configure(
        "TEntry", fieldbackground=CARD_DARK, foreground=TEXT_PRIMARY,
        insertcolor=TEXT_PRIMARY, bordercolor=BORDER_DARK, padding=[4, 3],
    )
    style.map("TEntry",
        fieldbackground=[("disabled", BG_DARK), ("readonly", CARD_DARK)],
        foreground=[("disabled", TEXT_MUTED)],
    )
    style.configure(
        "TCombobox", fieldbackground=CARD_DARK, background=CARD_DARK,
        foreground=TEXT_PRIMARY, arrowcolor=TEXT_PRIMARY, bordercolor=BORDER_DARK, padding=[4, 3],
    )
    style.map("TCombobox",
        fieldbackground=[("readonly", CARD_DARK), ("disabled", BG_DARK)],
        foreground=[("disabled", TEXT_MUTED)],
        background=[("active", "#252530")],
    )
    # ttk.Combobox's dropdown list is a plain Tk Listbox, not a themed ttk
    # widget -- style.configure/.map above never reaches it. Tk's
    # option database is the only way to set its colors, and it must be set
    # on `root` (the actual Tk() instance) before any Combobox is opened.
    root.option_add("*TCombobox*Listbox.background", CARD_DARK)
    root.option_add("*TCombobox*Listbox.foreground", TEXT_PRIMARY)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_BLUE)
    root.option_add("*TCombobox*Listbox.selectForeground", "#000000")
    style.configure(
        "TCheckbutton", background=BG_DARK, foreground=TEXT_PRIMARY,
        focuscolor=ACCENT_BLUE,
    )
    style.map("TCheckbutton",
        background=[("active", BG_DARK)],
        foreground=[("disabled", TEXT_MUTED)],
        indicatorcolor=[("selected", ACCENT_BLUE), ("!selected", CARD_DARK)],
    )
    style.configure(
        "TLabelframe", background=BG_DARK, bordercolor=BORDER_DARK, relief="groove",
    )
    style.configure(
        "TLabelframe.Label", background=BG_DARK, foreground=TEXT_MUTED, font=("Segoe UI", 9, "bold"),
    )
    style.configure(
        "TScrollbar", background=CARD_DARK, troughcolor=BG_DARK,
        bordercolor=BORDER_DARK, arrowcolor=TEXT_MUTED,
    )
    style.map("TScrollbar", background=[("active", "#323242")])
    return style
