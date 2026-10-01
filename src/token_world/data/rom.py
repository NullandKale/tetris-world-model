"""Where the Tetris ROM is. No ROM is in this repository or shipped with it.

Point TETRIS_ROM at your own copy of Tetris (USA).nes, or pass a folder holding roms/Tetris (USA).nes
(--tetris-repo). Without either, the ROM is looked for beside this project, in
../llm-thing19/roms/, where the research setup keeps it.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
NAME = "Tetris (USA).nes"


def tetris_rom(repo: str | None = None) -> Path:
    if repo:
        return Path(repo) / "roms" / NAME
    if os.environ.get("TETRIS_ROM"):
        return Path(os.environ["TETRIS_ROM"])
    return PROJECT.parent / "llm-thing19" / "roms" / NAME
