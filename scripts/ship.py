"""Assemble the public repository (site/, its own git checkout): the world model's code and the browser page.

    python scripts/export_onnx.py [checkpoint.pt]     # web/model/
    python scripts/ship.py                            # site/
    cd site; git add -A; git commit -m ...; git push  # GitHub Pages publishes web/ (Actions workflow)

Ships the code that makes, checks and exports the model: everything the ENTRIES import, followed through
token_world and the sibling scripts, plus the page (web/, with the exported model but not reference.bin,
which only the frame-for-frame checks read; export_onnx.py writes it). Not shipped: World NES (the
emulator, a separate package), any ROM (never in this repository: data/rom.py finds the user's own), and
the Contra port (SKIP). A ROM-like file anywhere in site/ stops the build. site/.git is kept.
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ["scripts/train_layered.py", "scripts/export_onnx.py", "scripts/ship.py", "scripts/play_world_model.py",
           "scripts/long_dream_check.py", "scripts/event_checks.py", "scripts/view_runs.py", "scripts/tetris_scenarios.py",
           "tests/test_layered.py", "tests/test_nes_layers.py", "tests/test_onnx.py", "tests/test_worlds.py",
           "tests/test_world_nes_tetris.py", "tests/test_tetris_events.py", "tests/test_tetris_bot.py",
           "tests/test_tetris_scenarios.py", "tests/test_long_dream.py", "tests/test_coherence.py", "tests/test_bias.py",
           "tests/test_restart.py", "tests/test_run_outputs.py", "tests/test_dynamics.py"]
EXTRA = ["scripts/run_until_stopped.ps1"]
SKIP = ("contra", "smb")                        # module names: the Contra port and SMB work are not shipped
PAGE = ["index.html", "play.js", "dreamer.js", "check_browser.mjs", "test_dreamer.mjs", "package.json",
        "package-lock.json"]
MODEL = ["embed.onnx", "prefill.onnx", "step.onnx", "model.json", "context.bin", "context.json"]
ROMS = (".nes", ".fds", ".unf", ".unif")

WORKFLOW = """# Publishes web/ to GitHub Pages on every push to main (Pages source: GitHub Actions).
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: web
      - id: deployment
        uses: actions/deploy-pages@v4
"""

GITIGNORE = """__pycache__/
*.py[cod]
output/
web/node_modules/
web/model/reference.bin
web/profile.log
# game ROMs are never published
*.nes
*.fds
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tetris-world-model"
version = "0.1.0"
description = "A world model of NES Tetris on exact palette pixels, playable in the browser"
requires-python = ">=3.10"
dependencies = ["torch", "numpy", "pillow", "matplotlib"]

[project.optional-dependencies]
onnx = ["onnx", "onnxruntime", "onnxscript"]   # scripts/export_onnx.py

[tool.setuptools.packages.find]
where = ["src"]
"""

README = """# Tetris world model

A {params}-parameter world model of NES Tetris over the NES's own layers: the background as cells fixed to
the world, the sprites as their own picture, and the border, 543 tokens a frame of exact palette pixels in
16 x 16 patches. A spatiotemporal MaskGIT transformer (64-frame windows, rotary time) generates every frame
from the frames before it and the controller; no emulator runs while it plays. Trained by next-frame
prediction from clean real context, 58,000 steps ending in a learning-rate decay.

**Play it:** the page in `web/` (published by GitHub Pages) runs the step-{step:,} checkpoint with
onnxruntime-web on WebGPU (Chrome or Edge; elsewhere a slow WebAssembly fallback), from the start of a
level-0 game (one frame's layers). Arrows move and drop, X / Z rotate, Enter is Start; on a phone, the
on-screen controller. The dream holds the playfield's camera, so it stays on the game screen.

## The code

- `src/token_world/models/layered.py`, `layered_pixels.py`: the model, its training mask and loss, the
  cached `Dreamer`; `src/token_world/data/nes_layers.py`: the layers, composed back into the screen exactly.
- `scripts/train_layered.py`: training on live emulator histories (a RAM-reading bot plays), with a run
  viewer; `scripts/run_until_stopped.ps1` runs it unattended.
- `src/token_world/diagnostics/`: 128-frame no-button dreams against the real game (`long_dream.py`), game
  events (`tetris_scenarios.py`), line clears and restarts followed to their end (`line_clears.py`,
  `restart.py`), the next piece's choices (`bias.py`) and game-agnostic coherence scores (`coherence.py`),
  for any model through `worlds.py`.
- `scripts/export_onnx.py`, `src/token_world/models/onnx_export.py`, `web/`: the browser version, checked
  frame for frame against PyTorch (`web/test_dreamer.mjs`, `web/check_browser.mjs`).
- `docs/`: the design, the measurements and the decisions behind them.

Training needs **World NES** (the emulator, a separate package, not included) and **your own Tetris
ROM**: set `TETRIS_ROM` to your copy of `Tetris (USA).nes`. No ROM is included in this repository.

```
pip install -e .[onnx]
python -m pytest tests            # tests that need World NES or the ROM skip without them
python scripts/train_layered.py --cooldown 46000 12000
python scripts/export_onnx.py output/world_model_tetris_layered_px/model_latest.pt
```
"""


def resolve(name: str) -> Path | None:
    if name.startswith("token_world"):
        base = ROOT / "src" / Path(*name.split("."))
        for p in (base.with_suffix(".py"), base / "__init__.py"):
            if p.is_file():
                return p
        return None
    for folder in ("scripts", "tests"):
        if (ROOT / folder / f"{name}.py").is_file():
            return ROOT / folder / f"{name}.py"
    return None


def closure() -> list[Path]:
    """ENTRIES and every project file they import (and their packages' __init__.py), but SKIP."""
    seen, todo = set(), [ROOT / e for e in ENTRIES]
    while todo:
        path = todo.pop()
        if path in seen or any(s in path.stem for s in SKIP):
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                     if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
            for p in filter(None, map(resolve, names)):
                todo.append(p)
                if p.is_relative_to(ROOT / "src"):
                    parts = p.relative_to(ROOT / "src").parts
                    todo += [ROOT / "src" / Path(*parts[:i]) / "__init__.py" for i in range(1, len(parts))
                             if (ROOT / "src" / Path(*parts[:i]) / "__init__.py").is_file()]
    return sorted(seen)


def main():
    out, web, model = ROOT / "site", ROOT / "web", ROOT / "web" / "model"
    missing = [name for name in MODEL if not (model / name).is_file()]
    if missing:
        raise SystemExit(f"{model} lacks {', '.join(missing)}: run scripts/export_onnx.py first")
    out.mkdir(exist_ok=True)
    for old in out.iterdir():                            # a fresh build in the repository's checkout
        if old.name != ".git":
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    files = closure() + [ROOT / e for e in EXTRA]
    for path in files:
        target = out / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (out / "web" / "model").mkdir(parents=True)
    for name in PAGE:
        shutil.copy2(web / name, out / "web" / name)
    for name in MODEL:
        shutil.copy2(model / name, out / "web" / "model" / name)
    (out / "docs").mkdir()
    for guide in ("dynamics.md", "layered_tokens.md", "recipe_attempts.md"):
        shutil.copy2(ROOT / "docs/guides" / guide, out / "docs" / guide)
    (out / ".github" / "workflows").mkdir(parents=True)
    (out / ".github" / "workflows" / "pages.yml").write_text(WORKFLOW, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (out / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    meta = json.loads((model / "context.json").read_text())
    params = f"{json.loads((model / 'model.json').read_text())['parameters'] / 1e6:.1f}M"
    (out / "README.md").write_text(README.format(step=meta["step"], params=params), encoding="utf-8")
    shipped = [f for f in out.rglob("*") if f.is_file() and ".git" not in f.relative_to(out).parts[:1]]
    roms = [f for f in shipped if f.suffix.lower() in ROMS or f.read_bytes()[:4] == b"NES\x1a"]
    if roms:
        raise SystemExit(f"ROM-like files in site/, not shipping: {roms}")
    print(f"site/: {len(files)} code files, the page and the step-{meta['step']:,} model, "
          f"{sum(f.stat().st_size for f in shipped) / 1e6:.1f} MB; no ROM")


if __name__ == "__main__":
    main()
