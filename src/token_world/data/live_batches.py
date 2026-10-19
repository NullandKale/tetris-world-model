"""Batches from live history workers: one chunk per worker, checked for repeats and gaps."""
from __future__ import annotations

import re
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader


def data_field(name: str) -> str:
    """A window label (e.g. "line clear: tetris") -> its metrics column, data_line_clear_tetris."""
    return "data_" + re.sub("[^a-z0-9]+", "_", name.lower()).strip("_")


def data_fields(names) -> list[str]:
    """metrics.csv's columns for the training data: windows, the share kept, the share holding each name, and
    how new its spawn boards are (diagnostics/novelty.py)."""
    from token_world.diagnostics.novelty import NOVELTY_FIELDS
    return ["data_windows", "data_kept", *(data_field(n) for n in names), *NOVELTY_FIELDS]


class LiveBatches:
    """Assemble one chunk per history worker into a [workers, T, ...] batch.

    Checks that no worker repeats within a batch and that every worker's tick
    advances by whole chunks (streams may toss windows, never parts of one), so
    a duplicated, restarted or misaligned history fails loudly. Marked streams
    (world_nes_tetris.WindowToss marked) send every window: `kept` [workers]
    says which of the last batch's the toss kept, and the data counts are of
    those alone. Windows with a counterfactual twin (world_nes_tetris.TurnBranches) add it as a row after
    the workers' own, in worker order; `branch_at` [rows] is each row's branch press position, -1 for the
    workers' own windows.
    """

    def __init__(self, loader: DataLoader, workers: int, new_frames: int, novelty=None):
        """novelty: a diagnostics.novelty.Novelty that sees every window trained on (take_stats reports it)."""
        self.iterator = iter(loader)
        self.novelty = novelty
        self.workers, self.new_frames = workers, new_frames
        self.last_tick: dict[int, int] = {}
        self.wait_sum_ms = self.wait_max_ms = 0.0
        self.wait_count = 0
        self.stream_stats: dict[str, int] = {}
        self.names: Counter = Counter()           # labelled windows (world_nes_tetris.WindowToss): what is in them
        self.windows = self.judged = 0
        self.kept = torch.ones(workers, dtype=torch.bool)
        self.motion = None                         # the last batch's motion codes, when the stream makes them
        self.branch_at = torch.full((workers,), -1, dtype=torch.long)
        self.piece = None                          # [rows, T] each frame's piece byte, when the stream has it
        self.events = None                         # [rows, T] each picture's events (tetris_events.frame_events)
        self.branch_kinds: Counter = Counter()    # the twins trained on since take_stats, by kind

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        start = time.monotonic()
        chunks, action_chunks, meta, kept, motion, twins, piece, events = {}, {}, {}, {}, {}, {}, {}, {}
        while len(chunks) < self.workers:
            part = next(self.iterator)
            worker = int(part["worker_id"])
            if worker in chunks:
                raise RuntimeError(f"duplicate history worker {worker} in mixed batch")
            tick = int(part["tick"])
            if worker in self.last_tick and (tick <= self.last_tick[worker]
                                             or (tick - self.last_tick[worker]) % self.new_frames):
                raise RuntimeError(f"worker {worker} frame stream jumped: tick {tick} after "
                                   f"{self.last_tick[worker]}, not whole {self.new_frames}-frame chunks later")
            self.last_tick[worker] = tick
            picture = (part["layers"] if "layers" in part else {"x": part["x"], "sprites": part["sprites"]}
                       if "sprites" in part else part["x"])
            chunks[worker], action_chunks[worker] = picture, part["action"]
            if "motion" in part:
                motion[worker] = part["motion"]
            if "piece" in part:
                piece[worker] = part["piece"]
            if "events" in part:
                events[worker] = part["events"]
            if "branch" in part:
                twins[worker] = part["branch"]
                self.branch_kinds[part["branch"]["kind"]] += 1
            meta[worker] = (tick, int(part["in_game_restarts"]))
            kept[worker] = bool(part.get("keep", True))
            if self.novelty is not None and kept[worker]:
                self.novelty.add(part)
            if "toss_names" in part:                # what this window holds, and the windows judged for it
                self.judged += int(part["toss_judged"])
                if kept[worker]:
                    self.names.update(part["toss_names"])
                    self.windows += 1
        wait_ms = (time.monotonic() - start) * 1000
        self.wait_sum_ms += wait_ms
        self.wait_max_ms = max(self.wait_max_ms, wait_ms)
        self.wait_count += 1
        self.stream_stats = {"stream_tick_min": min(t for t, _ in meta.values()),
                             "stream_tick_max": max(t for t, _ in meta.values()),
                             "in_game_restarts": sum(n for _, n in meta.values())}
        # Each chunk is already pinned (DataLoader pin_memory), so per-chunk copies are truly
        # asynchronous and the batch is stacked on the GPU. Stacking on the CPU first made an
        # unpinned tensor whose copy blocked the main thread until the GPU queue drained.
        order = range(self.workers)
        self.kept = torch.tensor([kept[i] for i in order] + [True] * len(twins))
        branched = sorted(twins)
        self.branch_at = torch.tensor([-1] * self.workers + [int(twins[i]["at"]) for i in branched])
        if isinstance(chunks[0], dict):            # layered windows (data/nes_layers.py), or x with sprites
            x = {k: torch.stack([chunks[i][k].cuda(non_blocking=True) for i in order]) for k in chunks[0]}
        else:
            x = torch.stack([chunks[i].cuda(non_blocking=True) for i in order]
                            + [twins[i]["x"].cuda(non_blocking=True) for i in branched])
        actions = torch.stack([action_chunks[i].cuda(non_blocking=True) for i in order]
                              + [twins[i]["action"].cuda(non_blocking=True) for i in branched])
        self.motion = (torch.stack([motion[i].cuda(non_blocking=True) for i in order]
                                   + [twins[i]["motion"].cuda(non_blocking=True) for i in branched])
                       if motion else None)
        self.piece = (torch.stack([piece[i].cuda(non_blocking=True) for i in order]
                                  + [twins[i]["piece"].cuda(non_blocking=True) for i in branched])
                      if piece else None)
        self.events = (torch.stack([events[i].cuda(non_blocking=True) for i in order]
                                   + [twins[i]["events"].cuda(non_blocking=True) for i in branched])
                       if events else None)
        return x, actions

    def take_stats(self) -> dict[str, float]:
        """Stream counters plus DataLoader wait since the previous call."""
        stats = {**self.stream_stats,
                 "data_wait_ms_mean": self.wait_sum_ms / max(self.wait_count, 1),
                 "data_wait_ms_max": self.wait_max_ms}
        if self.novelty is not None:
            stats.update(self.novelty.take())
        if self.branch_kinds:                       # the twins: per batch, and the share of each kind
            twins = sum(self.branch_kinds.values())
            stats.update({"data_branches": twins / max(self.wait_count, 1),
                          **{f"data_branch_{k}": c / twins for k, c in self.branch_kinds.items()}})
        if self.windows:                            # the windows trained on since the last call
            stats.update({"data_windows": self.windows, "data_kept": self.windows / max(self.judged, 1),
                          **{data_field(n): c / self.windows for n, c in self.names.items()}})
        self.wait_sum_ms = self.wait_max_ms = 0.0
        self.wait_count = self.windows = self.judged = 0
        self.names = Counter()
        self.branch_kinds = Counter()
        return stats
