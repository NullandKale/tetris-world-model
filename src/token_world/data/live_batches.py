"""Batches from live history workers: one chunk per worker, checked for repeats and gaps."""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader


class LiveBatches:
    """Assemble one chunk per history worker into a [workers, T, ...] batch.

    Checks that no worker repeats within a batch and that every worker's tick
    advances by whole chunks (streams may toss windows, never parts of one), so
    a duplicated, restarted or misaligned history fails loudly.
    """

    def __init__(self, loader: DataLoader, workers: int, new_frames: int):
        self.iterator = iter(loader)
        self.workers, self.new_frames = workers, new_frames
        self.last_tick: dict[int, int] = {}
        self.wait_sum_ms = self.wait_max_ms = 0.0
        self.wait_count = 0
        self.stream_stats: dict[str, int] = {}

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        start = time.monotonic()
        chunks, action_chunks, meta = {}, {}, {}
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
            chunks[worker], action_chunks[worker] = part["x"], part["action"]
            meta[worker] = (tick, int(part["in_game_restarts"]))
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
        x = torch.stack([chunks[i].cuda(non_blocking=True) for i in order])
        actions = torch.stack([action_chunks[i].cuda(non_blocking=True) for i in order])
        return x, actions

    def take_stats(self) -> dict[str, float]:
        """Stream counters plus DataLoader wait since the previous call."""
        stats = {**self.stream_stats,
                 "data_wait_ms_mean": self.wait_sum_ms / max(self.wait_count, 1),
                 "data_wait_ms_max": self.wait_max_ms}
        self.wait_sum_ms = self.wait_max_ms = 0.0
        self.wait_count = 0
        return stats
