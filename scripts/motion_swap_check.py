"""Does a motion model's DRAWN MOTION follow A vs B (2026-10-16)? The swap check's instances
(scripts/button_swap_check.py: tapped J/L/T turns and shifts where the other way fits), one frame ahead from 63 real
frames with their real motion, the press real or swapped: per kind, the share of the real move's moved cells the
drawn motion gets exactly right, and the share it draws as the opposite move's motion, real press against swapped.

    python scripts/motion_swap_check.py CHECKPOINT [CHECKPOINT ...] [--turns 150] [--shifts 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from button_swap_check import ROTATE, SHIFT, T, instance
import token_world.data.tetris_motion as M
from token_world.data.world_nes_tetris import WorldNesTetrisStreams
from token_world.models.dynamics import build, incoming_actions


def opposite_codes(ram, kind):
    """The motion the other way would have drawn: the RAM after with the other orientation / column."""
    e = T - 2
    before, after = ram[e - 1], ram[e].copy()
    from token_world.data.tetris_bot import IDS_OF, ORIENTATIONS, PIECE, TYPE_OF, X
    o0, o1 = int(before[PIECE]), int(after[PIECE])
    if kind == "turn":
        ids = IDS_OF[TYPE_OF[o1]]
        after[PIECE] = ids[(2 * ids.index(o0) - ids.index(o1)) % 4]
    else:
        after[X] = 2 * int(before[X]) - int(after[X])
    return M.piece_moves(before, after)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--turns", type=int, default=150)
    p.add_argument("--shifts", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda", help="cuda, or cpu while a training run holds the GPU")
    p.add_argument("--motion-steps", type=int, choices=(1, 2), default=2,
                   help="a motion-first model's motion decoding: 2 steps (confident cells first) or 1")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for the model (0: torch's default)")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    want = {"turn": args.turns, "shift": args.shifts}
    found = []
    loader = DataLoader(WorldNesTetrisStreams(T, 9191, stride=1), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4)
    for part in loader:
        ram, act = part["ram"].numpy(), part["action"].numpy()
        hit = instance(ram, act, want)
        if hit is None:
            continue
        x = part["x"].numpy()
        motion = M.stream_motion(x, ram)[0]
        found.append((hit, x, act, ram, motion))
        want[hit[0]] -= 1
        if not any(want.values()):
            break
    print(f"{sum(f[0][0] == 'turn' for f in found)} turns, {sum(f[0][0] == 'shift' for f in found)} shifts", flush=True)
    for ckpt in args.ckpts:
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model = build(s["args"]).to(device).eval()
        model.load_state_dict(s["ema"])
        model.motion_steps = args.motion_steps
        tally = {}
        for (kind, press, *_), x, act, ram, motion in found:
            group = ROTATE if kind == "turn" else SHIFT
            swapped = (press & ~group) | (group & ~press)
            real = motion[T - 1]                                       # the frame that shows the move
            moved = (real != M.STILL) & (real != M.NEW) & (real != M.UNKNOWN)
            other = np.full_like(real, M.STILL)
            for (r1, c1), (r0, c0) in opposite_codes(ram, kind) or []:
                if (r1, c1) != (r0, c0) and 0 <= r1 < 20 and 0 <= r0 < 20:
                    other[M.WELL_ROW + r1, M.WELL_COL + c1] = M.code_of(r1 - r0, c1 - c0)
            other_moved = other != M.STILL
            for name, byte in (("real press", press), ("swapped", swapped)):
                a = act.copy()
                a[T - 3] = byte
                if act[T - 2] & group:
                    a[T - 2] = (act[T - 2] & ~group) | (byte & group)
                inc = incoming_actions(torch.from_numpy(a[None])).to(device)
                xs = torch.from_numpy(x[None]).to(device)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    decoder = model.decoder(1, xs.device)
                    decoder.prefill(xs[:, :T - 1], inc[:, :T - 1], torch.from_numpy(motion[None, :T - 1]).to(device))
                    drawn = decoder.next(inc[:, T - 1:T], T - 1)["motion"][0].cpu().numpy()
                t = tally.setdefault((kind, name), [0.0, 0.0, 0])
                t[0] += float((drawn == real)[moved].mean()) if moved.any() else 0.0
                t[1] += float((drawn == other)[other_moved].mean()) if other_moved.any() else 0.0
                t[2] += 1
        print(f"\n{Path(ckpt).parent.name} @ {s['step']:,}: drawn motion, share of the real move's cells right / of "
              f"the opposite move's cells drawn")
        for kind in ("turn", "shift"):
            if (kind, "real press") not in tally:              # none asked for (--shifts 0)
                continue
            r, w = tally[(kind, "real press")], tally[(kind, "swapped")]
            print(f"  {kind:>5}: real press {r[0] / r[2]:.2f} / {r[1] / r[2]:.2f}   swapped {w[0] / w[2]:.2f} / "
                  f"{w[1] / w[2]:.2f}   gap (real-move cells right, real press - swapped) {(r[0] - w[0]) / r[2]:+.2f}",
                  flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
