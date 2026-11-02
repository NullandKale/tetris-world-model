"""The RAM-driven Tetris bot: drops, line clears, the placement search and the button driver."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.data import tetris_bot as T


def ram_with(board: np.ndarray, orientation: int, x: int, y: int, next_id: int = 0x0A) -> np.ndarray:
    ram = np.zeros(2048, np.uint8)
    ram[T.BOARD:T.BOARD + 200] = np.where(board, 0x7B, T.EMPTY).ravel()
    ram[T.PIECE], ram[T.X], ram[T.Y], ram[T.NEXT] = orientation, x, y, next_id
    return ram


SPAWNS = (0x02, 0x07, 0x08, 0x0A, 0x0B, 0x0E, 0x12)


def play_piece(brain, board: np.ndarray, rng, level: int, orientation: int, frames: int = 2000) -> dict:
    """One piece in a small model of NES Tetris: turns and shifts on press edges when the piece fits, shifts
    ignored under Down, soft drop (a row every 2 frames) armed by a Down press made without Left or Right,
    gravity every FALL_FRAMES[level], the piece locking when a drop fails. The brain sees the RAM before each
    frame. -> the piece's last (orientation, x, y), its target, and per shift press whether it moved and how
    many rows the piece then had left to fall."""
    x, y, previous, armed, timer = 5, 0, 0, False, 0
    fall = T.FALL_FRAMES[level]
    shifts, actions = [], []
    target = start = None
    for frame in range(frames):
        ram = ram_with(board, orientation, x, y)
        ram[T.LEVEL] = level
        action = brain(ram, rng)
        actions.append(action)
        if target is None:
            target, start = brain.target, brain.start
        pressed = action & ~previous
        ids = T.IDS_OF[T.TYPE_OF[orientation]]
        for bit, step in ((T.A, 1), (T.B, -1)):
            turned = ids[(ids.index(orientation) + step) % len(ids)]
            if pressed & bit and T.fits(board, turned, x, y):
                orientation = turned
        for bit, dx in ((T.LEFT, -1), (T.RIGHT, 1)):
            if pressed & bit and not action & T.DOWN:
                moved = T.fits(board, orientation, x + dx, y)
                x += dx * moved
                shifts.append((moved, T.rest_row(board, orientation, x, y) - y))
        armed = bool(action & T.DOWN) and (armed or (pressed & T.DOWN and not action & (T.LEFT | T.RIGHT)))
        timer += 1
        if timer >= (2 if armed else fall):
            timer = 0
            if not T.fits(board, orientation, x, y + 1):
                return {"final": (orientation, x, y), "target": target, "start": start, "shifts": shifts,
                        "actions": actions}
            y += 1
        previous = action
    raise AssertionError("the piece never locked")


class TetrisBotTests(unittest.TestCase):
    def test_moves_cover_every_on_board_column(self):
        dy, columns, moves = T.MOVES["I"]
        self.assertEqual(len(moves), 7 + 10)                     # horizontal x 7 columns, vertical x 10
        self.assertTrue((columns >= 0).all() and (columns <= 9).all())

    def test_drop_rests_on_column_tops_and_clears_lines(self):
        board = np.zeros((20, 10), bool)
        board[19, :9] = True                                     # one gap, far right
        board[18, 0] = True
        after, lines, ok = T.place_all(board[None], "I")
        moves = T.MOVES["I"][2]
        k = moves.index((0x11, 9))                               # vertical I into the gap
        self.assertTrue(ok[0, k])
        self.assertEqual(lines[0, k], 1)
        expected = np.zeros((20, 10), bool)
        expected[17:, 9] = True                                  # the I's top 3 cells, shifted down one row
        expected[19, 0] = True                                   # the cell that sat on the cleared row
        self.assertTrue(np.array_equal(after[0, k], expected))

    def test_placement_above_the_top_is_refused(self):
        board = np.ones((20, 10), bool)
        board[:, 4] = False
        board[0, :] = True
        _, _, ok = T.place_all(board[None], "O")
        self.assertFalse(ok.any())
        self.assertIsNone(T.best_placement(board, "O", None))

    def test_search_fills_the_well(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        self.assertEqual(T.best_placement(board, "I", "O"), (0x11, 9, None))

    def test_perfect_brain_rotates_and_shifts_on_press_edges_then_drops(self):
        brain, rng = T.PerfectBrain(), np.random.default_rng(0)
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        self.assertEqual(brain(ram_with(board, 0x12, 5, 0), rng), 0)          # plans; Down released
        self.assertEqual(brain.target, (0x11, 9))
        self.assertIn(brain(ram_with(board, 0x12, 5, 0), rng), (T.A, T.B))   # two orientations: either button
        self.assertEqual(brain(ram_with(board, 0x12, 5, 0), rng), 0)          # released before the next tap
        self.assertEqual(brain(ram_with(board, 0x11, 5, 0), rng), T.RIGHT)
        self.assertEqual(brain(ram_with(board, 0x11, 6, 0), rng), 0)
        self.assertEqual(brain(ram_with(board, 0x11, 6, 1), rng), T.RIGHT)
        for x in (7, 8):
            brain(ram_with(board, 0x11, x, 1), rng)
            brain(ram_with(board, 0x11, x, 1), rng)
        self.assertEqual(brain(ram_with(board, 0x11, 9, 1), rng), 0)          # every press is released
        self.assertEqual(brain(ram_with(board, 0x11, 9, 1), rng), T.DOWN)
        self.assertEqual(brain(ram_with(board, 0x11, 9, 2), rng), T.DOWN)     # Down is held, not tapped

    def test_perfect_brain_holds_its_presses_and_still_gets_there(self):
        """hold: each turn and shift press is held 1 to h frames and released 1 to g (per piece, by level),
        some pieces with Down through the turns (never with a shift); the same moves to the same place."""
        board = np.zeros((20, 10), bool)
        for level, longest in ((0, T.HOLD_MAX), (19, 1)):
            brain, rng, held, gaps, down_turns = T.PerfectBrain(hold=True), np.random.default_rng(level), [], [], 0
            for n in range(60):
                brain.reset()
                piece = play_piece(brain, board, rng, level, 0x02)
                self.assertEqual(piece["final"][:2], piece["target"])
                previous, run, gap = 0, 0, 0
                for action in piece["actions"]:
                    move, before = action & ~T.DOWN, previous & ~T.DOWN
                    if move and move == before:
                        run += 1
                    else:
                        if before:
                            held.append(run)
                            gap = 0
                        if not move:
                            gap += 1
                        elif not before and gap:
                            gaps.append(gap)
                        run = 1 if move else 0
                    self.assertFalse(action & T.DOWN and action & (T.LEFT | T.RIGHT))
                    down_turns += bool(action & T.DOWN and action & (T.A | T.B))
                    previous = action
            self.assertEqual(max(held), longest)
            if longest > 1:
                self.assertTrue(set(range(1, 7)) <= set(held) and set(range(1, 7)) <= set(gaps))
                self.assertGreater(down_turns, 0)
            else:                                                                  # taps, released a frame
                self.assertGreater(gaps.count(1), len(gaps) / 2)                   # (longer: a late shift's wait)

    def test_search_finds_a_tuck_under_an_overhang(self):
        board = np.zeros((20, 10), bool)
        board[18:, 3:] = True                                                      # two rows, columns 3-9
        board[17, 0] = True                                                        # over an empty column 0
        self.assertEqual(T.best_placement(board, "O", None, tuck=True), (0x0A, 1, 2))   # down 1-2, under to 0-1
        self.assertIsNone(T.best_placement(board, "O", None)[2])                   # straight drops only
        brain, rng = T.PerfectBrain(hold=True), np.random.default_rng(0)
        piece = play_piece(brain, board, rng, 0, 0x0A)
        self.assertEqual(piece["start"], 2)
        self.assertEqual(piece["final"], (0x0A, 1, 18))                            # it got under
        brain.reset()
        self.assertIsNone(play_piece(brain, board, rng, 19, 0x0A)["start"])         # too fast to tuck

    def test_perfect_brain_presses_into_walls_and_the_stack_and_shifts_late(self):
        """NUDGE pieces press into a wall or the stack (nothing moves), LATE ones shift a few rows above their
        landing, and every piece still lands where it was planned."""
        brain, rng = T.PerfectBrain(hold=True), np.random.default_rng(7)
        blocked = late = early = 0
        for n in range(300):
            board = np.zeros((20, 10), bool)
            for column, height in enumerate(rng.integers(0, 6, 10)):
                board[20 - height:, column] = True
            board[19, rng.integers(10)] = False                                    # no full rows
            brain.reset()
            piece = play_piece(brain, board, rng, int(rng.choice((0, 5, 12))), int(rng.choice(SPAWNS)))
            orientation, x, y = piece["final"]
            self.assertEqual((orientation, x), piece["target"])
            if piece["start"] is None:
                self.assertEqual(y, T.top_row(board, orientation, x))              # straight down, as planned
            moved = [rows for went, rows in piece["shifts"] if went]
            blocked += sum(not went for went, _ in piece["shifts"])
            late += bool(moved) and moved[-1] <= max(T.LATE_ROWS)
            early += bool(moved) and moved[-1] > 8
        self.assertGreater(blocked, 30)
        self.assertGreater(late, 30)
        self.assertGreater(early, late)                                            # most still shift high

    def test_turns_go_the_short_way(self):
        rng = np.random.default_rng(0)
        for piece in "JLT":
            ids = T.IDS_OF[piece]
            for i, o in enumerate(ids):
                self.assertEqual(T.turn_button(o, ids[(i + 1) % 4], rng), T.A)
                self.assertEqual(T.turn_button(o, ids[(i - 1) % 4], rng), T.B)
                halves = {T.turn_button(o, ids[(i + 2) % 4], rng) for _ in range(40)}
                self.assertEqual(halves, {T.A, T.B})
        self.assertEqual({T.turn_button(0x11, 0x12, rng) for _ in range(40)}, {T.A, T.B})

    def test_switcher_plays_one_brain_until_its_interval_ends(self):
        switcher = T.BrainSwitcher(np.random.default_rng(3), long=False)
        switcher.game_left = 10 ** 6
        first, left = switcher.current, switcher.left
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        for _ in range(left - 1):
            switcher(ram, np.random.default_rng(0))
            self.assertEqual(switcher.current, first)

    def test_switcher_gives_up_when_the_game_length_runs_out(self):
        rng = np.random.default_rng(5)
        switcher = T.BrainSwitcher(rng, long=False)
        switcher.game_left = 3
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        seen = set()
        for _ in range(20_000):
            switcher(ram, rng)
            if switcher.game_left <= 0:
                seen.add(switcher.current)
        self.assertEqual(seen, {"reckless", "random"})

    def test_long_games_are_perfect_play_short_games_mixed(self):
        rng = np.random.default_rng(0)
        long, short = T.BrainSwitcher(rng, long=True), T.BrainSwitcher(rng, long=False)
        self.assertGreaterEqual(long.game_left, T.BrainSwitcher.LONG[0])
        self.assertLessEqual(short.game_left, T.BrainSwitcher.SHORT[1])
        ram = ram_with(np.zeros((20, 10), bool), 0x02, 5, 0)
        seen = set()
        for _ in range(10_000):
            long(ram, rng)
            seen.add(long.current)
        self.assertEqual(seen, {"perfect"})

    def test_brain_taking_over_replans_the_falling_piece(self):
        """The stale-target bug: a perfect brain that played a T and takes over a J mid-fall
        must replan instead of tapping rotate toward a T orientation."""
        rng = np.random.default_rng(0)
        brain = T.PerfectBrain()
        brain(ram_with(np.zeros((20, 10), bool), 0x02, 5, 0), rng)
        for y in range(1, 13):
            brain(ram_with(np.zeros((20, 10), bool), brain.target[0], brain.target[1], y), rng)
        brain.reset()                                                            # BrainSwitcher does this
        brain(ram_with(np.zeros((20, 10), bool), 0x07, 5, 14), rng)
        self.assertEqual(T.TYPE_OF[brain.target[0]], "J")

    def test_patient_pieces_wait_then_soft_drop(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True
        brain, rng = T.PerfectBrain(patience=1.0), np.random.default_rng(0)
        brain(ram_with(board, 0x11, 9, 0), rng)                                    # plans: vertical I, column 9
        wait = brain.wait
        self.assertTrue(T.PATIENT_FRAMES[0] <= wait < T.PATIENT_FRAMES[1])
        pressed = [brain(ram_with(board, 0x11, 9, 1), rng) for _ in range(wait + 3)]
        self.assertEqual(pressed[:wait], [0] * wait)                               # gravity alone
        self.assertEqual(pressed[wait:], [T.DOWN] * 3)                             # then soft drop
        brain, rng = T.PerfectBrain(patience=0.3), np.random.default_rng(1)
        patient = []
        for piece in range(2000):
            brain.reset()
            brain(ram_with(board, 0x11, 9, 0), rng)
            patient.append(brain.wait > 0)
        self.assertAlmostEqual(np.mean(patient), 0.3, delta=0.03)

    def test_patient_pieces_spin_over_a_low_stack_then_turn_back_and_drop(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True                                                      # 4 rows: low
        brain, rng = T.PerfectBrain(patience=1.0, spin=1.0), np.random.default_rng(0)
        brain(ram_with(board, 0x11, 9, 0), rng)
        self.assertTrue(brain.spinning)
        orientation, pressed = 0x11, []
        for _ in range(T.PATIENT_FRAMES[1] + 20):
            action = brain(ram_with(board, orientation, 9, 1), rng)
            pressed.append(action)
            if action & (T.A | T.B):                                               # I piece: 0x11 <-> 0x12
                orientation = 0x12 if orientation == 0x11 else 0x11
        spins = sum(1 for a in pressed if a & (T.A | T.B))
        self.assertGreater(spins, 2)
        self.assertNotIn(T.DOWN, pressed[:T.PATIENT_FRAMES[0]])                    # no drop while it waits
        self.assertEqual(orientation, 0x11)                                        # turned back to the plan
        self.assertEqual(pressed[-1], T.DOWN)
        board[4:, :9] = True                                                       # 16 rows: too high to play
        brain.reset()
        brain(ram_with(board, 0x11, 9, 0), rng)
        self.assertFalse(brain.spinning)

    def test_holding_brain_holds_shifts_and_turns_and_never_drops(self):
        board = np.zeros((20, 10), bool)
        brain, rng = T.HoldingBrain(), np.random.default_rng(4)
        brain(ram_with(board, 0x02, 5, 0), rng)                                    # T: plans a place
        goal, column = brain.target
        orientation, x, pressed = 0x02, 5, []
        cycle = [o for o in T.ORIENTATIONS if T.TYPE_OF[o] == "T"]
        for frame in range(200):
            action = brain(ram_with(board, orientation, x, 1 + frame // 48), rng)
            pressed.append(action)
            if action & (T.A | T.B) and not (pressed[-2] if len(pressed) > 1 else 0) & (T.A | T.B):
                orientation = cycle[(cycle.index(orientation) + 1) % len(cycle)]  # a turn on the press
            if action & T.RIGHT and frame % 6 == 0:
                x += 1                                                             # the game's autorepeat
            if action & T.LEFT and frame % 6 == 0:
                x -= 1
        self.assertNotIn(T.DOWN, [a & T.DOWN for a in pressed if a & T.DOWN])
        self.assertEqual((orientation, x), (goal, column))                         # it gets there
        shifts = [i for i, a in enumerate(pressed) if a & (T.LEFT | T.RIGHT)]
        if len(shifts) > 1:                                                        # held, never tapped
            self.assertEqual(shifts, list(range(shifts[0], shifts[-1] + 1)))
        self.assertEqual(pressed[-1], 0)                                           # lined up: gravity alone
        self.assertIn("holding", T.BrainSwitcher.WEIGHTS)

    def test_person_brain_varies_its_presses_and_gets_there(self):
        """Per piece: hold lengths and gaps vary, Down is used four ways, turns and shifts sometimes overlap;
        every piece still reaches its planned place."""
        board = np.zeros((20, 10), bool)
        brain, rng = T.PersonBrain(), np.random.default_rng(3)
        cycle = [o for o in T.ORIENTATIONS if T.TYPE_OF[o] == "T"]
        held, downs, combos, styles = set(), set(), 0, set()
        for piece in range(200):
            brain.reset()
            orientation, x, previous = 0x02, 5, 0
            brain(ram_with(board, orientation, x, 0), rng)                         # plans
            styles.add(brain.down)
            goal, column = brain.target
            run = 0
            for frame in range(120):
                action = brain(ram_with(board, orientation, x, 1 + frame // 48), rng)
                pressed = action & ~previous                                       # press edges act
                if pressed & T.A:
                    orientation = cycle[(cycle.index(orientation) + 1) % len(cycle)]
                if pressed & T.B:
                    orientation = cycle[(cycle.index(orientation) - 1) % len(cycle)]
                x += bool(pressed & T.RIGHT) - bool(pressed & T.LEFT)
                if action & (T.A | T.B | T.LEFT | T.RIGHT):
                    run += 1
                elif run:
                    held.add(run)
                    run = 0
                combos += bin(action & (T.A | T.B | T.LEFT | T.RIGHT | T.DOWN)).count("1") >= 2
                if action & T.DOWN:
                    downs.add(brain.down)
                previous = action
            self.assertEqual((orientation, x), (goal, column))
        self.assertEqual(styles, set(T.PERSON_DOWN))
        self.assertEqual(downs, {"drop", "held", "tapped"})                            # gravity never presses
        self.assertTrue({1, 2, 3, 4, 5, 6} <= held)
        self.assertLess(max(held), 16)                                                 # under autorepeat
        self.assertGreater(combos, 0)
        self.assertIn("person", T.BrainSwitcher.WEIGHTS)

    def test_person_lets_go_of_a_down_that_does_nothing(self):
        """The game ignores shifts under Down, and a Down pressed with a shift never drops: let go, press again."""
        board = np.zeros((20, 10), bool)
        brain, rng = T.PersonBrain(), np.random.default_rng(0)
        brain(ram_with(board, 0x02, 5, 0), rng)
        brain.target, brain.down, brain.react, brain.together = (0x02, 8), "held", 0, False
        actions = [brain(ram_with(board, 0x02, 5, 1), rng) for _ in range(12)]   # the piece never moves
        first = next(i for i, a in enumerate(actions) if a & T.RIGHT)
        self.assertTrue(actions[first] & T.DOWN)                                   # Down with the shift
        self.assertFalse(any(a & T.DOWN for a in actions[first + 2:]))             # blocked: Down let go
        brain.target = (0x02, 5)                                                   # lined up: Down again,
        held = [brain(ram_with(board, 0x02, 5, 1), rng) & T.DOWN for _ in range(3 * T.PERSON_STALL)]
        self.assertTrue(all(held[1:T.PERSON_STALL - 2]))                           # held while it may drop,
        self.assertIn(0, held[T.PERSON_STALL:])                                    # re-pressed when it sticks

    def test_random_brain_presses_pairs(self):
        brain, rng = T.RandomBrain(), np.random.default_rng(0)
        actions = [brain(None, rng) for _ in range(20_000)]
        pairs = {a for a in actions if bin(a).count("1") == 2}
        self.assertGreater(len(pairs), 8)
        self.assertNotIn(3, [bin(a).count("1") for a in actions])

    def test_reckless_brain_picks_legal_random_placements(self):
        rng = np.random.default_rng(1)
        board = np.zeros((20, 10), bool)
        board[10:, :] = True
        board[10:, 4] = False
        brain = T.RecklessBrain()
        targets = set()
        for _ in range(50):
            brain.reset()
            brain(ram_with(board, 0x12, 5, 0), rng)
            targets.add(brain.target)
        ok = T.place_all(board[None], "I")[2][0]
        legal = {T.MOVES["I"][2][i] for i in np.flatnonzero(ok)}
        self.assertTrue(targets <= legal)
        self.assertGreater(len(targets), 5)

    def test_well_builder_keeps_the_well_and_takes_the_tetris(self):
        board = np.zeros((20, 10), bool)
        board[16:, :9] = True                                                    # four rows ready, well open
        self.assertEqual(T.best_placement(board, "I", None, well=True), (0x11, 9, None))   # vertical I: a tetris
        board = np.zeros((20, 10), bool)
        board[18:, :7] = True                                                    # covering the well clears nothing
        self.assertNotEqual(T.best_placement(board, "O", None, well=True)[1], 9)  # O at x 9 fills columns 8-9
        self.assertEqual(T.best_placement(board, "O", None, well=False)[1], 8)   # the classic brain fills 7-8


class StartLevelTests(unittest.TestCase):
    def test_long_games_start_low_short_games_anywhere(self):
        from token_world.data.world_nes_tetris import TetrisSession
        session = TetrisSession(0)
        drawn = [(session._start_level(), session.long_game) for _ in range(20_000)]
        long = [lv for lv, is_long in drawn if is_long]
        short = [lv for lv, is_long in drawn if not is_long]
        self.assertAlmostEqual(len(long) / len(drawn), T.BrainSwitcher.LONG_SHARE, delta=0.01)
        self.assertEqual(set(long), set(range(10)))
        self.assertEqual(set(short), set(range(20)))

if __name__ == "__main__":
    unittest.main()
