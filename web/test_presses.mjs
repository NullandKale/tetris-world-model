// presses.js's buttons at 60 frames a second: a frame's byte is what was held at its start and anything pressed
// during it, wherever the presses fall against the game's clock.
//
//     cd web && node test_presses.mjs
import { FRAME_MS, Presses } from "./presses.js";

const LEFT = 64, A = 1, DOWN = 32;
let failed = 0;
const check = (name, got, want) => {
  const ok = JSON.stringify([...got]) === JSON.stringify(want);
  failed += !ok;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${JSON.stringify([...got])}${ok ? "" : ` (want ${JSON.stringify(want)})`}`);
};

const presses = new Presses();
presses.restart(0, 1000);                              // frame k starts at 1000 + k * FRAME_MS
const at = (k, part = 0) => 1000 + (k + part) * FRAME_MS;
presses.set(LEFT, at(1, 0.2));                         // Left held from inside frame 1 to inside frame 3
presses.set(0, at(3, 0.5));
presses.set(A, at(5, 0.1));                            // a tap inside frame 5, shorter than a frame
presses.set(0, at(5, 0.6));
presses.set(DOWN, at(7));                              // Down from frame 7's very start, and A on top in frame 8
presses.set(DOWN | A, at(8, 0.5));
check("frames 0..9", presses.bytes(0, 10), [0, LEFT, LEFT, LEFT, 0, A, 0, DOWN, DOWN | A, DOWN | A]);
check("a step of frames 4..6", presses.bytes(4, 7), [0, A, 0]);
presses.set(DOWN | A, at(9));                          // no change: no event
check("no change kept", [presses.events.length], [7]);

presses.restart(20, at(30));                           // the clock slipped: frame 20 starts where 30 did
check("after a restart", presses.bytes(19, 21), [DOWN | A, DOWN | A]);
process.exit(failed ? 1 : 0);
