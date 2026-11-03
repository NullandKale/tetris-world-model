// The buttons at 60 frames a second, whatever the model's pace. Every press and release is kept with the time the
// browser saw it (event.timeStamp, not when the handler ran: a busy page still gets the true times), so the NES
// frames a slow step skips have their buttons too. The game's clock: frame k is the 60th of a second from
// origin + k * FRAME_MS. A frame's byte is the buttons held at its start and any pressed during it (a tap shorter
// than a frame gets that frame).
//
// Timeline draws the last SHOWN frames' buttons, a row each, every step's frames a band and a tick on the frame
// it showed; under it, over the last STATS steps: the skips, and of the skipped steps with buttons in them, how
// many one byte for all the step's frames would give exactly (rule A), and how many the first frame's byte and
// one for the rest would (rule B).
export const FRAME_MS = 1000 / 60;
const KEEP_MS = 10_000, SHOWN = 360, STATS = 300;
const ROWS = [["↑", 16], ["↓", 32], ["←", 64], ["→", 128], ["A", 1], ["B", 2], ["Se", 4], ["St", 8]];

export class Presses {
  constructor() {
    this.events = [{ time: -Infinity, held: 0 }];       // each change of the buttons held, oldest first
    this.origin = performance.now();
  }

  set(held, time) {
    const last = this.events.at(-1);
    if (held === last.held) return;
    this.events.push({ time: Math.max(time, last.time), held });
    let old = 0;
    while (old + 1 < this.events.length && this.events[old + 1].time <= time - KEEP_MS) old++;
    this.events.splice(0, old);
  }

  frame(time) { return Math.floor((time - this.origin) / FRAME_MS); }

  start(k) { return this.origin + k * FRAME_MS; }

  // The clock moves on so that frame k starts at `time` (after a pause; or a step slower than the most a step
  // may skip: the game slows down).
  restart(k, time) { this.origin = time - k * FRAME_MS; }

  // Frames from..to-1 -> their bytes, a Uint8Array.
  bytes(from, to) {
    const out = new Uint8Array(to - from), events = this.events;
    let e = 0;
    for (let k = from; k < to; k++) {
      const begins = this.start(k), ends = begins + FRAME_MS;
      while (e + 1 < events.length && events[e + 1].time <= begins) e++;
      let byte = events[e].held;
      for (let j = e + 1; j < events.length && events[j].time < ends; j++) byte |= events[j].held;
      out[k - from] = byte;
    }
    return out;
  }
}

export class Timeline {
  constructor(canvas, stats) {
    Object.assign(this, { canvas, stats, ctx: canvas.getContext("2d"), steps: [], count: 0 });
  }

  // A step: its frames' bytes, the first frame's number `from`.
  add(from, bytes) {
    this.steps.push({ from, bytes, n: this.count++ });
    if (this.steps.length > Math.max(STATS, SHOWN)) this.steps.shift();
    this.draw();
  }

  draw() {
    const { canvas, ctx, steps } = this, scale = devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * scale), h = Math.round(canvas.clientHeight * scale);
    if (canvas.width !== w || canvas.height !== h) Object.assign(canvas, { width: w, height: h });
    ctx.clearRect(0, 0, w, h);
    if (!steps.length) return;
    const newest = steps.at(-1), end = newest.from + newest.bytes.length, first = end - SHOWN;
    const left = 22 * scale, fw = (w - left) / SHOWN, rh = h / ROWS.length;
    for (const { from, bytes, n } of steps) {
      if (from + bytes.length <= first) continue;
      const x = left + (from - first) * fw;
      ctx.fillStyle = n % 2 ? "#262633" : "#1c1c26";                 // the step's frames
      ctx.fillRect(x, 0, bytes.length * fw, h);
      ctx.fillStyle = "#7cc4ff";
      bytes.forEach((byte, j) => ROWS.forEach(([, bit], r) => {
        if (byte & bit) ctx.fillRect(x + j * fw, r * rh + scale, Math.max(fw, scale), rh - 2 * scale);
      }));
      ctx.fillStyle = "rgba(228, 228, 236, 0.45)";                     // the frame the step showed
      ctx.fillRect(x + bytes.length * fw - scale, 0, scale, h);
    }
    ctx.fillStyle = "#15151c";
    ctx.fillRect(0, 0, left, h);
    ctx.fillStyle = "#9a9aae";
    ctx.font = `${Math.min(10 * scale, rh * 0.9)}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    ROWS.forEach(([name], r) => ctx.fillText(name, 2 * scale, (r + 0.5) * rh));

    const recent = steps.slice(-STATS), skips = recent.map((s) => s.bytes.length).sort((a, b) => a - b);
    const pct = (k, of) => `${Math.round((100 * k) / of)}%`;
    const pressed = recent.filter((s) => s.bytes.length > 1 && s.bytes.some((b) => b));
    const one = pressed.filter((s) => s.bytes.every((b) => b === s.bytes[0])).length;
    const two = pressed.filter((s) => s.bytes.every((b, j) => j === 0 || b === s.bytes.at(-1))).length;
    this.stats.textContent = `skip: median ${skips[skips.length >> 1]}, most ${skips.at(-1)}; ` +
      `${pct(skips.filter((s) => s > 1).length, skips.length)} of steps skip` +
      (pressed.length ? `. Of ${pressed.length} skips with buttons: one byte fits ${pct(one, pressed.length)}, ` +
                        `first + rest ${pct(two, pressed.length)}` : "");
  }
}
