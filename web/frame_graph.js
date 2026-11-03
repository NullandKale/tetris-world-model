// The page's frame-time graph: the last COUNT frames, each a bar as tall as the time since the frame before it,
// split into the step's run (with the wait for the GPU's step before: dreamer.timing), the cache write, the draw
// and the rest (the wait for the next 60th of a second, and whatever else the browser did meanwhile); lines at
// 60 and 30 frames/s, and the median, 90% and worst frame time under it. A bar taller than the graph is cut, a
// red tip on top. On a slow device (top over 100 ms) the lines are at round times instead.
const COUNT = 240;
const PARTS = [["step", "#7cc4ff"], ["cache write", "#c792ea"], ["draw", "#7fd88f"], ["rest", "#4a4a5c"]];
const LINES = [[1000 / 60, "60/s"], [1000 / 30, "30/s"]];

export class FrameGraph {
  constructor(canvas, legend) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.frames = [];                                    // [total, step, write, draw] ms
    legend.innerHTML = PARTS.map(([name, colour]) => `<span><i style="background:${colour}"></i>${name}</span>`)
      .join("") + `<span class="stats"></span>`;
    this.stats = legend.querySelector(".stats");
  }

  add(total, step, write, draw) {
    this.frames.push([total, step, write, draw]);
    if (this.frames.length > COUNT) this.frames.shift();
    this.draw();
  }

  draw() {
    const { canvas, ctx } = this, scale = devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * scale), h = Math.round(canvas.clientHeight * scale);
    if (canvas.width !== w || canvas.height !== h) Object.assign(canvas, { width: w, height: h });
    ctx.clearRect(0, 0, w, h);
    const totals = this.frames.map((f) => f[0]).sort((a, b) => a - b);
    const at = (q) => totals[Math.min(totals.length - 1, Math.floor(q * totals.length))];
    // the scale: at least 30 frames/s's 33 ms, else a little over 98% of the frames shown
    const top = Math.max(40, Math.ceil(1.15 * (at(0.98) ?? 0) / 10) * 10), y = (ms) => h - (ms / top) * h;
    const bar = w / COUNT, x0 = w - this.frames.length * bar;
    this.frames.forEach(([total, ...parts], i) => {
      const x = x0 + i * bar;
      let below = 0;
      for (const [k, ms] of [...parts, Math.max(0, total - parts.reduce((a, b) => a + b, 0))].entries()) {
        ctx.fillStyle = PARTS[k][1];
        ctx.fillRect(x, y(below + ms), Math.max(1, bar - scale * 0.5), y(below) - y(below + ms));
        below += ms;
      }
      if (total > top) { ctx.fillStyle = "#ff5c5c"; ctx.fillRect(x, 0, Math.max(1, bar), 3 * scale); }
    });
    ctx.font = `${10 * scale}px system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    // a fast device's lines at 60 and 30 frames/s; a slow one's (a phone: 200 ms) at round times
    const every = [50, 100, 200, 500, 1000].find((s) => top / s <= 4) ?? 2000;
    const lines = top <= 100 ? LINES : Array.from({ length: Math.floor((top - 1) / every) },
                                                   (_, i) => [(i + 1) * every, `${(i + 1) * every} ms`]);
    for (const [ms, label] of [...lines.filter(([ms]) => ms < 0.85 * top), [top, `${top} ms`]]) {   // none on the top's label
      ctx.fillStyle = "rgba(228, 228, 236, 0.35)";
      ctx.fillRect(0, Math.round(y(ms)), w, scale);
      ctx.fillStyle = "#9a9aae";
      ctx.fillText(label, 3 * scale, Math.max(11 * scale, y(ms) - scale));
    }
    if (totals.length)
      this.stats.textContent = `frame: median ${at(0.5).toFixed(1)} ms, 90% ${at(0.9).toFixed(1)}, ` +
                               `worst ${totals.at(-1).toFixed(1)}`;
  }
}
