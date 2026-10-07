// What the page tells the console (F12), so a slow or broken browser can be debugged from its output alone:
// every line starts with "[world-model]". At start: the browser, onnxruntime-web's version, WebGPU's adapter
// (vendor, architecture, whether it is a software fallback), its features and limits, or why there is no
// WebGPU; each model file's size and load time and each graph's setup time; a lost GPU device or a GPU error
// as it happens. While playing, every 5 s: the frame rate and the times of a step (the graph's run, the cache
// write, the draw) and of the window re-encodes. With ?profile: after PROFILE_FRAMES frames, onnxruntime's
// profile of each graph summed by kind of operation and by node: GPU time per run where the adapter has
// timestamp queries, and every node onnxruntime placed on the CPU (each one a round trip from the GPU).
export const log = (...args) => console.log("[world-model]", ...args);
export const warn = (...args) => console.warn("[world-model]", ...args);

const MB = (bytes) => `${(bytes / 2 ** 20).toFixed(1)} MB`;

// The backend to run on: WebGPU when the browser gives an adapter, else WebAssembly (slow); ?backend=wasm or
// ?backend=webgpu chooses. Logs what the browser offers. -> { backend, gpu: a short name for the status line }
export async function findBackend(ort, wanted) {
  log(`browser: ${navigator.userAgent}`);
  log(`onnxruntime-web ${ort.env.versions?.web ?? "?"}; secure context ${isSecureContext}, ` +
      `cross-origin isolated ${globalThis.crossOriginIsolated === true} (WebAssembly threads need it)`);
  const wasm = (why) => { (wanted === "wasm" ? log : warn)(`backend: WebAssembly, on the CPU: ${why}`); return { backend: "wasm", gpu: "" }; };
  if (wanted === "wasm") return wasm("asked for with ?backend=wasm");
  if (!("gpu" in navigator))
    return wasm("this browser has no navigator.gpu (WebGPU switched off or not supported; Firefox: about:config " +
                "dom.webgpu.enabled; WebGPU also needs https or localhost)");
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (error) {
    warn("navigator.gpu.requestAdapter() threw:", error);
  }
  if (!adapter) return wasm("WebGPU is there but gave no adapter (the GPU or its driver may be blocklisted; Chrome: " +
                            "chrome://gpu, Firefox: about:support)");
  const info = adapter.info ?? {};
  const gpu = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ");
  log(`WebGPU adapter: vendor "${info.vendor ?? ""}", architecture "${info.architecture ?? ""}", device ` +
      `"${info.device ?? ""}", description "${info.description ?? ""}"` +
      (info.isFallbackAdapter ? " (SOFTWARE FALLBACK: no real GPU, slow)" : ""));
  log(`WebGPU features: ${[...adapter.features].sort().join(", ") || "none"}`);
  const limits = adapter.limits;
  log(`WebGPU limits: buffer ${MB(limits.maxBufferSize)}, storage binding ${MB(limits.maxStorageBufferBindingSize)}, ` +
      `${limits.maxStorageBuffersPerShaderStage} storage buffers a stage, workgroup storage ` +
      `${limits.maxComputeWorkgroupStorageSize} B, ${limits.maxComputeInvocationsPerWorkgroup} invocations a workgroup`);
  if (/intel|amd|ati/i.test(info.vendor ?? ""))
    log("(on a laptop with two GPUs this may be the integrated one: Windows ignores powerPreference; choose the " +
        "browser's GPU in Windows Settings > System > Display > Graphics)");
  return { backend: "webgpu", gpu };
}

// After onnxruntime made its device: report it being lost and any GPU error nothing else caught.
export async function watchDevice(ort) {
  const device = await ort.env.webgpu?.device;
  if (!device) return;
  device.lost.then((info) => warn(`GPU device lost (${info.reason}): ${info.message}`));
  device.addEventListener("uncapturederror", (event) => warn("GPU error:", event.error?.message ?? event.error));
}

// fetch, logging each file's size and time.
export function logged(load) {
  return async (name) => {
    const began = performance.now();
    const bytes = await load(name);
    log(`loaded ${name}: ${MB(bytes.length)} in ${(performance.now() - began).toFixed(0)} ms`);
    return bytes;
  };
}

// The play loop's times, logged every 5 s.
export class Timings {
  constructor() { this.reset(performance.now()); }

  reset(now) {
    Object.assign(this, { since: now, frames: 0, run: [], write: [], encode: [], draw: [] });
  }

  add({ run, write, encode }, draw) {
    this.frames++;
    this.run.push(run);
    this.write.push(write);
    this.draw.push(draw);
    if (encode) this.encode.push(encode);
    const now = performance.now();
    if (now - this.since < 5000) return;
    const s = (now - this.since) / 1000, f = (xs) => xs.length ? `${quantile(xs, 0.5).toFixed(1)} ms (90%: ` +
      `${quantile(xs, 0.9).toFixed(1)})` : "none";
    log(`${(this.frames / s).toFixed(1)} frames/s over ${s.toFixed(1)} s: a step's run ${f(this.run)}, cache write ` +
        `${f(this.write)}, draw ${f(this.draw)}; ${this.encode.length} re-encodes, ${f(this.encode)}`);
    this.reset(now);
  }
}

const quantile = (xs, q) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))];

// ?profile: onnxruntime prints a session's profile to the console when it ends (Chrome trace events, one a
// line), summed here: profile() ends each graph's profile and logs the summary (returned as text too).
let caught = [];

// Its WebAssembly binds console.log when it loads: this must run before the first session is made. From then
// on the profile's lines are kept out of the console (tens of thousands: slow to print) for profile().
export function catchProfiles() {
  for (const kind of ["log", "info", "debug"]) {
    const original = console[kind].bind(console);
    console[kind] = (...args) => {
      const text = args.length === 1 && typeof args[0] === "string" ? args[0] : "";
      if (/^\s*(\[|\]|\{"cat")/.test(text)) caught.push(text);
      else original(...args);
    };
  }
}

export function profile(sessions, runs) {
  const lines = [];
  for (const [name, session] of Object.entries(sessions)) {
    caught = [];
    session.endProfiling();
    lines.push(...summary(name, events(caught), runs[name] ?? 0));
  }
  caught = [];
  for (const line of lines) log(line);
  return lines.join("\n");
}

function events(texts) {
  const found = [];
  for (const text of texts)
    for (const line of text.split("\n")) {
      const json = line.trim().replace(/,$/, "");
      if (!json.startsWith("{")) continue;
      try { found.push(JSON.parse(json)); } catch { /* not an event */ }
    }
  return found;
}

function summary(name, events, runs) {
  const lines = [`profile of ${name}.onnx, ${runs} runs:`];
  const gpu = events.filter((e) => e.cat === "Api"), nodes = events.filter((e) => e.cat === "Node");
  const per = Math.max(runs, 1);
  const cpu = new Map();
  for (const e of nodes) if (e.args?.provider === "CPUExecutionProvider") cpu.set(e.name, e.args?.op_name ?? "?");
  lines.push(`  ${cpu.size} nodes on the CPU` + (cpu.size ? `: ${[...new Set(cpu.values())].join(", ")}` : ""));
  if (!gpu.length) {
    lines.push("  no GPU timings (WebGPU without timestamp-query, or WebAssembly): CPU-side node times instead");
    table(lines, nodes.map((e) => ({ op: e.args?.op_name ?? e.name, node: e.name, shapes: "", dur: e.dur })), per);
    return lines;
  }
  table(lines, gpu.map((e) => {
    const parts = e.name.split("&");
    return { op: parts.at(-1), node: parts[0], shapes: e.args?.shapes ?? "", dur: e.dur };
  }), per);
  return lines;
}

function table(lines, rows, runs) {
  const total = rows.reduce((a, r) => a + r.dur, 0);
  lines.push(`  ${(total / 1000 / runs).toFixed(2)} ms a run in ${Math.round(rows.length / runs)} kernels`);
  const by = (key) => {
    const sums = new Map();
    for (const r of rows) {
      const k = key(r), s = sums.get(k) ?? { dur: 0, count: 0 };
      s.dur += r.dur;
      s.count++;
      sums.set(k, s);
    }
    return [...sums].sort((a, b) => b[1].dur - a[1].dur);
  };
  const line = (label, s) => `    ${(s.dur / 1000 / runs).toFixed(3).padStart(8)} ms ${(100 * s.dur / total).toFixed(1).padStart(5)}%` +
    `  x${String(Math.round(s.count / runs)).padEnd(4)} ${label}`;
  lines.push("  by kind of operation:");
  for (const [op, s] of by((r) => r.op).slice(0, 15)) lines.push(line(op, s));
  lines.push("  slowest nodes:");
  for (const [k, s] of by((r) => `${r.op} ${r.node} ${r.shapes}`).slice(0, 20)) lines.push(line(k.slice(0, 160), s));
}
