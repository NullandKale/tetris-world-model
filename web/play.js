// The page: keyboard -> NES controller byte -> Dreamer (dreamer.js) -> canvas, at most 60 frames a second.
// The console (F12) says what runs where and how fast (diagnostics.js). Options in the address:
//   ?backend=wasm|webgpu  run there (default: WebGPU when the browser has it)
//   ?profile              after PROFILE_FRAMES frames, log each graph's GPU time by operation and node
//   ?verbose              onnxruntime's own log, everything (node placements, ...)
//   ?check                first dream the start with no buttons against PyTorch's frames (model/reference.bin)
//   ?slow=MS              each step takes at least MS ms (a slow GPU, to see frame skip on a fast one)
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { Dreamer, GpuScreen, TRACE, startFrames } from "./dreamer.js";
import { setupEditor } from "./shot_editor.js";
import { FrameGraph } from "./frame_graph.js";
import { Presses, Timeline } from "./presses.js";
import { Timings, catchProfiles, findBackend, log, logged, profile, warn, watchDevice } from "./diagnostics.js";

const OPTIONS = new URLSearchParams(location.search);
const PROFILE = OPTIONS.has("profile"), VERBOSE = OPTIONS.has("verbose"), PROFILE_FRAMES = 120;
const SLOW = Number(OPTIONS.get("slow")) || 0;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = VERBOSE ? "verbose" : "warning";
ort.env.webgpu.powerPreference = "high-performance";
if (PROFILE) catchProfiles();                     // before onnxruntime's WebAssembly loads

const A = 1, B = 2, SELECT = 4, START = 8, UP = 16, DOWN = 32, LEFT = 64, RIGHT = 128;
const KEYS = { ArrowLeft: LEFT, ArrowRight: RIGHT, ArrowDown: DOWN, ArrowUp: UP, x: A, X: A, z: B, Z: B,
               Enter: START, Shift: SELECT };
const NAMES = [["A", A], ["B", B], ["Select", SELECT], ["Start", START], ["Up", UP], ["Down", DOWN],
               ["Left", LEFT], ["Right", RIGHT]];
const MAX_SKIP = 6;                                 // frames a step may cover at most; slower, the game slows
                                                    // (a frame-skip model's: at most its skip)

const $ = (id) => document.getElementById(id);
const screen = $("screen");
let held = 0, paused = false, pending = Promise.resolve(), dreamer, context, backend, gpu = "", painter;
let keyboard = 0;                                   // buttons held on the keyboard
let tapped = 0;                                     // buttons pressed since the last frame read them (2026-11-01:
                                                    // a tap shorter than a frame, 250 ms at 4 frames/s, never
                                                    // reached the model; now it gets one frame)
const touches = new Map();                          // on-screen button held by each pointer (finger, mouse)
const presses = new Presses();                      // the buttons at 60 frames a second (presses.js)
let traced = [];                                    // every game frame's byte the steps since the last reset got,
                                                    // oldest first: the trace's own record (Save trace)
let skipChoice = 0;                                 // the frames each step covers (the Frame skip menu): 0, the
                                                    // frames passed since the last step (at most the model's skip)
const SKIP_KEY = "frame-skip";                      // the choice, kept in this browser

const load = logged(async (name) => {
  const response = await fetch(`model/${name}`);
  if (!response.ok) throw new Error(`model/${name}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
});

// A dreamed frame (dreamer.next's colours, an ort.Tensor [256, 256, 3]) on the screen: on WebGPU drawn from the
// GPU (GpuScreen; nothing read back), elsewhere through a 2D canvas. The caller disposes the tensor.
function draw(rgb) {
  if (dreamer.device) {
    painter ??= new GpuScreen(dreamer.device, screen);
    painter.draw(rgb);
    return;
  }
  painter ??= { ctx: screen.getContext("2d") };
  const image = painter.ctx.createImageData(256, 256), px = image.data, data = rgb.data;
  for (let i = 0, o = 0; i < data.length; i += 3, o += 4) {
    px[o] = data[i]; px[o + 1] = data[i + 1]; px[o + 2] = data[i + 2]; px[o + 3] = 255;
  }
  painter.ctx.putImageData(image, 0, 0);
}

function status(text, warn = false) {
  $("status").textContent = text;
  $("status").className = warn ? "warn" : "";
}

async function start() {
  try {
    // which GPU: on a laptop the browser may take the integrated one (Windows ignores powerPreference)
    ({ backend, gpu } = await findBackend(ort, OPTIONS.get("backend")));
    status(`Loading the model (${backend})…`, backend !== "webgpu");
    const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
    context = { start: startFrames(meta, await load("context.bin")), actions: meta.actions };
    const original = context;                      // the page's own start; an image start replaces `context`
    setupEditor({ palette: meta.palette, base: original.start,
                  play: (start, actions) => restartWith({ start, actions }), normal: () => restartWith(original) });
    dreamer = await Dreamer.create(ort, load, context.start, context.actions, {
      executionProviders: [backend], log,
      sessionOptions: { ...(PROFILE ? { enableProfiling: true } : {}),
                        ...(VERBOSE ? { logSeverityLevel: 0, logVerbosityLevel: 0 } : {}) } });
    if (backend === "webgpu") await watchDevice(ort);
    setupSkip();
    log(`ready: ${dreamer.meta.parameters.toLocaleString()} parameters, ${dreamer.n} tokens a frame, ` +
        `${dreamer.meta.frames}-frame window`);
    screen.focus();
    if (new URLSearchParams(location.search).has("check")) await check();
    run();
  } catch (error) {
    status(`Could not start: ${error.message} (the console, F12, has more)`, true);
    warn("could not start:", error.stack ?? error);
  }
}

// ?check: dream the start with no buttons and compare with PyTorch's frames (model/reference.bin, written
// by scripts/export_onnx.py), then play. The result is in the status and in document.body.dataset.check.
async function check() {
  const reference = await load("reference.bin"), count = reference.length / (65536 * 3);
  let first = -1, wrong = 0, worst = 0, ms = 0;
  const steps = [];
  for (let i = 0; i < count; i++) {
    const began = performance.now();
    const frame = await dreamer.next(0);
    draw(frame);
    const colours = await frame.getData();                // read back: the check's only
    frame.dispose();
    const took = performance.now() - began;
    ms += took;
    steps.push(took);
    let differ = 0;
    for (let j = 0; j < 65536 * 3; j++) differ += Math.abs(colours[j] - reference[i * 65536 * 3 + j]) > TOLERANCE;
    if (differ && first < 0) first = i;
    wrong += differ;
    worst = Math.max(worst, differ);
  }
  const result = `${count} frames: ` +
    (first < 0 ? `as PyTorch (colours within ${TOLERANCE})` : `DIFFERENT: first differs at +${first + 1}, ` +
                                          `${wrong} colour values in all, at most ${worst} in a frame`) +
    `; ${(ms / count).toFixed(1)} ms per frame on ${backend} (median ${median(steps).toFixed(1)} ms a step)`;
  document.body.dataset.check = result;
  log(result);
  await dreamer.reset(context.start, context.actions);
  traced = [];
}

const TOLERANCE = 2;                             // colour values: PyTorch's are rounded to bytes
const median = (xs) => xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : NaN;

// The Frame skip menu: Auto (0) or 1..the model's skip (at most MAX_SKIP); a model without frame skip has only 1.
function setupSkip() {
  const select = $("skip"), most = dreamer.skip > 1 ? Math.min(MAX_SKIP, dreamer.skip) : 1;
  const counts = Array.from({ length: most }, (_, i) => new Option(`${i + 1} frame${i ? "s" : ""}`, `${i + 1}`));
  select.replaceChildren(...(most > 1 ? [new Option(`Auto (1–${most})`, "0")] : []), ...counts);
  let saved = 0;
  try { saved = Number(localStorage.getItem(SKIP_KEY)) || 0; } catch { /* no storage: Auto */ }
  skipChoice = most > 1 && saved <= most ? saved : 0;
  select.value = String(skipChoice || (most > 1 ? 0 : 1));
  select.disabled = most === 1;
  select.addEventListener("change", () => {
    skipChoice = Number(select.value);
    try { localStorage.setItem(SKIP_KEY, select.value); } catch { /* kept for this visit only */ }
    screen.focus();
  });
}

// The play loop on the game's clock (presses.js). Auto: a step starts at the earliest at the next frame's 60th of a
// second and covers the frames since the last step's, at most the model's skip (MAX_SKIP at most); a chosen skip N:
// a step starts at the earliest when N frames have passed and covers exactly N. A step slower than that most
// slows the game down (the Timeline shows each step's frames and their buttons). A frame-skip model's step is all
// of them: the newest frame's byte and the others', newest first (dreamer.next), as training gives a step
// (data/world_nes_tetris.py step_bytes). Any other model takes one frame a step whatever the skip, from the
// buttons held now and those tapped since the last step.
async function run() {
  let average = 0, frames = 0, last = performance.now();
  const skips = dreamer.skip > 1, cap = skips ? Math.min(MAX_SKIP, dreamer.skip) : MAX_SKIP;
  const each = () => skips ? skipChoice : 0;          // the chosen skip; 0: Auto
  let shown = presses.frame(last);                   // the game frame on the screen
  const timings = new Timings(), graph = new FrameGraph($("frame-graph"), $("frame-legend"));
  const timeline = new Timeline($("press-timeline"), $("press-stats"));
  for (;;) {
    if (paused) {
      await sleep(50);
      last = performance.now();
      timings.reset(last);
      presses.restart(shown + 1, last);              // the clock stops while paused
      continue;
    }
    const began = performance.now(), most = each() || cap;
    let reached = Math.max(shown + (each() || 1), presses.frame(began));   // the game frame this step shows
    let dropped = 0;
    if (reached - shown > most) {                    // too slow even skipping: the game slows down, and the
      for (const byte of presses.bytes(shown + 1, reached - most + 1)) dropped |= byte;   // frames it drops
      presses.restart(shown + most, began);          // give their buttons to the step's first, as a tap
      reached = shown + most;
    }
    const bytes = presses.bytes(shown + 1, reached + 1);
    bytes[0] |= dropped;
    timeline.add(shown + 1, bytes);
    shown = reached;
    const fed = skips ? Array.from(bytes) : [held | tapped];
    if (dreamer.trace.steps.length < TRACE) traced.push(...fed);
    pending = dreamer.next(fed.at(-1), fed.slice(0, -1).reverse());   // restart waits for it: never two runs on
    tapped = 0;                                                        // one cache
    const frame = await pending, drawing = performance.now();
    draw(frame);
    frame.dispose();
    const drawn = performance.now() - drawing;
    timings.add(dreamer.timing, drawn);
    if (PROFILE && frames + 1 === PROFILE_FRAMES) document.body.dataset.profile = profile(dreamer.sessions, dreamer.runs);
    const took = performance.now() - began;
    average = frames++ ? 0.9 * average + 0.1 * took : took;
    if (SLOW) await sleep(SLOW - took);
    await sleep(presses.start(shown + (each() || 1)) - performance.now());
    const now = performance.now(), fps = 1000 / (now - last);
    graph.add(now - last, dreamer.timing.run, dreamer.timing.write, drawn);
    last = now;
    if (frames % 10 === 0) {
      status(`${backend === "webgpu" ? `WebGPU${gpu ? ` (${gpu})` : ""}` : "WebAssembly (no WebGPU: slow)"}\n` +
             `${average.toFixed(1)} ms per frame, ${Math.min(fps, 60).toFixed(0)} frames/s\n` +
             `frame ${frames}`, backend !== "webgpu");
    }
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The controller byte the next frame gets: the keyboard's buttons and the on-screen ones, together.
function update(time = performance.now()) {
  held = keyboard;
  for (const bit of touches.values()) held |= bit;
  presses.set(held, time);
  $("held").textContent = NAMES.filter(([, b]) => held & b).map(([n]) => n).join(" ");
  for (const button of $("pad").querySelectorAll("button"))
    button.classList.toggle("down", (held & Number(button.dataset.bit)) !== 0);
}

function key(event, down) {
  const bit = KEYS[event.key];
  if (bit === undefined) return;
  event.preventDefault();
  keyboard = down ? keyboard | bit : keyboard & ~bit;
  if (down) tapped |= bit;
  update(event.timeStamp);
}

for (const button of $("pad").querySelectorAll("button")) {
  const bit = Number(button.dataset.bit);
  const release = (e) => { touches.delete(e.pointerId); update(e.timeStamp); };
  button.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    touches.set(e.pointerId, bit);
    tapped |= bit;
    update(e.timeStamp);
  });
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("pointerleave", release);
}
$("pad").addEventListener("contextmenu", (e) => e.preventDefault());   // no long-press menu on phones

addEventListener("keydown", (e) => key(e, true));
addEventListener("keyup", (e) => key(e, false));
addEventListener("blur", () => { keyboard = 0; tapped = 0; touches.clear(); update(); });
// Start the dream again from `start` (the page's, or an image's: shot_editor.js); Restart replays the current one.
async function restartWith(start) {
  if (!dreamer) return;
  paused = true;
  await pending;
  context = start;
  await dreamer.reset(context.start, context.actions);
  traced = [];
  paused = false;
  $("pause").textContent = "Pause";
  screen.focus();
}
$("restart").addEventListener("click", () => restartWith(context));
// The steps since the last reset as a file: what each fed the model (dreamer.trace) and the page's own record of
// every frame's byte; scripts/input_trace_check.py checks they are in training's order.
$("save-trace").addEventListener("click", () => {
  if (!dreamer) return;
  const trace = { ...dreamer.trace, frames: traced, agent: navigator.userAgent };
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(trace)], { type: "application/json" }));
  link.download = `trace_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 10_000);
  screen.focus();
});
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  screen.focus();
});

start();
