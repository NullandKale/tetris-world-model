// The page: keyboard -> NES controller byte -> Dreamer (dreamer.js) -> canvas, at most 60 frames a second.
// The console (F12) says what runs where and how fast (diagnostics.js). Options in the address:
//   ?backend=wasm|webgpu  run there (default: WebGPU when the browser has it)
//   ?profile              after PROFILE_FRAMES frames, log each graph's GPU time by operation and node
//   ?verbose              onnxruntime's own log, everything (node placements, ...)
//   ?check                first dream the start with no buttons against PyTorch's frames (model/reference.bin)
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { Dreamer, GpuScreen, startFrames } from "./dreamer.js";
import { setupEditor } from "./shot_editor.js";
import { Timings, catchProfiles, findBackend, log, logged, profile, warn, watchDevice } from "./diagnostics.js";

const OPTIONS = new URLSearchParams(location.search);
const PROFILE = OPTIONS.has("profile"), VERBOSE = OPTIONS.has("verbose"), PROFILE_FRAMES = 120;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = VERBOSE ? "verbose" : "warning";
ort.env.webgpu.powerPreference = "high-performance";
if (PROFILE) catchProfiles();                     // before onnxruntime's WebAssembly loads

const A = 1, B = 2, SELECT = 4, START = 8, UP = 16, DOWN = 32, LEFT = 64, RIGHT = 128;
const KEYS = { ArrowLeft: LEFT, ArrowRight: RIGHT, ArrowDown: DOWN, ArrowUp: UP, x: A, X: A, z: B, Z: B,
               Enter: START, Shift: SELECT };
const NAMES = [["A", A], ["B", B], ["Select", SELECT], ["Start", START], ["Up", UP], ["Down", DOWN],
               ["Left", LEFT], ["Right", RIGHT]];
const FRAME_MS = 1000 / 60;

const $ = (id) => document.getElementById(id);
const screen = $("screen");
let held = 0, paused = false, pending = Promise.resolve(), dreamer, context, backend, gpu = "", painter;
let keyboard = 0;                                   // buttons held on the keyboard
const touches = new Map();                          // on-screen button held by each pointer (finger, mouse)

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
  const plain = [], slides = [];
  for (let i = 0; i < count; i++) {
    const slide = dreamer.content.length === dreamer.meta.frames;    // this step re-encodes the window first
    const began = performance.now();
    const frame = await dreamer.next(0);
    draw(frame);
    const colours = await frame.getData();                // read back: the check's only
    frame.dispose();
    const took = performance.now() - began;
    ms += took;
    (slide ? slides : plain).push(took);
    let differ = 0;
    for (let j = 0; j < 65536 * 3; j++) differ += Math.abs(colours[j] - reference[i * 65536 * 3 + j]) > TOLERANCE;
    if (differ && first < 0) first = i;
    wrong += differ;
    worst = Math.max(worst, differ);
  }
  const result = `${count} frames: ` +
    (first < 0 ? `as PyTorch (colours within ${TOLERANCE})` : `DIFFERENT: first differs at +${first + 1}, ` +
                                          `${wrong} colour values in all, at most ${worst} in a frame`) +
    `; ${(ms / count).toFixed(1)} ms per frame on ${backend} (median ${median(plain).toFixed(1)} ms a step, ` +
    `${median(slides).toFixed(1)} ms with a re-encode, ${slides.length} of them)`;
  document.body.dataset.check = result;
  log(result);
  await dreamer.reset(context.start, context.actions);
}

const TOLERANCE = 2;                             // colour values: PyTorch's are rounded to bytes
const median = (xs) => xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : NaN;

async function run() {
  let average = 0, frames = 0, last = performance.now();
  const timings = new Timings();
  for (;;) {
    if (paused) { await sleep(50); last = performance.now(); timings.reset(last); continue; }
    const began = performance.now();
    pending = dreamer.next(held);              // restart waits for it: never two runs on one cache
    const frame = await pending, drawing = performance.now();
    draw(frame);
    frame.dispose();
    timings.add(dreamer.timing, performance.now() - drawing);
    if (PROFILE && frames + 1 === PROFILE_FRAMES) document.body.dataset.profile = profile(dreamer.sessions, dreamer.runs);
    const took = performance.now() - began;
    average = frames++ ? 0.9 * average + 0.1 * took : took;
    await sleep(Math.max(0, FRAME_MS - (performance.now() - last)));
    const now = performance.now(), fps = 1000 / (now - last);
    last = now;
    if (frames % 10 === 0) {
      status(`${backend === "webgpu" ? `WebGPU${gpu ? ` (${gpu})` : ""}` : "WebAssembly (no WebGPU: slow)"}\n` +
             `${average.toFixed(1)} ms per frame, ${Math.min(fps, 60).toFixed(0)} frames/s\n` +
             `frame ${frames}, window position ${dreamer.content.length}`, backend !== "webgpu");
    }
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The controller byte the next frame gets: the keyboard's buttons and the on-screen ones, together.
function update() {
  held = keyboard;
  for (const bit of touches.values()) held |= bit;
  $("held").textContent = NAMES.filter(([, b]) => held & b).map(([n]) => n).join(" ");
  for (const button of $("pad").querySelectorAll("button"))
    button.classList.toggle("down", (held & Number(button.dataset.bit)) !== 0);
}

function key(event, down) {
  const bit = KEYS[event.key];
  if (bit === undefined) return;
  event.preventDefault();
  keyboard = down ? keyboard | bit : keyboard & ~bit;
  update();
}

for (const button of $("pad").querySelectorAll("button")) {
  const bit = Number(button.dataset.bit);
  const release = (e) => { touches.delete(e.pointerId); update(); };
  button.addEventListener("pointerdown", (e) => { e.preventDefault(); touches.set(e.pointerId, bit); update(); });
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("pointerleave", release);
}
$("pad").addEventListener("contextmenu", (e) => e.preventDefault());   // no long-press menu on phones

addEventListener("keydown", (e) => key(e, true));
addEventListener("keyup", (e) => key(e, false));
addEventListener("blur", () => { keyboard = 0; touches.clear(); update(); });
// Start the dream again from `start` (the page's, or an image's: shot_editor.js); Restart replays the current one.
async function restartWith(start) {
  if (!dreamer) return;
  paused = true;
  await pending;
  context = start;
  await dreamer.reset(context.start, context.actions);
  paused = false;
  $("pause").textContent = "Pause";
  screen.focus();
}
$("restart").addEventListener("click", () => restartWith(context));
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  screen.focus();
});

start();
