// The page: keyboard -> NES controller byte -> Dreamer (dreamer.js) -> canvas, at most 60 frames a second.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import { Dreamer } from "./dreamer.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// ?profile: onnxruntime-web logs each WebGPU kernel's time and where every node runs (check_browser.mjs reads it)
const PROFILE = new URLSearchParams(location.search).has("profile");
if (PROFILE) ort.env.webgpu.profiling = { mode: "default" };

const A = 1, B = 2, SELECT = 4, START = 8, UP = 16, DOWN = 32, LEFT = 64, RIGHT = 128;
const KEYS = { ArrowLeft: LEFT, ArrowRight: RIGHT, ArrowDown: DOWN, ArrowUp: UP, x: A, X: A, z: B, Z: B,
               Enter: START, Shift: SELECT };
const NAMES = [["A", A], ["B", B], ["Select", SELECT], ["Start", START], ["Up", UP], ["Down", DOWN],
               ["Left", LEFT], ["Right", RIGHT]];
const FRAME_MS = 1000 / 60;

const $ = (id) => document.getElementById(id);
const screen = $("screen"), ctx = screen.getContext("2d"), image = ctx.createImageData(256, 256);
let held = 0, paused = false, pending = Promise.resolve(), dreamer, context, palette, backend;

const load = async (name) => {
  const response = await fetch(`model/${name}`);
  if (!response.ok) throw new Error(`model/${name}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
};

function draw(frame) {
  const px = image.data;
  for (let i = 0; i < frame.length; i++) {
    const c = frame[i] * 3, o = i * 4;
    px[o] = palette[c]; px[o + 1] = palette[c + 1]; px[o + 2] = palette[c + 2]; px[o + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function status(text, warn = false) {
  $("status").textContent = text;
  $("status").className = warn ? "warn" : "";
}

async function start() {
  try {
    backend = "gpu" in navigator && (await navigator.gpu.requestAdapter()) ? "webgpu" : "wasm";
    status(`Loading the model (${backend})…`, backend !== "webgpu");
    const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
    palette = Uint8Array.from(meta.palette.flat());
    const pixels = await load("context.bin");
    context = { frames: Array.from({ length: meta.actions.length }, (_, i) => pixels.subarray(i * 65536, (i + 1) * 65536)),
                actions: meta.actions };
    for (let level = 0; level < meta.levels; level++) $("level").add(new Option(level === 0 ? "0 (as real)" : `${level}`, level));
    $("level").value = meta.level;
    draw(context.frames.at(-1));
    dreamer = await Dreamer.create(ort, load, context.frames, context.actions,
                                   { level: meta.level, keep: meta.keep, executionProviders: [backend],
                                     sessionOptions: PROFILE ? { logSeverityLevel: 0, logVerbosityLevel: 0 } : {} });
    screen.focus();
    if (new URLSearchParams(location.search).has("check")) await check();
    run();
  } catch (error) {
    status(`Could not start: ${error.message}`, true);
    console.error(error.stack ?? error);
  }
}

// ?check: dream the start with no buttons and compare with PyTorch's frames (model/reference.bin, written
// by scripts/export_onnx.py), then play. The result is in the status and in document.body.dataset.check.
async function check() {
  const reference = await load("reference.bin"), count = reference.length / 65536;
  let first = -1, wrong = 0, worst = 0, ms = 0;
  const plain = [], slides = [];
  for (let i = 0; i < count; i++) {
    const slide = dreamer.frames.length === dreamer.meta.frames;     // this step re-encodes the window first
    const began = performance.now();
    const frame = await dreamer.next(0);
    const took = performance.now() - began;
    ms += took;
    (slide ? slides : plain).push(took);
    draw(frame);
    let differ = 0;
    for (let j = 0; j < 65536; j++) differ += frame[j] !== reference[i * 65536 + j];
    if (differ && first < 0) first = i;
    wrong += differ;
    worst = Math.max(worst, differ);
  }
  const result = `${count} frames, level ${dreamer.level}: ` +
    (first < 0 ? "identical to PyTorch" : `${worst <= 4 ? "near-ties only" : "DIFFERENT"}: first differs at +${first + 1}, ` +
                                          `${wrong} pixels in all, at most ${worst} in a frame`) +
    `; ${(ms / count).toFixed(1)} ms per frame on ${backend} (median ${median(plain).toFixed(1)} ms a step, ` +
    `${median(slides).toFixed(1)} ms with a re-encode, ${slides.length} of them)`;
  document.body.dataset.check = result;
  console.log(result);
  await dreamer.reset(context.frames, context.actions);
}

const median = (xs) => xs.length ? [...xs].sort((a, b) => a - b)[xs.length >> 1] : NaN;

async function run() {
  let average = 0, frames = 0, last = performance.now();
  for (;;) {
    if (paused) { await sleep(50); last = performance.now(); continue; }
    const began = performance.now();
    pending = dreamer.next(held);              // restart waits for it: never two runs on one cache
    draw(await pending);
    const took = performance.now() - began;
    average = frames++ ? 0.9 * average + 0.1 * took : took;
    await sleep(Math.max(0, FRAME_MS - (performance.now() - last)));
    const now = performance.now(), fps = 1000 / (now - last);
    last = now;
    if (frames % 10 === 0) {
      status(`${backend === "webgpu" ? "WebGPU" : "WebAssembly (no WebGPU: slow)"}\n` +
             `${average.toFixed(1)} ms per frame, ${Math.min(fps, 60).toFixed(0)} frames/s\n` +
             `frame ${frames}, window position ${dreamer.frames.length}`, backend !== "webgpu");
    }
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function key(event, down) {
  const bit = KEYS[event.key];
  if (bit === undefined) return;
  event.preventDefault();
  held = down ? held | bit : held & ~bit;
  $("held").textContent = NAMES.filter(([, b]) => held & b).map(([n]) => n).join(" ");
}

addEventListener("keydown", (e) => key(e, true));
addEventListener("keyup", (e) => key(e, false));
addEventListener("blur", () => { held = 0; $("held").textContent = ""; });
$("level").addEventListener("change", () => { dreamer.level = Number($("level").value); screen.focus(); });
$("restart").addEventListener("click", async () => {
  paused = true;
  await pending;
  await dreamer.reset(context.frames, context.actions);
  draw(context.frames.at(-1));
  paused = false;
  $("pause").textContent = "Pause";
  screen.focus();
});
$("pause").addEventListener("click", () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
  screen.focus();
});

start();
