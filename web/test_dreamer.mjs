// dreamer.js against PyTorch, on the CPU (onnxruntime-node): the page's start, dreamed with no buttons,
// must give the colours scripts/export_onnx.py recorded from PyTorch's Dreamer (model/reference.bin), within
// TOLERANCE per colour value. A broken cache or position changes thousands. A frame-skip model is then dreamed
// export_onnx.py's steps of 1..skip frames (skip_steps.json) against PyTorch's colours (skip_reference.bin); with
// TRACE=file its input trace is written there (scripts/input_trace_check.py checks it against training's order).
//
//     cd web && npm install && node test_dreamer.mjs      (MODEL=folder: another export than model/)
import { readFile, writeFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { Dreamer, startFrames } from "./dreamer.js";

const SIZE = 256 * 256 * 3, TOLERANCE = 2;
const folder = process.env.MODEL ?? "model";
const load = async (name) => new Uint8Array(await readFile(new URL(`${folder}/${name}`, import.meta.url)));
const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
const start = startFrames(meta, await load("context.bin"));
const reference = await load("reference.bin");
const count = reference.length / SIZE;

const dreamer = await Dreamer.create(ort, load, start, meta.actions, { executionProviders: ["cpu"] });

// Each step (its bytes, oldest first) against PyTorch's colours -> the worst step's colour values off.
async function dream(name, steps, reference) {
  const began = performance.now();
  let first = -1, wrong = 0, worst = 0;
  for (let i = 0; i < steps.length; i++) {
    const bytes = steps[i];
    const rgb = await (await dreamer.next(bytes.at(-1), bytes.slice(0, -1).reverse())).getData();
    let differ = 0;
    for (let j = 0; j < SIZE; j++) differ += Math.abs(rgb[j] - reference[i * SIZE + j]) > TOLERANCE;
    if (differ && first < 0) first = i;
    wrong += differ;
    worst = Math.max(worst, differ);
  }
  const fps = steps.length / ((performance.now() - began) / 1000);
  console.log(`${name}: ` + (first < 0 ? `as PyTorch (colours within ${TOLERANCE})` :
              `first differs at +${first + 1}, ${wrong} colour values in all, at most ${worst} in a frame`) +
              `; ${fps.toFixed(1)} steps/s`);
  return worst;
}

let worst = await dream(`${count} frames, no buttons`, Array.from({ length: count }, () => [0]), reference);
if (dreamer.skip > 1) {
  const { steps } = JSON.parse(new TextDecoder().decode(await load("skip_steps.json")));
  await dreamer.reset(start, meta.actions);
  worst = Math.max(worst, await dream(`${steps.length} steps of 1..${dreamer.skip} frames`, steps,
                                      await load("skip_reference.bin")));
  if (process.env.TRACE) {
    await writeFile(process.env.TRACE, JSON.stringify({ ...dreamer.trace, frames: steps.flat() }));
    console.log(`its input trace: ${process.env.TRACE}`);
  }
}
process.exit(worst === 0 ? 0 : 1);
