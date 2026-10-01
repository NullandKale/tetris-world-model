// dreamer.js against PyTorch, on the CPU (onnxruntime-node): the page's start, dreamed with no buttons,
// must give the frames scripts/export_onnx.py recorded from PyTorch's Dreamer (model/reference.bin), up
// to TIES pixels a frame: the boot screens' blacks are separate palette entries the model cannot tell
// apart (a logit gap of 6e-6), which another runtime's rounding picks differently. A broken cache or
// position changes thousands.
//
//     cd web && npm install && node test_dreamer.mjs
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { Dreamer } from "./dreamer.js";

const SIZE = 256 * 256, TIES = 4;
const load = async (name) => new Uint8Array(await readFile(new URL(`model/${name}`, import.meta.url)));
const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
const pixels = await load("context.bin");
const frames = Array.from({ length: meta.actions.length }, (_, i) => pixels.subarray(i * SIZE, (i + 1) * SIZE));
const reference = await load("reference.bin");
const count = reference.length / SIZE;

const dreamer = await Dreamer.create(ort, load, frames, meta.actions,
                                     { level: meta.level, keep: meta.keep, executionProviders: ["cpu"] });
const began = performance.now();
let first = -1, wrong = 0, worst = 0;
for (let i = 0; i < count; i++) {
  const frame = await dreamer.next(0);
  const expected = reference.subarray(i * SIZE, (i + 1) * SIZE);
  let differ = 0;
  for (let j = 0; j < SIZE; j++) differ += frame[j] !== expected[j];
  if (differ && first < 0) first = i;
  wrong += differ;
  worst = Math.max(worst, differ);
}
const fps = count / ((performance.now() - began) / 1000);
console.log(`${count} frames at level ${meta.level}: ` +
            (first < 0 ? "identical to PyTorch" : `first differs at +${first + 1}, ${wrong} pixels in all, at most ${worst} in a frame`) +
            `; ${fps.toFixed(1)} frames/s`);
process.exit(worst <= TIES ? 0 : 1);
