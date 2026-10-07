// dreamer.js against PyTorch, on the CPU (onnxruntime-node): the page's start, dreamed with no buttons,
// must give the colours scripts/export_onnx.py recorded from PyTorch's Dreamer (model/reference.bin), within
// TOLERANCE per colour value. A broken cache or position changes thousands.
//
//     cd web && npm install && node test_dreamer.mjs      (MODEL=folder: another export than model/)
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { Dreamer, startLayers } from "./dreamer.js";

const SIZE = 256 * 256 * 3, TOLERANCE = 2;
const folder = process.env.MODEL ?? "model";
const load = async (name) => new Uint8Array(await readFile(new URL(`${folder}/${name}`, import.meta.url)));
const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
const start = startLayers(meta, await load("context.bin"));
const reference = await load("reference.bin");
const count = reference.length / SIZE;

const dreamer = await Dreamer.create(ort, load, start, meta.actions, { executionProviders: ["cpu"] });
const began = performance.now();
let first = -1, wrong = 0, worst = 0;
for (let i = 0; i < count; i++) {
  const rgb = await dreamer.next(0);
  let differ = 0;
  for (let j = 0; j < SIZE; j++) differ += Math.abs(rgb[j] - reference[i * SIZE + j]) > TOLERANCE;
  if (differ && first < 0) first = i;
  wrong += differ;
  worst = Math.max(worst, differ);
}
const fps = count / ((performance.now() - began) / 1000);
console.log(`${count} frames: ` +
            (first < 0 ? `as PyTorch (colours within ${TOLERANCE})` : `first differs at +${first + 1}, ${wrong} colour values in all, at most ${worst} in a frame`) +
            `; ${fps.toFixed(1)} frames/s`);
process.exit(worst === 0 ? 0 : 1);
