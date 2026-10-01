// The world model as a game engine in the browser: one generated frame per controller state.
//
// Plays the graphs scripts/export_onnx.py writes (src/token_world/models/onnx_export.py) as
// src/token_world/models/onnx_dreamer.py does in Python: real frames fill the key/value cache
// (prefill.onnx), each step decodes the next frame for a controller byte and returns the cache with
// it (step.onnx), and when the window is full the last `keep` frames are re-encoded at positions
// 0..keep-1. Generated frames carry `level`, real ones 0. A step is one pass over the previous frame
// (written into the cache) and the new one, so the cache holds every frame but the last. On WebGPU the
// cache stays in GPU buffers from step to step; only the frame comes back. (onnxruntime-web's graph
// capture would replay a step without its dispatches, but up to 1.30 a captured graph fails to replay
// once another session, here prefill, has run on the device.)
//
// `ort` is passed in, so the same class runs on onnxruntime-web (browser) and onnxruntime-node
// (tests): Dreamer.create(ort, load, frames, actions, options), where load(name) resolves a model
// file's bytes.

export class Dreamer {
  static async create(ort, load, frames, actions,
                      { level = 0, keep = 48, executionProviders = ["webgpu"], sessionOptions = {} } = {}) {
    const meta = JSON.parse(new TextDecoder().decode(await load("model.json")));
    if (keep < 3 || keep >= meta.frames || frames.length < 3) throw new Error(`keep must be 3..${meta.frames - 1}, from 3 or more frames`);
    const onGpu = executionProviders.includes("webgpu");
    const session = async (name, gpuOutputs) => ort.InferenceSession.create(await load(name), {
      ...sessionOptions,
      executionProviders,
      ...(onGpu ? { preferredOutputLocation: Object.fromEntries(gpuOutputs.map((o) => [o, "gpu-buffer"])) } : {}),
    });
    const prefill = await session("prefill.onnx", ["keys", "values"]);
    const step = await session("step.onnx", ["new_keys", "new_values"]);
    const dreamer = new Dreamer(ort, meta, prefill, step, level, keep);
    await dreamer.reset(frames, actions);
    return dreamer;
  }

  constructor(ort, meta, prefill, step, level, keep) {
    Object.assign(this, { ort, meta, prefill, step, level, keep, keys: null, values: null });
  }

  // Start again from real frames (an array of Uint8Array(65536)) and the incoming controller bytes.
  async reset(frames, actions) {
    this.frames = frames.map((f) => f.slice());
    this.actions = Array.from(actions, Number);
    this.levels = this.frames.map(() => 0);
    await this.encode();
  }

  async encode() {
    this.frames = this.frames.slice(-this.keep);
    this.actions = this.actions.slice(-this.keep);
    this.levels = this.levels.slice(-this.keep);
    const t = this.frames.length - 1;                  // the last frame goes in with the next step
    const pixels = new Uint8Array(t * 65536);
    this.frames.slice(0, t).forEach((f, i) => pixels.set(f, i * 65536));
    const out = await this.prefill.run({
      frames: new this.ort.Tensor("uint8", pixels, [t, 256, 256]),
      actions: new this.ort.Tensor("int32", Int32Array.from(this.actions.slice(0, t)), [t]),
      levels: new this.ort.Tensor("int32", Int32Array.from(this.levels.slice(0, t)), [t]),
    });
    this.replaceCache(out.keys, out.values);
  }

  // The frame (Uint8Array(65536) of palette indices) that controller byte `action` produces.
  async next(action) {
    if (this.frames.length === this.meta.frames) await this.encode();
    const int32 = (values) => new this.ort.Tensor("int32", Int32Array.from(values), [values.length]);
    const out = await this.step.run({
      previous: new this.ort.Tensor("int32", Int32Array.from(this.frames.at(-1)), [256, 256]),
      actions: int32([this.actions.at(-1), action]), levels: int32([this.levels.at(-1), this.level]),
      at: int32([this.frames.length]), keys: this.keys, values: this.values });
    this.replaceCache(out.new_keys, out.new_values);
    const frame = Uint8Array.from(out.frame.data);
    this.frames.push(frame);
    this.actions.push(action);
    this.levels.push(this.level);
    return frame;
  }

  replaceCache(keys, values) {
    for (const old of [this.keys, this.values]) if (old && old.location === "gpu-buffer") old.dispose();
    this.keys = keys;
    this.values = values;
  }
}
