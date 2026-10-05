// The world model as a game engine in the browser: one generated frame per controller state.
//
// Plays the graphs scripts/export_onnx.py writes (src/token_world/models/onnx_export.py) as
// src/token_world/models/onnx_dreamer.py does in Python. The model is soft: a frame is its colour
// probabilities, kept as context in the form the model reads, its patch tokens (98 KB a frame), and shown
// as each pixel's expected colour. Real frames are embedded (embed.onnx) and fill the key/value cache
// (prefill.onnx); each step decodes the next frame for a controller byte and returns its colours, its
// tokens and the cache with it (step.onnx); when the window is full the last `keep` frames' tokens are
// re-encoded at positions 0..keep-1. A step is one pass over the previous frame (written into the cache)
// and the new one, so the cache holds every frame but the last. On WebGPU the cache stays in GPU buffers
// from step to step; only the picture and the tokens come back. (onnxruntime-web's graph capture would
// replay a step without its dispatches, but up to 1.30 a captured graph fails to replay once another
// session, here prefill, has run on the device.)
//
// `ort` is passed in, so the same class runs on onnxruntime-web (browser) and onnxruntime-node
// (tests): Dreamer.create(ort, load, frames, actions, options), where load(name) resolves a model
// file's bytes.

export class Dreamer {
  static async create(ort, load, frames, actions, { keep = null, executionProviders = ["webgpu"], sessionOptions = {} } = {}) {
    const meta = JSON.parse(new TextDecoder().decode(await load("model.json")));
    keep ??= meta.keep;
    if (keep < 3 || keep >= meta.frames || frames.length < 3) throw new Error(`keep must be 3..${meta.frames - 1}, from 3 or more frames`);
    const onGpu = executionProviders.includes("webgpu");
    const session = async (name, gpuOutputs = []) => ort.InferenceSession.create(await load(name), {
      ...sessionOptions,
      executionProviders,
      ...(onGpu && gpuOutputs.length ? { preferredOutputLocation: Object.fromEntries(gpuOutputs.map((o) => [o, "gpu-buffer"])) } : {}),
    });
    const embed = await session("embed.onnx");
    const prefill = await session("prefill.onnx", ["keys", "values"]);
    const step = await session("step.onnx", ["new_keys", "new_values"]);
    const dreamer = new Dreamer(ort, meta, embed, prefill, step, keep);
    await dreamer.reset(frames, actions);
    return dreamer;
  }

  constructor(ort, meta, embed, prefill, step, keep) {
    Object.assign(this, { ort, meta, embed, prefill, step, keep, keys: null, values: null });
    [this.n, this.dim] = meta.tokens;
  }

  // Start again from real frames (an array of Uint8Array(65536)) and the incoming controller bytes.
  async reset(frames, actions) {
    const pixels = new Uint8Array(frames.length * 65536);
    frames.forEach((f, i) => pixels.set(f, i * 65536));
    const { tokens } = await this.embed.run({ frames: new this.ort.Tensor("uint8", pixels, [frames.length, 256, 256]) });
    const size = this.n * this.dim;
    this.tokens = frames.map((_, i) => tokens.data.slice(i * size, (i + 1) * size));
    this.actions = Array.from(actions, Number);
    await this.encode();
  }

  async encode() {
    this.tokens = this.tokens.slice(-this.keep);
    this.actions = this.actions.slice(-this.keep);
    const t = this.tokens.length - 1;                  // the last frame goes in with the next step
    const size = this.n * this.dim, all = new Float32Array(t * size);
    this.tokens.slice(0, t).forEach((x, i) => all.set(x, i * size));
    const out = await this.prefill.run({
      tokens: new this.ort.Tensor("float32", all, [t, this.n, this.dim]),
      actions: new this.ort.Tensor("int32", Int32Array.from(this.actions.slice(0, t)), [t]),
    });
    this.replaceCache(out.keys, out.values);
  }

  // The frame that controller byte `action` produces: each pixel's expected colour, Float32Array(256 * 256 * 3).
  async next(action) {
    if (this.tokens.length === this.meta.frames) await this.encode();
    const int32 = (values) => new this.ort.Tensor("int32", Int32Array.from(values), [values.length]);
    const out = await this.step.run({
      previous: new this.ort.Tensor("float32", this.tokens.at(-1), [this.n, this.dim]),
      actions: int32([this.actions.at(-1), action]), at: int32([this.tokens.length]),
      keys: this.keys, values: this.values });
    this.replaceCache(out.new_keys, out.new_values);
    this.tokens.push(Float32Array.from(out.tokens.data));
    this.actions.push(action);
    return out.rgb.data;
  }

  replaceCache(keys, values) {
    for (const old of [this.keys, this.values]) if (old && old.location === "gpu-buffer") old.dispose();
    this.keys = keys;
    this.values = values;
  }
}
