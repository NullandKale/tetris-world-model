// The world model as a game engine in the browser: one generated frame per controller state.
//
// Plays the graphs scripts/export_onnx.py writes (src/token_world/models/onnx_export.py) as
// src/token_world/models/onnx_dreamer.py does in Python. The model is the layered one: a frame is its layers
// (background cells, the sprite picture, the border bands), kept as context in the form the model reads, its
// content (what each token holds, soft: each pixel's colour probabilities through the model's colour table,
// [content tokens, dim] float32), and shown composed, each pixel its most likely colour. The start's real
// layers are embedded (embed.onnx) and fill the key/value cache (prefill.onnx); each step decodes the next
// frame for a controller byte and returns its colours, its content and the cache with it (step.onnx); when the
// window is full the last `keep` frames' content is encoded again at positions 0..keep-1. A step is one pass
// over the previous frame (written into the cache) and the new one, so the cache holds every frame but the
// last. On WebGPU the cache stays in GPU buffers from step to step; only the picture and the content come back.
//
// `ort` is passed in, so the same class runs on onnxruntime-web (browser) and onnxruntime-node (tests):
// Dreamer.create(ort, load, start, actions, options), where load(name) resolves a model file's bytes and
// start holds the start's layers: for each of model.json's inputs, { data: Uint8Array, shape }.

export class Dreamer {
  static async create(ort, load, start, actions, { keep = null, executionProviders = ["webgpu"], sessionOptions = {} } = {}) {
    const meta = JSON.parse(new TextDecoder().decode(await load("model.json")));
    keep ??= meta.keep;
    if (keep < 3 || keep >= meta.frames || actions.length < 3) throw new Error(`keep must be 3..${meta.frames - 1}, from 3 or more frames`);
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
    await dreamer.reset(start, actions);
    return dreamer;
  }

  constructor(ort, meta, embed, prefill, step, keep) {
    Object.assign(this, { ort, meta, embed, prefill, step, keep, keys: null, values: null });
    [this.n, this.dim] = meta.content;
  }

  // Start again from the start's real layers and their incoming controller bytes.
  async reset(start, actions) {
    const feeds = Object.fromEntries(this.meta.inputs.map((name) =>
      [name, new this.ort.Tensor("uint8", start[name].data, start[name].shape)]));
    const { content } = await this.embed.run(feeds);
    const size = this.n * this.dim;
    this.content = Array.from(actions, (_, i) => content.data.slice(i * size, (i + 1) * size));
    this.actions = Array.from(actions, Number);
    await this.encode();
  }

  async encode() {
    this.content = this.content.slice(-this.keep);
    this.actions = this.actions.slice(-this.keep);
    const t = this.content.length - 1;                 // the last frame goes in with the next step
    const size = this.n * this.dim, all = new Float32Array(t * size);
    this.content.slice(0, t).forEach((x, i) => all.set(x, i * size));
    const out = await this.prefill.run({
      content: new this.ort.Tensor("float32", all, [t, this.n, this.dim]),
      actions: new this.ort.Tensor("int32", Int32Array.from(this.actions.slice(0, t)), [t]),
    });
    this.replaceCache(out.keys, out.values);
  }

  // The frame that controller byte `action` produces: its colours, Float32Array(256 * 256 * 3).
  async next(action) {
    if (this.content.length === this.meta.frames) await this.encode();
    const int32 = (values) => new this.ort.Tensor("int32", Int32Array.from(values), [values.length]);
    const out = await this.step.run({
      previous: new this.ort.Tensor("float32", this.content.at(-1), [this.n, this.dim]),
      actions: int32([this.actions.at(-1), action]), at: int32([this.content.length]),
      keys: this.keys, values: this.values });
    this.replaceCache(out.new_keys, out.new_values);
    this.content.push(Float32Array.from(out.content.data));
    this.actions.push(action);
    return out.rgb.data;
  }

  replaceCache(keys, values) {
    for (const old of [this.keys, this.values]) if (old && old.location === "gpu-buffer") old.dispose();
    this.keys = keys;
    this.values = values;
  }
}

// The page's start (scripts/export_onnx.py): context.json says where each layer sits in context.bin.
export function startLayers(meta, bytes) {
  return Object.fromEntries(meta.inputs.map(({ name, shape, offset }) =>
    [name, { data: bytes.subarray(offset, offset + shape.reduce((a, b) => a * b, 1)), shape }]));
}
