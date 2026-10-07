// The world model as a game engine in the browser: one generated frame per controller state.
//
// Plays the graphs scripts/export_onnx.py writes (src/token_world/models/onnx_export.py) as
// src/token_world/models/onnx_dreamer.py does in Python. The model is the layered one: a frame is its layers
// (background cells, the sprite picture, the border bands), kept as context in the form the model reads, its
// content (what each token holds, soft: each pixel's colour probabilities through the model's colour table,
// [content tokens, dim] float32), and shown composed, each pixel its most likely colour. The start's real
// layers are embedded (embed.onnx) and fill the key/value cache (prefill.onnx); each step decodes the next
// frame for a controller byte and returns its colours, its content and the previous frame's keys and values
// (step.onnx), which go into the cache in place; when the window is full the last `keep` frames' content is
// encoded again at positions 0..keep-1. A step is one pass over the previous frame and the new one, so the
// cache holds every frame but the last.
//
// The cache is one keys and one values tensor a layer (model.json's cache: keys [tokens, heads, head_dim,
// frames], values [tokens, heads, frames, head_dim]). On WebGPU it stays in GPU buffers for the whole dream:
// a compute shader writes each step's frame into them (WRITE below), so only the picture and the content
// cross to the CPU. The graph never copies the cache (passing it through whole rewrote all of it every step).
//
// `ort` is passed in, so the same class runs on onnxruntime-web (browser) and onnxruntime-node (tests):
// Dreamer.create(ort, load, start, actions, options), where load(name) resolves a model file's bytes and
// start holds the start's layers: for each of model.json's inputs, { data: Uint8Array, shape }.
// dreamer.timing holds the last step's times (ms): run (the graph, with the picture's download), write
// (the cache write) and encode (a window slide's re-encode, 0 when none); dreamer.runs counts each graph's runs
// and dreamer.sessions holds them (diagnostics.js profile). options.log(text) hears each graph's setup time.

export class Dreamer {
  static async create(ort, load, start, actions, { keep = null, executionProviders = ["webgpu"], sessionOptions = {},
                                                   log = () => {} } = {}) {
    const meta = JSON.parse(new TextDecoder().decode(await load("model.json")));
    keep ??= meta.keep;
    if (keep < 3 || keep > meta.prefill + 1 || actions.length < 3 || actions.length > meta.prefill + 1)
      throw new Error(`keep and the start: 3..${meta.prefill + 1} frames`);
    const onGpu = executionProviders.includes("webgpu");
    const session = async (name, gpuOutputs = []) => {
      const bytes = await load(name), began = performance.now();
      const made = await ort.InferenceSession.create(bytes, {
        ...sessionOptions,
        executionProviders,
        ...(onGpu && gpuOutputs.length ? { preferredOutputLocation: Object.fromEntries(gpuOutputs.map((o) => [o, "gpu-buffer"])) } : {}),
      });
      log(`${name}: set up on ${executionProviders.join(", ")} in ${(performance.now() - began).toFixed(0)} ms`);
      return made;
    };
    const embed = await session("embed.onnx");
    const prefill = await session("prefill.onnx", meta.cache.names);
    const step = await session("step.onnx", ["new_keys", "new_values"]);
    const writer = onGpu ? new GpuWriter(await ort.env.webgpu.device, meta) : new CpuWriter(meta);
    const dreamer = new Dreamer(ort, meta, embed, prefill, step, keep, writer);
    await dreamer.reset(start, actions);
    return dreamer;
  }

  constructor(ort, meta, embed, prefill, step, keep, writer) {
    Object.assign(this, { ort, meta, embed, prefill, step, keep, writer, cache: null });
    [this.n, this.dim] = meta.content;
    this.sessions = { embed, prefill, step };
    this.runs = { embed: 0, prefill: 0, step: 0 };
    this.timing = { run: 0, write: 0, encode: 0 };
  }

  // Start again from the start's real layers and their incoming controller bytes.
  async reset(start, actions) {
    const feeds = Object.fromEntries(this.meta.inputs.map((name) =>
      [name, new this.ort.Tensor("uint8", start[name].data, start[name].shape)]));
    const { content } = await this.embed.run(feeds);
    this.runs.embed++;
    const size = this.n * this.dim;
    this.content = Array.from(actions, (_, i) => content.data.slice(i * size, (i + 1) * size));
    this.actions = Array.from(actions, Number);
    await this.encode();
  }

  async encode() {
    this.content = this.content.slice(-this.keep);
    this.actions = this.actions.slice(-this.keep);
    // the last frame goes in with the next step; prefill takes a fixed meta.prefill frames, these padded after
    // them (no real frame sees a later one, and the steps write over them)
    const t = this.content.length - 1, p = this.meta.prefill;
    const size = this.n * this.dim, all = new Float32Array(p * size);
    this.content.slice(0, t).forEach((x, i) => all.set(x, i * size));
    const actions = new Int32Array(p).fill(-1);
    actions.set(this.actions.slice(0, t));
    const out = await this.prefill.run({
      content: new this.ort.Tensor("float32", all, [p, this.n, this.dim]),
      actions: new this.ort.Tensor("int32", actions, [p]),
    });
    this.runs.prefill++;
    for (const old of this.cache ?? []) if (old.location === "gpu-buffer") old.dispose();
    this.cache = this.meta.cache.names.map((name) => out[name]);
  }

  // The frame that controller byte `action` produces: its colours, Float32Array(256 * 256 * 3).
  async next(action) {
    let began = performance.now();
    this.timing.encode = 0;
    if (this.content.length === this.meta.frames) {
      await this.encode();
      this.timing.encode = performance.now() - began;
      began = performance.now();
    }
    const at = this.content.length;
    const int32 = (values) => new this.ort.Tensor("int32", Int32Array.from(values), [values.length]);
    const out = await this.step.run({
      previous: new this.ort.Tensor("float32", this.content.at(-1), [this.n, this.dim]),
      actions: int32([this.actions.at(-1), action]), at: int32([at]),
      ...Object.fromEntries(this.meta.cache.names.map((name, i) => [name, this.cache[i]])) });
    const ran = performance.now();
    this.runs.step++;
    this.writer.write(this.cache, out.new_keys, out.new_values, at - 1);     // the previous frame, at at - 1
    for (const t of [out.new_keys, out.new_values]) if (t.location === "gpu-buffer") t.dispose();
    this.content.push(Float32Array.from(out.content.data));
    this.actions.push(action);
    this.timing.run = ran - began;
    this.timing.write = performance.now() - ran;
    return out.rgb.data;
  }
}

// The cache write: a step's new_keys, new_values [layers, tokens, heads, head_dim] into each layer's keys
// [tokens, heads, head_dim, frames] and values [tokens, heads, frames, head_dim] at frame `at`.
const WRITE = /* wgsl */ `
struct Write { at: u32, base: u32, frames: u32, hd: u32, count: u32, keys: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;
@group(0) @binding(2) var<uniform> w: Write;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= w.count) { return; }
  let row = i / w.hd;                                     // token * heads + head
  let d = i % w.hd;
  var to = (row * w.frames + w.at) * w.hd + d;            // values
  if (w.keys == 1u) { to = (row * w.hd + d) * w.frames + w.at; }
  cache[to] = source[w.base + i];
}`;

const UNIFORM = 256;                                      // a dispatch's uniforms: offsets must be multiples of 256

class GpuWriter {
  constructor(device, meta) {
    if (!device) throw new Error("onnxruntime-web gave no WebGPU device (ort.env.webgpu.device)");
    const [n, h, hd, frames] = meta.cache.keys;
    this.device = device;
    this.layers = meta.cache.layers;
    this.count = n * h * hd;
    this.pipeline = device.createComputePipeline({
      layout: "auto", compute: { module: device.createShaderModule({ code: WRITE }), entryPoint: "main" } });
    this.uniforms = device.createBuffer({ size: 2 * this.layers * UNIFORM, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fields = new Uint32Array(2 * this.layers * UNIFORM / 4);
    for (let j = 0; j < 2 * this.layers; j++)            // keys' layers, then values'
      this.fields.set([0, (j % this.layers) * this.count, frames, hd, this.count, j < this.layers ? 1 : 0], j * UNIFORM / 4);
  }

  write(cache, keys, values, at) {
    for (let j = 0; j < 2 * this.layers; j++) this.fields[j * UNIFORM / 4] = at;
    this.device.queue.writeBuffer(this.uniforms, 0, this.fields);
    const encoder = this.device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    for (let j = 0; j < 2 * this.layers; j++) {
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: (j < this.layers ? keys : values).gpuBuffer } },
        { binding: 1, resource: { buffer: cache[j].gpuBuffer } },
        { binding: 2, resource: { buffer: this.uniforms, offset: j * UNIFORM, size: 32 } }] }));
      pass.dispatchWorkgroups(Math.ceil(this.count / 256));
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

class CpuWriter {
  constructor(meta) {
    [this.n, this.h, this.hd, this.frames] = meta.cache.keys;
    this.layers = meta.cache.layers;
  }

  write(cache, keys, values, at) {
    const { n, h, hd, frames, layers } = this, count = n * h * hd;
    for (let l = 0; l < layers; l++) {
      const k = cache[l].data, v = cache[layers + l].data, base = l * count;
      for (let row = 0; row < n * h; row++)
        for (let d = 0; d < hd; d++) {
          const i = row * hd + d;
          k[(row * hd + d) * frames + at] = keys.data[base + i];
          v[(row * frames + at) * hd + d] = values.data[base + i];
        }
    }
  }
}

// The page's start (scripts/export_onnx.py): context.json says where each layer sits in context.bin.
export function startLayers(meta, bytes) {
  return Object.fromEntries(meta.inputs.map(({ name, shape, offset }) =>
    [name, { data: bytes.subarray(offset, offset + shape.reduce((a, b) => a * b, 1)), shape }]));
}
