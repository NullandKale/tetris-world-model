// The world model as a game engine in the browser: one generated frame per controller state.
//
// Plays the graphs scripts/export_onnx.py writes (src/token_world/models/onnx_export.py) as
// src/token_world/models/onnx_dreamer.py does in Python. The model is the pixel one with motion first: a frame
// is kept as context in the form the model reads, its content (its patch tokens, soft: each pixel's colour
// probabilities through the model's colour table, [tokens, dim] float32) and its motion (each token's 8 x 8
// cells' codes, [tokens, cells] float32; meta.unknown where not known, as for the start's real frames), and shown
// as each pixel's most likely colour. The start's real frames are embedded (embed.onnx) and fill the key/value
// cache (prefill.onnx); each step decodes the next frame for a controller byte, its motion first and then its
// pixels, and returns its colours, its content, its motion and the previous frame's keys and values
// (step.onnx), which go into the cache in place; when the window is full the last `keep` frames are encoded
// again at positions 0..keep-1. A step reads the previous frame and the new one, so the cache holds every frame
// but the last.
//
// On WebGPU the dream stays on the GPU: the cache (one keys and one values tensor a layer, model.json's cache:
// keys [tokens, heads, head_dim, frames], values [tokens, heads, frames, head_dim]) is written in place by a
// compute shader (WRITE below), each frame's content and motion go from one step to the next as GPU buffers and
// are stacked by GPU copies for a re-encode, and next() returns the frame's colours as a GPU tensor, which
// GpuScreen draws: nothing comes back to the CPU. Reading the picture back each frame left the GPU idle while
// the page waited, and the driver then lowered its clocks (an RTX 3090 fell to its 810 MHz memory clock after
// about 150 steps: 70 ms a step instead of 24, 2026-10-18). At most IN_FLIGHT steps are queued ahead of the GPU,
// so the buttons stay that many frames from the screen at most. Elsewhere (WebAssembly, onnxruntime-node) every
// tensor is on the CPU.
//
// `ort` is passed in, so the same class runs on onnxruntime-web (browser) and onnxruntime-node (tests):
// Dreamer.create(ort, load, start, actions, options), where load(name) resolves a model file's bytes and
// start holds the start's frames: { data: Uint8Array of palette indices, shape: [frames, 256, 256] }.
// next(action) returns the frame's colours as an ort.Tensor [256, 256, 3] float32 (getData() reads them).
// dreamer.timing holds the last step's times (ms): run (the graph's run, with the wait for the step before on
// WebGPU), write (the cache write) and encode (a window slide's re-encode, 0 when none); dreamer.runs counts each
// graph's runs and dreamer.sessions holds them (diagnostics.js profile). options.log(text) hears each graph's
// setup time.

// Steps queued on the GPU before the next one waits: Firefox tells a step's work is done only on a coarse timer
// (waiting for each one: 103 ms a step on a 3090 against 42 with two in flight), Chrome at once (25 against 23);
// a third gained nothing in either (2026-10-18).
const IN_FLIGHT = 2;

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
    const step = await session("step.onnx", ["rgb", "content", "motion", "new_keys", "new_values"]);
    const device = onGpu ? await ort.env.webgpu.device : null;
    const writer = onGpu ? new GpuWriter(device, meta) : new CpuWriter(meta);
    const dreamer = new Dreamer(ort, meta, embed, prefill, step, keep, writer, device);
    await dreamer.reset(start, actions);
    return dreamer;
  }

  constructor(ort, meta, embed, prefill, step, keep, writer, device) {
    Object.assign(this, { ort, meta, embed, prefill, step, keep, writer, device, cache: null, flight: [] });
    [this.n, this.dim] = meta.content;
    this.cells = meta.motion[1];
    this.sessions = { embed, prefill, step };
    this.runs = { embed: 0, prefill: 0, step: 0 };
    this.timing = { run: 0, write: 0, encode: 0 };
    this.content = [];
    this.motion = [];
    if (device) {                                          // the re-encode's inputs, stacked on the GPU
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      this.stacked = { content: device.createBuffer({ size: meta.prefill * this.n * this.dim * 4, usage }),
                       motion: device.createBuffer({ size: meta.prefill * this.n * this.cells * 4, usage }) };
    }
  }

  // Start again from the start's real frames and their incoming controller bytes (their motion not known).
  async reset(start, actions) {
    await Promise.all(this.flight.splice(0));
    const { content } = await this.embed.run({ frames: new this.ort.Tensor("uint8", start.data, start.shape) });
    this.runs.embed++;
    const size = this.n * this.dim, cells = this.n * this.cells;
    this.drop(this.content.concat(this.motion));
    this.content = Array.from(actions, (_, i) =>
      new this.ort.Tensor("float32", content.data.slice(i * size, (i + 1) * size), [this.n, this.dim]));
    this.motion = Array.from(actions, () =>
      new this.ort.Tensor("float32", new Float32Array(cells).fill(this.meta.unknown), [this.n, this.cells]));
    this.actions = Array.from(actions, Number);
    await this.encode();
  }

  drop(tensors) {
    for (const t of tensors) if (t.location === "gpu-buffer") t.dispose();
  }

  async encode() {
    this.drop(this.content.slice(0, -this.keep).concat(this.motion.slice(0, -this.keep)));
    this.content = this.content.slice(-this.keep);
    this.motion = this.motion.slice(-this.keep);
    this.actions = this.actions.slice(-this.keep);
    // the last frame goes in with the next step; prefill takes a fixed meta.prefill frames, these padded after
    // them (no real frame sees a later one, and the steps write over them)
    const t = this.content.length - 1, p = this.meta.prefill;
    const actions = new Int32Array(p).fill(-1);
    actions.set(this.actions.slice(0, t));
    const out = await this.prefill.run({
      content: this.stack("content", this.content.slice(0, t), [p, this.n, this.dim]),
      motion: this.stack("motion", this.motion.slice(0, t), [p, this.n, this.cells]),
      actions: new this.ort.Tensor("int32", actions, [p]),
    });
    this.runs.prefill++;
    this.drop(this.cache ?? []);
    this.cache = this.meta.cache.names.map((name) => out[name]);
  }

  // Frames' tensors (each [n, k], on the GPU or the CPU) one after another and zero after them, as one tensor
  // of `dims`: on the GPU by copies into this.stacked[name], elsewhere in an array.
  stack(name, tensors, dims) {
    const size = dims[1] * dims[2];
    if (!this.device) {
      const all = new Float32Array(dims[0] * size);
      tensors.forEach((x, i) => all.set(x.data, i * size));
      return new this.ort.Tensor("float32", all, dims);
    }
    const target = this.stacked[name], encoder = this.device.createCommandEncoder();
    encoder.clearBuffer(target);
    tensors.forEach((x, i) => {
      if (x.location === "gpu-buffer") encoder.copyBufferToBuffer(x.gpuBuffer, 0, target, i * size * 4, size * 4);
    });
    this.device.queue.submit([encoder.finish()]);
    tensors.forEach((x, i) => { if (x.location !== "gpu-buffer") this.device.queue.writeBuffer(target, i * size * 4, x.data); });
    return this.ort.Tensor.fromGpuBuffer(target, { dataType: "float32", dims });
  }

  // The frame that controller byte `action` produces: its colours, an ort.Tensor [256, 256, 3] (on the GPU on
  // WebGPU: GpuScreen draws it; getData() reads it; dispose() it when drawn).
  async next(action) {
    let began = performance.now();
    this.timing.encode = 0;
    if (this.content.length === this.meta.frames) {
      await this.encode();
      this.timing.encode = performance.now() - began;
      began = performance.now();
    }
    if (this.flight.length >= IN_FLIGHT) await this.flight.shift();     // IN_FLIGHT steps ahead of the GPU at most
    const at = this.content.length;
    const int32 = (values) => new this.ort.Tensor("int32", Int32Array.from(values), [values.length]);
    const out = await this.step.run({
      previous: this.content.at(-1), previous_motion: this.motion.at(-1),
      actions: int32([this.actions.at(-1), action]), at: int32([at]),
      ...Object.fromEntries(this.meta.cache.names.map((name, i) => [name, this.cache[i]])) });
    const ran = performance.now();
    this.runs.step++;
    this.writer.write(this.cache, out.new_keys, out.new_values, at - 1);     // the previous frame, at at - 1
    this.drop([out.new_keys, out.new_values]);
    this.content.push(out.content);
    this.motion.push(out.motion);
    this.actions.push(action);
    if (this.device) this.flight.push(this.device.queue.onSubmittedWorkDone());
    this.timing.run = ran - began;
    this.timing.write = performance.now() - ran;
    return out.rgb;
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

// A frame's colours on the GPU ([256, 256, 3] float32, 0-255) onto a canvas's WebGPU context: one triangle over
// the canvas, each pixel reading its colour from the buffer.
const DRAW = /* wgsl */ `
@group(0) @binding(0) var<storage, read> rgb: array<f32>;

@vertex
fn corner(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn colour(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
  let p = (u32(at.y) * 256u + u32(at.x)) * 3u;
  return vec4<f32>(rgb[p], rgb[p + 1u], rgb[p + 2u], 255.0) / 255.0;
}`;

export class GpuScreen {
  constructor(device, canvas) {
    this.device = device;
    this.context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format, alphaMode: "opaque" });
    const module = device.createShaderModule({ code: DRAW });
    this.pipeline = device.createRenderPipeline({ layout: "auto", vertex: { module, entryPoint: "corner" },
                                                  fragment: { module, entryPoint: "colour", targets: [{ format }] } });
  }

  draw(rgb) {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(),
                                                                loadOp: "clear", storeOp: "store" }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0),
                                                       entries: [{ binding: 0, resource: { buffer: rgb.gpuBuffer } }] }));
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

// The page's start (scripts/export_onnx.py): context.bin is its frames, context.json their shape.
export function startFrames(meta, bytes) {
  return { data: bytes, shape: meta.shape };
}
