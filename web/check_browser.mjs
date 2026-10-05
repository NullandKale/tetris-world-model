// The page in Chrome with WebGPU: serves web/, opens it with ?check (dream the start with no buttons and
// compare with PyTorch's frames), then lets it play a few seconds and reports the check, the frame rate
// and anything onnxruntime-web logged (nodes it could not run on WebGPU, errors). Uses the GPU, so not
// while training runs.
//
//     cd web && npm install && node check_browser.mjs [path to chrome.exe] [--root ../site/web]
//
// --root serves another folder (a built site's page, site/web); without model/reference.bin there it plays only.
import { createServer } from "node:http";
import { existsSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const rootArg = process.argv.indexOf("--root");
const ROOT = rootArg > 0 ? join(process.cwd(), process.argv[rootArg + 1]) : fileURLToPath(new URL(".", import.meta.url));
const checking = existsSync(join(ROOT, "model", "reference.bin"));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };
const chrome = process.argv.slice(2).find((a, i, all) => !a.startsWith("--") && all[i - 1] !== "--root")
  ?? "C:/Program Files/Google/Chrome/Application/chrome.exe";

const server = createServer(async (request, response) => {
  const asked = normalize(join(ROOT, decodeURIComponent(new URL(request.url, "http://x").pathname)));
  const path = asked.endsWith("\\") || asked.endsWith("/") ? join(asked, "index.html") : asked;
  try {
    const body = await readFile(path);
    response.writeHead(200, { "Content-Type": TYPES[extname(path)] ?? "application/octet-stream" });
    response.end(body);
  } catch {
    response.writeHead(404).end();
  }
}).listen(0);
const profile = process.argv.includes("--profile");
const url = `http://localhost:${server.address().port}/` + (checking ? `?check${profile ? "&profile" : ""}` : "");

const browser = await puppeteer.launch({ executablePath: chrome, headless: true,
                                         args: ["--enable-unsafe-webgpu", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
const logged = [], all = [];
page.on("console", (m) => {
  all.push(m.text());
  if (!profile && (m.type() !== "log" || /onnxruntime|ort|webgpu|DBG/i.test(m.text()))) logged.push(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => logged.push(`page error: ${e.message}`));
const status = async () => ((await page.$("#status")) ? page.$eval("#status", (s) => s.textContent) : "");
await page.goto(url);
console.log("adapter:", await page.evaluate(async () => {
  const a = await navigator.gpu?.requestAdapter();
  return a ? JSON.stringify({ vendor: a.info.vendor, architecture: a.info.architecture, description: a.info.description,
                              fallback: a.info.isFallbackAdapter }) : "none";
}));
try {
  if (checking) {
    await page.waitForSelector("body[data-check]", { timeout: 300_000 });
    console.log(await page.$eval("body", (b) => b.dataset.check));
  }
  await new Promise((resolve) => setTimeout(resolve, checking ? 5000 : 20000));   // a site warms up while playing
  console.log((await status()).replaceAll("\n", " | "));
} catch (error) {
  console.log(`no check result: ${error.message}; status: ${await status()}`);
}
for (const line of [...new Set(logged)].slice(0, 30)) console.log(line);
if (profile) report(all);
await browser.close();
server.close();

// --profile: GPU time per kernel type over the run, and the nodes onnxruntime placed on the CPU.
function report(lines) {
  writeFileSync(new URL("profile.log", import.meta.url), lines.join("\n"));
  const kernels = new Map();
  let total = 0;
  for (const line of lines) {
    const m = line.match(/\[profiling\] kernel "([^"|]*)\|([^"|]*)\|?[^"]*".*execution time: (\d+) ns/);
    if (!m) continue;
    const ns = Number(m[3]);
    total += ns;
    kernels.set(m[2], (kernels.get(m[2]) ?? 0) + ns);
  }
  console.log(`GPU kernel time ${(total / 1e6).toFixed(0)} ms in all`);
  for (const [op, ns] of [...kernels].sort((a, b) => b[1] - a[1]).slice(0, 15))
    console.log(`  ${op.padEnd(24)} ${(ns / 1e6).toFixed(1).padStart(8)} ms  ${(100 * ns / total).toFixed(1)}%`);
  const cpu = lines.filter((l) => /CPUExecutionProvider|assigned to.*CPU|Node placements/i.test(l));
  console.log(`placement lines: ${cpu.length}`);
  for (const line of [...new Set(cpu)].slice(0, 40)) console.log("  " + line.replace(/\x1b\[[0-9;]*m/g, "").slice(0, 200));
}
