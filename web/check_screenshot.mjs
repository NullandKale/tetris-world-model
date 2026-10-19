// "Start from an image" in a real browser, on the CPU (?backend=wasm, so a training run keeps the GPU): serves
// web/, makes a messy screenshot of the page's own start (scaled 2.5x, a letterbox, dimmed, noisy), uploads it
// through the panel, checks the panel's picture against the start's, plays from it, then does the same with a
// picture that is not Tetris at all. Prints what it finds and saves what the model drew to --out.
//
//     cd web && node check_screenshot.mjs [--out ../output/screenshot_check]
import { createServer } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const outArg = args.indexOf("--out");
const OUT = outArg >= 0 ? args[outArg + 1] : fileURLToPath(new URL("../output/screenshot_check", import.meta.url));
const ROOT = fileURLToPath(new URL(".", import.meta.url));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };
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

const browser = await puppeteer.launch({ executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
                                         headless: true });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) errors.push(m.text()); });
page.on("response", (r) => { if (r.status() >= 400 && !r.url().endsWith("/favicon.ico")) errors.push(`${r.status()} ${r.url()}`); });
await page.goto(`http://localhost:${server.address().port}/?backend=wasm`);
await page.waitForFunction(() => /frame \d+/.test(document.getElementById("status").textContent), { timeout: 300_000 });
const frameCount = () => page.$eval("#status", (s) => Number((s.textContent.match(/frame (\d+)/) || [0, 0])[1]));
await mkdir(OUT, { recursive: true });

// Upload a picture made in the page (kind: "tetris" or "other") and read the panel's result.
async function upload(kind) {
  if (await page.$eval("#shot", (panel) => panel.hidden)) await page.click("#from-image");
  const handle = await page.evaluateHandle(async (kind) => {
    const { startPicture, pictureImage } = await import("./screenshot.js");
    const { startFrames } = await import("./dreamer.js");
    const meta = await (await fetch("model/context.json")).json();
    const bytes = new Uint8Array(await (await fetch("model/context.bin")).arrayBuffer());
    const real = startPicture(startFrames(meta, bytes));
    const canvas = new OffscreenCanvas(720, 680), c = canvas.getContext("2d");
    c.fillStyle = "rgb(40, 40, 46)";
    c.fillRect(0, 0, canvas.width, canvas.height);
    if (kind === "tetris") {
      const { data } = pictureImage(real, meta.palette);
      for (let i = 0; i < data.length; i += 4)                 // dimmed to 80% and a little noise
        for (let k = 0; k < 3; k++) data[i + k] = Math.max(0, Math.min(255, data[i + k] * 0.8 + (Math.random() - 0.5) * 16));
      const small = new OffscreenCanvas(256, 240);
      small.getContext("2d").putImageData(new ImageData(data, 256, 240), 0, 0);
      c.imageSmoothingEnabled = false;
      c.drawImage(small, 40, 40, 640, 600);                     // 2.5x
    } else {
      const g = c.createLinearGradient(0, 0, canvas.width, canvas.height);
      g.addColorStop(0, "#ff4d8d"); g.addColorStop(0.5, "#ffe066"); g.addColorStop(1, "#3a86ff");
      c.fillStyle = g;
      c.fillRect(0, 0, canvas.width, canvas.height);
    }
    const blob = await canvas.convertToBlob({ type: "image/png" });
    window.__real = real;
    return new File([blob], `${kind}.png`, { type: "image/png" });
  }, kind);
  if (kind === "tetris") await page.$eval("#shot-overscan", (box) => { if (!box.checked) box.click(); });
  await page.evaluate((file) => {
    const input = document.getElementById("shot-file"), transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    input.dispatchEvent(new Event("change"));
  }, handle);
  await page.waitForFunction(() => !document.getElementById("shot-play").disabled, { timeout: 30_000 });
  return page.evaluate(async () => {
    const { pictureImage } = await import("./screenshot.js");
    const meta = await (await fetch("model/context.json")).json();
    const shown = document.getElementById("shot-out").getContext("2d").getImageData(0, 0, 256, 240).data;
    const want = pictureImage(window.__real, meta.palette).data;
    let differ = 0;                                       // NES lines 8-231: the start holds no overscan lines
    for (let p = 8 * 256; p < 232 * 256; p++)
      differ += shown[4 * p] !== want[4 * p] || shown[4 * p + 1] !== want[4 * p + 1] || shown[4 * p + 2] !== want[4 * p + 2];
    return { differ, status: document.getElementById("shot-status").textContent };
  });
}

async function playAndSave(name) {
  const before = await frameCount();
  await page.click("#shot-play");
  await new Promise((resolve) => setTimeout(resolve, 8000));
  const after = await frameCount();
  const shot = await (await page.$("#screen")).screenshot();
  await writeFile(join(OUT, `${name}.png`), shot);
  return { before, after };
}

const tetris = await upload("tetris");
console.log(`tetris screenshot (2.5x, letterbox, dimmed 80%, noise): ${tetris.differ} pixels differ from the real ` +
            `start's picture (NES lines 8-231); ${tetris.status}`);
const played = await playAndSave("from_tetris_screenshot");
console.log(`played from it: frame counter ${played.before} -> ${played.after}`);
const other = await upload("other");
console.log(`a picture that is not Tetris: ${other.status}`);
const played2 = await playAndSave("from_other_picture");
console.log(`played from it: frame counter ${played2.before} -> ${played2.after}`);
console.log(errors.length ? `page errors:\n  ${errors.join("\n  ")}` : "no page errors");
console.log(`what the model drew: ${OUT}`);
await browser.close();
server.close();
process.exit(tetris.differ === 0 && !errors.length ? 0 : 1);
