// The page in a real browser with WebGPU: serves web/, opens it with ?check (dream the start with no buttons and
// compare with PyTorch's frames), then lets it play a few seconds and prints the check, the status line and the
// page's console lines (diagnostics.js: the GPU, load and setup times, frame times, errors). Uses the GPU, so
// not while training runs (or expect slow times).
//
//     cd web && npm install && node check_browser.mjs [browser executable] [--root ../site/web] [--profile] [--firefox]
//
// --root serves another folder (a built site's page, site/web); without model/reference.bin there it plays only.
// --profile adds ?profile: each graph's GPU time by operation and node, after the page's PROFILE_FRAMES frames.
// --firefox runs Firefox (WebDriver BiDi) instead of Chrome.
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const rootArg = args.indexOf("--root");
const ROOT = rootArg >= 0 ? join(process.cwd(), args[rootArg + 1]) : fileURLToPath(new URL(".", import.meta.url));
const checking = existsSync(join(ROOT, "model", "reference.bin"));
const profile = args.includes("--profile"), firefox = args.includes("--firefox");
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };
const executable = args.find((a, i) => !a.startsWith("--") && args[i - 1] !== "--root")
  ?? (firefox ? "C:/Program Files/Mozilla Firefox/firefox.exe" : "C:/Program Files/Google/Chrome/Application/chrome.exe");

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
const options = [checking && "check", profile && "profile"].filter(Boolean);
const url = `http://localhost:${server.address().port}/` + (options.length ? `?${options.join("&")}` : "");

const browser = await puppeteer.launch(firefox
  ? { browser: "firefox", executablePath: executable, headless: true,
      extraPrefsFirefox: { "dom.webgpu.enabled": true, "gfx.webgpu.ignore-blocklist": true } }
  : { executablePath: executable, headless: true, args: ["--enable-unsafe-webgpu", "--enable-gpu", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
page.on("console", (m) => {
  if (m.text().startsWith("[world-model]") || m.type() === "error" || m.type() === "warn") console.log(`${m.type()}: ${m.text()}`);
});
page.on("pageerror", (e) => console.log(`page error: ${e.message}`));
const status = async () => ((await page.$("#status")) ? page.$eval("#status", (s) => s.textContent) : "");
await page.goto(url);
try {
  if (checking) await page.waitForSelector("body[data-check]", { timeout: 300_000 });
  if (profile) await page.waitForSelector("body[data-profile]", { timeout: 300_000 });
  await new Promise((resolve) => setTimeout(resolve, checking || profile ? 6000 : 20000));   // a site warms up while playing
  console.log(`status: ${(await status()).replaceAll("\n", " | ")}`);
} catch (error) {
  console.log(`no result: ${error.message}; status: ${await status()}`);
}
await browser.close();
server.close();
