// screenshot.js against the page's own start: its last frame drawn as an image and converted back must give
// the same picture and build the same start frames, at the screen's size and scaled up (a big screenshot), and
// from the 224 lines an emulator shows.
//
//     cd web && node test_screenshot.mjs
import { readFile } from "node:fs/promises";
import { startFrames } from "./dreamer.js";
import { LINES, WIDTH, guessCorners, pictureImage, startPicture, toPicture, toStart } from "./screenshot.js";

const load = async (name) => new Uint8Array(await readFile(new URL(`model/${name}`, import.meta.url)));
const meta = JSON.parse(new TextDecoder().decode(await load("context.json")));
const start = startFrames(meta, await load("context.bin"));
const palette = meta.palette;
let failed = 0;
const check = (what, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}${detail ? `: ${detail}` : ""}`);
  failed += !ok;
};
const differ = (a, b, from = 0, to = a.length) => {
  let n = 0;
  for (let i = from; i < to; i++) n += a[i] !== b[i];
  return n;
};

const real = startPicture(start);
const image = pictureImage(real, palette);
const full = [[0, 0], [WIDTH, 0], [WIDTH, LINES], [0, LINES]];

// the screen's own size, with and without the levels stretch
for (const levels of [false, true]) {
  const picture = toPicture(image, full, palette, { visible: LINES, levels });
  check(`picture back at 256 x 240${levels ? ", levels" : ""}`, differ(picture, real) === 0,
        `${differ(picture, real)} pixels differ`);
}

// scaled up 3x, as a big screenshot, with the corners found by guessCorners on a black letterbox
const S = 3, pad = 20, W = WIDTH * S + 2 * pad, H = LINES * S + 2 * pad;
const big = { data: new Uint8ClampedArray(W * H * 4), width: W, height: H };
for (let i = 3; i < big.data.length; i += 4) big.data[i] = 255;
const grey = [40, 40, 40];
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) big.data.set(grey, 4 * (y * W + x));
for (let y = 0; y < LINES * S; y++)
  for (let x = 0; x < WIDTH * S; x++)
    big.data.set(image.data.subarray(4 * (Math.floor(y / S) * WIDTH + Math.floor(x / S)), 4 * (Math.floor(y / S) * WIDTH + Math.floor(x / S)) + 3),
                 4 * ((y + pad) * W + x + pad));
const corners = guessCorners(big);
check("corners found around the letterbox", JSON.stringify(corners) ===
      JSON.stringify([[pad, pad], [pad + WIDTH * S, pad], [pad + WIDTH * S, pad + LINES * S], [pad, pad + LINES * S]]),
      JSON.stringify(corners));
const scaled = toPicture(big, corners, palette, { visible: LINES });
check("picture back from 3x with a letterbox", differ(scaled, real) === 0, `${differ(scaled, real)} pixels differ`);

// the 224 lines an emulator shows (overscan cropped)
const crop = { data: image.data.slice(4 * 8 * WIDTH, 4 * 232 * WIDTH), width: WIDTH, height: 224 };
const cropped = toPicture(crop, [[0, 0], [WIDTH, 0], [WIDTH, 224], [0, 224]], palette, { visible: 224 });
check("picture back from the 224 visible lines", differ(cropped, real, 8 * WIDTH, 232 * WIDTH) === 0,
      `${differ(cropped, real, 8 * WIDTH, 232 * WIDTH)} pixels differ in lines 8-231`);

// the start: the picture back as the start's last frame, border bands and all, three times
const built = toStart(scaled, start).start, size = 256 * 256;
const lastFrame = (x) => x.data.subarray(x.data.length - size);
for (let f = 0; f < built.shape[0]; f++) {
  const n = differ(built.data.subarray(f * size, (f + 1) * size), lastFrame(start));
  check(`start frame ${f} as the real start's last`, n === 0, `${n} pixels differ`);
}
process.exit(failed ? 1 : 0);
