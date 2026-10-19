// Start a dream from any picture: frame it (four corners), snap its colours to the NES's, and build the pixel
// model's start, as dreamer.js takes it ({ data, shape }: model frames of palette indices).
//
// A model frame (src/token_world/data/nes_layers.py model_frame) is the tick border's top band (BAND rows), NES
// lines 8-231 and the border's bottom band. A picture becomes that frame with the page's own start's border
// bands; the start is the frame three times, with no buttons. Any picture works: the model reads pixels, so a
// Tetris screenshot needs nothing marked (the layered model needed its falling piece as sprites).

export const LINES = 240, WIDTH = 256, SIZE = 256, BAND = 16, OVERSCAN = 8;
export const NES_COLOURS = 55, BACKDROP = 0;
export const FRAMES = 3;
const INNER = 0.6, EDGE = (1 - INNER) / 2;                              // where in a pixel its samples fall

// The quad's corners, clockwise from top left, around what differs from the image's edge colour (a
// screenshot's plain border, a letterbox): [[x, y] x 4].
export function guessCorners(image) {
  const { data, width, height } = image;
  const at = (x, y) => 4 * (y * width + x);
  const edge = [data[0], data[1], data[2]];
  const plain = (x0, y0, dx, dy, n) => {
    for (let i = 0; i < n; i++) {
      const p = at(x0 + dx * i, y0 + dy * i);
      if (Math.abs(data[p] - edge[0]) + Math.abs(data[p + 1] - edge[1]) + Math.abs(data[p + 2] - edge[2]) > 48)
        return false;
    }
    return true;
  };
  let top = 0, bottom = height - 1, left = 0, right = width - 1;
  while (top < bottom && plain(0, top, 1, 0, width)) top++;
  while (bottom > top && plain(0, bottom, 1, 0, width)) bottom--;
  while (left < right && plain(left, 0, 0, 1, height)) left++;
  while (right > left && plain(right, 0, 0, 1, height)) right--;
  if (right - left < 16 || bottom - top < 16) [left, top, right, bottom] = [0, 0, width - 1, height - 1];
  return [[left, top], [right + 1, top], [right + 1, bottom + 1], [left, bottom + 1]];
}

// The 3 x 3 homography taking the unit square's corners (0,0) (1,0) (1,1) (0,1) to the quad's.
function homography(quad) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = quad;
  const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
  const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
  const det = dx1 * dy2 - dx2 * dy1;
  const g = (sx * dy2 - dx2 * sy) / det, h = (dx1 * sy - sx * dy1) / det;
  return [x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h, 1];
}

// The image inside `corners`, as a frame's picture: [LINES * WIDTH] NES colour indices (0..54). visible: the
// lines the quad covers (224: an emulator's picture without the overscan, lines 8-231, the rest black; 240:
// all of them). levels: fit each colour channel's gain and offset to the palette (photos, faded or tinted
// captures; stretch). Each pixel takes the colour most of its samples x samples points
// snap to (spread over the middle INNER of the pixel), so a scaled or slightly blurred screenshot still gives
// crisp NES pixels.
export function toPicture(image, corners, palette, { visible = 224, levels = true, samples = 3, prefer = new Set() } = {}) {
  const { data, width, height } = image;
  const H = homography(corners);
  const top = (LINES - visible) >> 1;
  const points = new Uint8Array(visible * WIDTH * samples * samples * 3);
  let n = 0;
  for (let y = 0; y < visible; y++)
    for (let x = 0; x < WIDTH; x++)
      for (let j = 0; j < samples; j++)
        for (let i = 0; i < samples; i++) {
          // the middle INNER of the pixel: clear of its edges, which a non-integer scale shifts by half a pixel
          const u = (x + EDGE + INNER * (i + 0.5) / samples) / WIDTH, v = (y + EDGE + INNER * (j + 0.5) / samples) / visible;
          const w = H[6] * u + H[7] * v + H[8];
          const px = Math.min(width - 1, Math.max(0, Math.floor((H[0] * u + H[1] * v + H[2]) / w)));
          const py = Math.min(height - 1, Math.max(0, Math.floor((H[3] * u + H[4] * v + H[5]) / w)));
          const p = 4 * (py * width + px);
          points[n++] = data[p]; points[n++] = data[p + 1]; points[n++] = data[p + 2];
        }
  const map = levels ? stretch(points, palette) : (c, value) => value;
  const nearest = new Map();
  const snap = (r, g, b) => {
    const key = (r << 16) | (g << 8) | b;
    let best = nearest.get(key);
    if (best === undefined) {
      let d = Infinity;
      for (let k = 0; k < NES_COLOURS; k++) {
        const [pr, pg, pb] = palette[k];
        const e = 2 * (r - pr) ** 2 + 4 * (g - pg) ** 2 + 3 * (b - pb) ** 2;     // a little like the eye's weights
        if (e < d) [d, best] = [e, k];
      }
      nearest.set(key, best);
    }
    return best;
  };
  const picture = new Uint8Array(LINES * WIDTH).fill(BACKDROP);
  const votes = new Uint8Array(NES_COLOURS), per = samples * samples;
  for (let q = 0, s = 0; q < visible * WIDTH; q++) {
    votes.fill(0);
    let winner = 0;
    for (let k = 0; k < per; k++, s += 3) {
      const c = snap(map(0, points[s]), map(1, points[s + 1]), map(2, points[s + 2]));
      if (++votes[c] > votes[winner] || (votes[c] === votes[winner] && c === winner)) winner = c;
    }
    picture[(top + Math.floor(q / WIDTH)) * WIDTH + (q % WIDTH)] = winner;
  }
  return settleTwins(picture, palette, prefer);
}

// Some NES colours are near twins (greys 124 and 120, whites 252 and 248: NEAR or less apart); noise splits a
// pixel between them at random. Each twin group settles on one colour for the whole picture: the one most of
// its pixels took (a picture cannot meaningfully use two colours that look the same), or a colour in `prefer`
// (the colours of the page's own start: the twin the game itself uses) if it took at least PREFER of them.
const NEAR = 12, PREFER = 0.25;
export function settleTwins(picture, palette, prefer = new Set()) {
  const twin = Array.from({ length: NES_COLOURS }, (_, a) => a);
  for (let a = 0; a < NES_COLOURS; a++)
    for (let b = 0; b < a; b++)
      if (Math.hypot(...palette[a].map((v, i) => v - palette[b][i])) <= NEAR) twin[a] = twin[b];
  const counts = new Uint32Array(NES_COLOURS);
  for (const c of picture) counts[c]++;
  const choice = new Map(), group = new Map();
  for (let c = 0; c < NES_COLOURS; c++) {
    group.set(twin[c], (group.get(twin[c]) ?? 0) + counts[c]);
    if (!choice.has(twin[c]) || counts[c] > counts[choice.get(twin[c])]) choice.set(twin[c], c);
  }
  for (let c = 0; c < NES_COLOURS; c++)
    if (prefer.has(c) && counts[c] >= PREFER * group.get(twin[c]) && counts[c] > 0) choice.set(twin[c], c);
  return picture.map((c) => choice.get(twin[c]));
}

// Levels fitted to the palette: start from each channel's 0.5% and 99.5% points (to 0 and the palette's
// brightest), then a few rounds of snapping a subsample to its nearest colours and fitting each channel's gain and
// offset to them by least squares (noise averages out; percentiles alone sit a noise's width too far out).
function stretch(points, palette) {
  const colours = palette.slice(0, NES_COLOURS), white = Math.max(...colours.flat());
  const total = points.length / 3;
  let fit = [0, 1, 2].map((c) => {
    const counts = new Uint32Array(256);
    for (let i = c; i < points.length; i += 3) counts[points[i]]++;
    let lo = 0, hi = 255, seen = 0;
    while (lo < 255 && (seen += counts[lo]) < total * 0.005) lo++;
    seen = 0;
    while (hi > 0 && (seen += counts[hi]) < total * 0.005) hi--;
    const gain = hi > lo ? white / (hi - lo) : 1;
    return [gain, -lo * gain];
  });
  const every = Math.max(1, Math.floor(total / 30000));
  for (let round = 0; round < 4; round++) {
    const sums = [0, 1, 2].map(() => ({ n: 0, x: 0, t: 0, xx: 0, xt: 0 }));
    for (let i = 0; i < total; i += every) {
      const raw = [points[3 * i], points[3 * i + 1], points[3 * i + 2]];
      const v = raw.map((x, c) => fit[c][0] * x + fit[c][1]);
      let best = 0, d = Infinity;
      for (let k = 0; k < NES_COLOURS; k++) {
        const e = (v[0] - colours[k][0]) ** 2 + (v[1] - colours[k][1]) ** 2 + (v[2] - colours[k][2]) ** 2;
        if (e < d) [d, best] = [e, k];
      }
      for (let c = 0; c < 3; c++) {
        const s = sums[c], x = raw[c], t = colours[best][c];
        s.n++; s.x += x; s.t += t; s.xx += x * x; s.xt += x * t;
      }
    }
    fit = sums.map((s, c) => {
      const varx = s.xx / s.n - (s.x / s.n) ** 2;
      if (varx < 1) return fit[c];
      const gain = (s.xt / s.n - (s.x / s.n) * (s.t / s.n)) / varx;
      return [gain, s.t / s.n - gain * (s.x / s.n)];
    });
  }
  return (c, value) => Math.min(255, Math.max(0, Math.round(fit[c][0] * value + fit[c][1])));
}

// The picture as the model's start: FRAMES copies of one model frame, NES lines 8-231 from the picture and the
// border bands from base's last frame (the page's own start). -> { start, actions }.
export function toStart(picture, base) {
  const size = SIZE * SIZE, frame = Uint8Array.from(base.data.subarray(base.data.length - size));
  frame.set(picture.subarray(OVERSCAN * WIDTH, (LINES - OVERSCAN) * WIDTH), BAND * SIZE);
  const data = new Uint8Array(size * FRAMES);
  for (let f = 0; f < FRAMES; f++) data.set(frame, f * size);
  return { start: { data, shape: [FRAMES, SIZE, SIZE] }, actions: new Array(FRAMES).fill(0) };
}

// A start's last frame as a picture: [LINES * WIDTH] colour indices, NES lines 8-231 from the frame and the
// overscan lines BACKDROP, for the round-trip test and the editor's colours.
export function startPicture(start) {
  const size = SIZE * SIZE, frame = start.data.subarray(start.data.length - size);
  const picture = new Uint8Array(LINES * WIDTH).fill(BACKDROP);
  picture.set(frame.subarray(BAND * SIZE, (SIZE - BAND) * SIZE), OVERSCAN * WIDTH);
  return picture;
}

// A picture's RGB as an ImageData-like { data: RGBA, width, height }.
export function pictureImage(picture, palette) {
  const data = new Uint8ClampedArray(LINES * WIDTH * 4);
  for (let p = 0; p < picture.length; p++) {
    const [r, g, b] = palette[picture[p]];
    data.set([r, g, b, 255], p * 4);
  }
  return { data, width: WIDTH, height: LINES };
}
