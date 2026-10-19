// The "Start from an image" panel (index.html #shot): load a picture (file, drop or paste), drag its four
// corners onto the game screen, see it as the model will (screenshot.js), and play.
// setupEditor({ palette, base, play, normal }): base is the page's own start (its border bands are used);
// play(start, actions) starts a dream from the picture; normal() goes back to the page's start.
import { LINES, WIDTH, guessCorners, pictureImage, startPicture, toPicture, toStart } from "./screenshot.js";

const MAX_SIDE = 1600;                         // a bigger picture is scaled down first: corners stay easy to drag
const HANDLE = 14;                             // display pixels: how near a corner a press grabs it

export function setupEditor({ palette, base, play, normal }) {
  const $ = (id) => document.getElementById(id);
  const src = $("shot-src"), out = $("shot-out"), sctx = src.getContext("2d"), octx = out.getContext("2d");
  let image = null, bitmap = null, corners = null, picture = null, scale = 1, dragging = -1;
  const prefer = new Set(startPicture(base));      // the colours the game itself uses: twins settle on these
  const say = (text) => { $("shot-status").textContent = text; };

  async function load(blob) {
    try {
      const raw = await createImageBitmap(blob);
      const shrink = Math.min(1, MAX_SIDE / Math.max(raw.width, raw.height));
      const w = Math.round(raw.width * shrink), h = Math.round(raw.height * shrink);
      const canvas = new OffscreenCanvas(w, h), c = canvas.getContext("2d");
      c.drawImage(raw, 0, 0, w, h);
      bitmap = canvas;
      image = c.getImageData(0, 0, w, h);
      corners = guessCorners(image);
      scale = Math.min(512 / w, 384 / h, 1);
      src.width = Math.round(w * scale);
      src.height = Math.round(h * scale);
      convert();
      $("shot-play").disabled = false;
    } catch (error) {
      say(`Could not read that image: ${error.message}`);
    }
  }

  // The picture again from the corners and options.
  function convert() {
    if (!image) return;
    picture = toPicture(image, corners, palette, { visible: $("shot-overscan").checked ? LINES : 224,
                                                    levels: $("shot-levels").checked, prefer });
    drawSource();
    drawPicture();
    say("Drag the corners onto the game screen; the right shows what the model will start from.");
  }

  function drawSource() {
    sctx.drawImage(bitmap, 0, 0, src.width, src.height);
    sctx.strokeStyle = "#7cc4ff";
    sctx.lineWidth = 2;
    sctx.beginPath();
    corners.forEach(([x, y], i) => (i ? sctx.lineTo : sctx.moveTo).call(sctx, x * scale, y * scale));
    sctx.closePath();
    sctx.stroke();
    sctx.fillStyle = "#7cc4ff";
    for (const [x, y] of corners) sctx.fillRect(x * scale - 5, y * scale - 5, 10, 10);
  }

  function drawPicture() {
    const { data } = pictureImage(picture, palette);
    octx.putImageData(new ImageData(data, WIDTH, LINES), 0, 0);
  }

  const at = (canvas, event) => {
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) * canvas.width / box.width, (event.clientY - box.top) * canvas.height / box.height];
  };
  src.addEventListener("pointerdown", (event) => {
    if (!image) return;
    const [x, y] = at(src, event);
    const near = corners.map(([cx, cy], i) => [Math.hypot(cx * scale - x, cy * scale - y), i]).sort((a, b) => a[0] - b[0])[0];
    if (near[0] > HANDLE * 2) return;
    dragging = near[1];
    src.setPointerCapture(event.pointerId);
  });
  src.addEventListener("pointermove", (event) => {
    if (dragging < 0) return;
    const [x, y] = at(src, event);
    corners[dragging] = [Math.min(image.width, Math.max(0, x / scale)), Math.min(image.height, Math.max(0, y / scale))];
    drawSource();
  });
  src.addEventListener("pointerup", () => {
    if (dragging < 0) return;
    dragging = -1;
    convert();
  });

  $("shot-file").addEventListener("change", (event) => event.target.files[0] && load(event.target.files[0]));
  const panel = $("shot");
  panel.addEventListener("dragover", (event) => event.preventDefault());
  panel.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = [...event.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) load(file);
  });
  addEventListener("paste", (event) => {
    if (panel.hidden) return;
    const item = [...event.clipboardData.items].find((i) => i.type.startsWith("image/"));
    if (item) load(item.getAsFile());
  });
  $("shot-overscan").addEventListener("change", () => convert());
  $("shot-levels").addEventListener("change", () => convert());
  $("from-image").addEventListener("click", () => { panel.hidden = !panel.hidden; });
  $("shot-close").addEventListener("click", () => { panel.hidden = true; });
  $("shot-play").addEventListener("click", () => {
    if (!picture) return;
    const { start, actions } = toStart(picture, base);
    play(start, actions);
  });
  $("shot-normal").addEventListener("click", () => normal());
}
