// Chest X-ray: 14 findings with calibrated probabilities, run entirely in the browser.
// Model: DenseNet121 trained on NIH ChestX-ray14 (official patient-wise split), exported to ONNX.
import { CALIBRATION } from "./calibration.js";

const MODEL_URL = "chest_xray_densenet121.onnx";
const SIZE = 224;
const CAM_SIZE = 7;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const NAMES = CALIBRATION.labels.map((l) => (l.includes("_") ? l.replace("_", " ").replace(/^(\w)(.*)$/, (m, a, b) => a + b.toLowerCase()) : l));

const $ = (id) => document.getElementById(id);
const state = { session: null, gray: null, probs: null, cams: null, busy: false };

// ---------- maths (exported for tests) ----------
export function calibrate(raw) {
  return raw.map((p, k) => {
    const q = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    const z = CALIBRATION.a[k] * Math.log(q / (1 - q)) + CALIBRATION.b[k];
    return 1 / (1 + Math.exp(-z));
  });
}

export function toGray(rgba) {
  // ITU-R 601-2 luma, as in PIL's convert("L")
  const gray = new Uint8ClampedArray(rgba.length / 4);
  for (let i = 0; i < gray.length; i++) {
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    gray[i] = Math.round((r * 299 + g * 587 + b * 114) / 1000);
  }
  return gray;
}

// Bilinear resize with the same antialiasing as Pillow's Image.resize(..., BILINEAR), which the model
// was trained with. A canvas drawImage() downscale filters differently and shifts the probabilities.
export function resizeLikePillow(src, w, h, outW, outH) {
  const pass = (input, inLen, outLen, lines, stride, step, lineStride, outLineStride, outStep) => {
    const scale = inLen / outLen;
    const fscale = Math.max(scale, 1);
    const support = fscale;
    const out = new Uint8ClampedArray(outLen * lines);
    const taps = [];
    for (let i = 0; i < outLen; i++) {
      const center = (i + 0.5) * scale;
      const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
      const xmax = Math.min(Math.trunc(center + support + 0.5), inLen);
      const ws = [];
      let total = 0;
      for (let x = xmin; x < xmax; x++) {
        const t = Math.abs((x - center + 0.5) / fscale);
        const wgt = t < 1 ? 1 - t : 0;
        ws.push(wgt);
        total += wgt;
      }
      taps.push([xmin, ws.map((v) => (total ? v / total : 0))]);
    }
    for (let l = 0; l < lines; l++) {
      for (let i = 0; i < outLen; i++) {
        const [xmin, ws] = taps[i];
        let acc = 0;
        for (let k = 0; k < ws.length; k++) acc += ws[k] * input[l * lineStride + (xmin + k) * step];
        out[l * outLineStride + i * outStep] = Math.round(acc);
      }
    }
    return out;
  };
  // horizontal pass: h lines of w pixels -> h lines of outW
  const tmp = pass(src, w, outW, h, 0, 1, w, outW, 1);
  // vertical pass: outW columns of h pixels -> outW columns of outH
  return pass(tmp, h, outH, outW, 0, outW, 1, 1, outW);
}

export function toTensor(gray) {
  const n = SIZE * SIZE;
  const data = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) data[c * n + i] = (gray[i] / 255 - MEAN[c]) / STD[c];
  }
  return data;
}

export function upsampleCam(cam, out) {
  // Bilinear resize of a CAM_SIZE x CAM_SIZE map to out x out (align_corners = false), then ReLU and max-normalise.
  const res = new Float32Array(out * out);
  const scale = CAM_SIZE / out;
  for (let y = 0; y < out; y++) {
    let sy = Math.min(Math.max((y + 0.5) * scale - 0.5, 0), CAM_SIZE - 1);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, CAM_SIZE - 1), wy = sy - y0;
    for (let x = 0; x < out; x++) {
      let sx = Math.min(Math.max((x + 0.5) * scale - 0.5, 0), CAM_SIZE - 1);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, CAM_SIZE - 1), wx = sx - x0;
      const top = cam[y0 * CAM_SIZE + x0] * (1 - wx) + cam[y0 * CAM_SIZE + x1] * wx;
      const bot = cam[y1 * CAM_SIZE + x0] * (1 - wx) + cam[y1 * CAM_SIZE + x1] * wx;
      res[y * out + x] = Math.max(0, top * (1 - wy) + bot * wy);
    }
  }
  let max = 0;
  for (const v of res) max = Math.max(max, v);
  for (let i = 0; i < res.length; i++) res[i] /= max + 1e-8;
  return res;
}

export function jet(v) {
  const c = (t) => Math.min(Math.max(1.5 - Math.abs(4 * v - t), 0), 1);
  return [c(3), c(2), c(1)];
}

// ---------- model ----------
async function loadModel() {
  setStatus("Loading the model (about 28 MB, only the first time)…", "busy");
  ort.env.wasm.numThreads = 1;
  state.session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  setStatus("Model ready. Upload a frontal chest X-ray or try an example.", "ok");
  $("file").disabled = false;
  document.querySelectorAll(".example").forEach((b) => (b.disabled = false));
}

async function analyse(img) {
  // Read the image at full size (capped at 2048 px per side), convert to grayscale like Pillow's
  // convert("L"), then resize to 224x224 the way the training pipeline did.
  const cap = 2048 / Math.max(img.naturalWidth, img.naturalHeight, 2048);
  const w = Math.max(1, Math.round(img.naturalWidth * cap));
  const h = Math.max(1, Math.round(img.naturalHeight * cap));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  state.gray = resizeLikePillow(toGray(ctx.getImageData(0, 0, w, h).data), w, h, SIZE, SIZE);

  setStatus("Analysing…", "busy");
  const t0 = performance.now();
  const input = new ort.Tensor("float32", toTensor(state.gray), [1, 3, SIZE, SIZE]);
  const out = await state.session.run({ image: input });
  state.probs = calibrate(Array.from(out.probs.data));
  state.cams = Float32Array.from(out.cams.data);
  input.dispose?.();
  out.probs.dispose?.();
  out.cams.dispose?.();
  const ms = Math.round(performance.now() - t0);

  renderTable();
  const top = state.probs.indexOf(Math.max(...state.probs));
  $("finding").value = String(top);
  renderCam(top);
  $("results").hidden = false;
  $("placeholder").hidden = true;
  setStatus(`Done in ${ms} ms, on your device. The image was not uploaded anywhere.`, "ok");
}

// ---------- rendering ----------
function renderTable() {
  const order = state.probs.map((p, k) => k).sort((i, j) => state.probs[j] - state.probs[i]);
  const rows = order.map((k) => {
    const p = state.probs[k], prev = CALIBRATION.prevalence[k], ratio = p / prev;
    const level = ratio >= 3 ? "high" : ratio >= 1.5 ? "mid" : "low";
    return `<tr data-k="${k}"><td>${NAMES[k]}</td>
      <td class="num"><span class="bar"><span style="width:${Math.min(100, p * 100).toFixed(1)}%"></span></span>${(p * 100).toFixed(1)}%</td>
      <td class="num muted">${(prev * 100).toFixed(1)}%</td>
      <td class="num"><span class="pill ${level}">${ratio.toFixed(1)}×</span></td></tr>`;
  });
  $("table-body").innerHTML = rows.join("");
  document.querySelectorAll("#table-body tr").forEach((tr) =>
    tr.addEventListener("click", () => {
      $("finding").value = tr.dataset.k;
      renderCam(Number(tr.dataset.k));
    }));
}

function renderCam(k) {
  const out = 448;
  const canvas = $("cam");
  canvas.width = canvas.height = out;
  const ctx = canvas.getContext("2d");
  const base = document.createElement("canvas");
  base.width = base.height = SIZE;
  const bctx = base.getContext("2d");
  const img = bctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = state.gray[i];
    img.data[4 * i + 3] = 255;
  }
  bctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(base, 0, 0, out, out);

  const cam = state.cams.slice(k * CAM_SIZE * CAM_SIZE, (k + 1) * CAM_SIZE * CAM_SIZE);
  const heat = upsampleCam(cam, out);
  const frame = ctx.getImageData(0, 0, out, out);
  for (let i = 0; i < out * out; i++) {
    const [r, g, b] = jet(heat[i]);
    frame.data[4 * i] = 0.65 * frame.data[4 * i] + 0.35 * 255 * r;
    frame.data[4 * i + 1] = 0.65 * frame.data[4 * i + 1] + 0.35 * 255 * g;
    frame.data[4 * i + 2] = 0.65 * frame.data[4 * i + 2] + 0.35 * 255 * b;
  }
  ctx.putImageData(frame, 0, 0);
  $("cam-caption").textContent =
    `Heatmap for ${NAMES[k]} (calibrated probability ${(state.probs[k] * 100).toFixed(1)}%). ` +
    "Warmer colours mark the regions that pushed this score up most. It is a hint about what the model used, not the location of disease.";
  document.querySelectorAll("#table-body tr").forEach((tr) => tr.classList.toggle("sel", Number(tr.dataset.k) === k));
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + kind;
}

async function loadImage(src) {
  // Example files are served from Hugging Face's file storage on another domain. Fetching them
  // into a local blob keeps the canvas same-origin, so the pixels can be read for the model.
  let url = src;
  if (!src.startsWith("blob:") && !src.startsWith("data:")) {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Could not load ${src} (HTTP ${res.status}).`);
    url = URL.createObjectURL(await res.blob());
  }
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("This file could not be read as an image."));
    img.src = url;
  });
}

function setBusy(busy) {
  state.busy = busy;
  $("file").disabled = busy;
  document.querySelectorAll(".example").forEach((b) => (b.disabled = busy));
}

async function handle(src, preview, nihLabel) {
  if (state.busy || !state.session) return; // one analysis at a time
  setBusy(true);
  const note = $("example-label");
  note.hidden = !nihLabel;
  if (nihLabel) {
    note.textContent = `NIH label for this test image: ${nihLabel}. These labels were mined from radiology report text and are sometimes wrong, so the model can disagree with them.`;
  }
  try {
    const img = await loadImage(src);
    $("preview").src = preview || src;
    $("preview").hidden = false;
    $("results").hidden = true;
    await analyse(img);
  } catch (err) {
    console.error(err);
    setStatus("Analysis failed: " + (err && err.message ? err.message : String(err)), "error");
  } finally {
    setBusy(false);
  }
}

// ---------- wiring ----------
function init() {
  $("finding").innerHTML = NAMES.map((n, k) => `<option value="${k}">${n}</option>`).join("");
  $("finding").addEventListener("change", (e) => state.probs && renderCam(Number(e.target.value)));
  $("file").addEventListener("change", (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) handle(URL.createObjectURL(f));
  });
  const drop = $("drop");
  ["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && state.session) handle(URL.createObjectURL(f));
  });
  document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => handle(b.dataset.src, null, b.dataset.label)));
  loadModel().catch((err) => setStatus("The model could not be loaded: " + (err.message || err), "error"));
}

if (typeof window !== "undefined" && document.getElementById("app")) init();
