import "./style.css";
import * as faceapi from "@vladmandic/face-api";
import * as ort from "onnxruntime-web";

// Load ONNX Runtime's WebAssembly files from a CDN matching the installed version
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const ctx = canvas.getContext("2d");

let session, meta;

// Small hidden canvas used to crop and resize the face for the model
const crop = document.createElement("canvas");
const cropCtx = crop.getContext("2d", { willReadFrequently: true });

// --- Smoothing: average the last N predictions so labels don't flicker ---
const SMOOTHING_FRAMES = 8;
const history = [];

function smooth(probs) {
  history.push(probs);
  if (history.length > SMOOTHING_FRAMES) history.shift();
  return probs.map(
    (_, i) => history.reduce((s, p) => s + p[i], 0) / history.length,
  );
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// --- Preprocessing: must match training exactly ---
// grayscale (3 channels) -> resize to SIZE x SIZE -> normalize with mean/std
function preprocess(box) {
  const SIZE = meta.size;
  const pad = 0.1; // include a little margin around the detected face
  const sx = Math.max(0, box.x - box.width * pad);
  const sy = Math.max(0, box.y - box.height * pad);
  const sw = Math.min(video.videoWidth - sx, box.width * (1 + 2 * pad));
  const sh = Math.min(video.videoHeight - sy, box.height * (1 + 2 * pad));

  crop.width = SIZE;
  crop.height = SIZE;
  cropCtx.imageSmoothingQuality = "high";
  cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, SIZE, SIZE);
  const { data } = cropCtx.getImageData(0, 0, SIZE, SIZE);

  const plane = SIZE * SIZE;
  const input = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4],
      g = data[i * 4 + 1],
      b = data[i * 4 + 2];
    const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    for (let c = 0; c < 3; c++) {
      input[c * plane + i] = (gray - meta.mean[c]) / meta.std[c];
    }
  }
  return new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
}

// --- Setup ---
async function loadModels() {
  await faceapi.tf.ready();
  await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
  meta = await (await fetch("/models/emotion/meta.json")).json();
  session = await ort.InferenceSession.create("/models/emotion/model.onnx", {
    executionProviders: ["wasm"],
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// --- Main loop ---
async function detectLoop() {
  const det = await faceapi.detectSingleFace(
    video,
    new faceapi.TinyFaceDetectorOptions({
      inputSize: 224,
      scoreThreshold: 0.5,
    }),
  );

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (det) {
    const { x, y, width, height } = det.box;
    const out = await session.run({ input: preprocess(det.box) });
    const probs = smooth(softmax(out.logits.data));

    let best = 0;
    probs.forEach((p, i) => {
      if (p > probs[best]) best = i;
    });
    const label = meta.classes[best];

    ctx.strokeStyle = "#00c853";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = "#00c853";
    ctx.font = "20px system-ui";
    ctx.fillText(
      `${label} (${Math.round(probs[best] * 100)}%)`,
      x,
      Math.max(y - 8, 20),
    );
  } else {
    history.length = 0;
  }

  requestAnimationFrame(detectLoop);
}

async function init() {
  try {
    await loadModels();
    statusEl.textContent = "Starting camera…";
    await startCamera();
    statusEl.textContent = "Running with your own model.";
    detectLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
}

init();
