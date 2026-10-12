import "./style.css";
import * as faceapi from "@vladmandic/face-api";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const ctx = canvas.getContext("2d");

// --- Smoothing: average the last N predictions so labels don't flicker ---
const SMOOTHING_FRAMES = 8;
const history = [];

function smooth(probs) {
  history.push(probs);
  if (history.length > SMOOTHING_FRAMES) history.shift();
  const avg = {};
  for (const key of Object.keys(probs)) {
    avg[key] = history.reduce((sum, p) => sum + p[key], 0) / history.length;
  }
  return avg;
}

// --- Setup ---
async function loadModels() {
  await faceapi.tf.ready();
  await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
  await faceapi.nets.faceExpressionNet.loadFromUri("/models");
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
  const result = await faceapi
    .detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 224,
        scoreThreshold: 0.5,
      }),
    )
    .withFaceExpressions();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (result) {
    const probs = smooth({ ...result.expressions });
    const [label, score] = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
    const { x, y, width, height } = result.detection.box;

    ctx.strokeStyle = "#00c853";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, width, height);

    ctx.fillStyle = "#00c853";
    ctx.font = "20px system-ui";
    ctx.fillText(
      `${label} (${Math.round(score * 100)}%)`,
      x,
      Math.max(y - 8, 20),
    );
  } else {
    history.length = 0; // no face: reset smoothing
  }

  requestAnimationFrame(detectLoop);
}

async function init() {
  try {
    await loadModels();
    statusEl.textContent = "Starting camera…";
    await startCamera();
    statusEl.textContent = "Running. Try different expressions.";
    detectLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Error: " + err.message;
  }
}

init();
