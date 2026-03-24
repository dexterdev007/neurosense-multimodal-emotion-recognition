const NUMERIC_MODALITIES = {
  eeg: {
    label: "EEG",
    fields: [
      { key: "fp1", label: "Fp1 Activity", min: -40, max: 40, step: 0.1 },
      { key: "fp2", label: "Fp2 Activity", min: -40, max: 40, step: 0.1 },
      { key: "alpha", label: "Alpha Power", min: 0, max: 1, step: 0.01 },
      { key: "beta", label: "Beta Power", min: 0, max: 1, step: 0.01 },
      { key: "asymmetry", label: "Asymmetry", min: -1, max: 1, step: 0.01 },
    ],
    examples: [
      { fp1: 12.4, fp2: 13.1, alpha: 0.82, beta: 0.24, asymmetry: 0.36 },
      { fp1: -8.2, fp2: -12.8, alpha: 0.18, beta: 0.78, asymmetry: -0.44 },
      { fp1: 2.1, fp2: 2.4, alpha: 0.48, beta: 0.41, asymmetry: 0.03 },
    ],
    valenceWeights: { fp1: 0.18, fp2: 0.2, alpha: 0.28, beta: -0.12, asymmetry: 0.22 },
    arousalWeights: { fp1: -0.08, fp2: -0.06, alpha: -0.14, beta: 0.42, asymmetry: 0.08 },
  },
  meg: {
    label: "MEG",
    fields: [
      { key: "frontalFlux", label: "Frontal Flux", min: -30, max: 30, step: 0.1 },
      { key: "temporalSync", label: "Temporal Sync", min: 0, max: 1, step: 0.01 },
      { key: "gammaBurst", label: "Gamma Burst", min: 0, max: 1, step: 0.01 },
      { key: "coherence", label: "Coherence", min: 0, max: 1, step: 0.01 },
      { key: "leftBias", label: "Left Bias", min: -1, max: 1, step: 0.01 },
    ],
    examples: [
      { frontalFlux: 10.8, temporalSync: 0.74, gammaBurst: 0.61, coherence: 0.8, leftBias: 0.42 },
      { frontalFlux: -11.6, temporalSync: 0.49, gammaBurst: 0.86, coherence: 0.57, leftBias: -0.38 },
      { frontalFlux: 1.4, temporalSync: 0.42, gammaBurst: 0.3, coherence: 0.46, leftBias: 0.04 },
    ],
    valenceWeights: { frontalFlux: 0.24, temporalSync: 0.16, gammaBurst: -0.08, coherence: 0.18, leftBias: 0.24 },
    arousalWeights: { frontalFlux: 0.08, temporalSync: 0.14, gammaBurst: 0.36, coherence: 0.18, leftBias: -0.06 },
  },
};

const FILE_MODALITIES = {
  mri: {
    label: "MRI",
    kind: "image",
    previewId: "mri-preview",
    metaId: "mri-meta",
    inputId: "mri-file",
    randomId: "mri-random",
    clearId: "mri-clear",
    examples: [
      { note: "Balanced resting MRI descriptor", stats: { brightness: 120, contrast: 24, symmetry: 0.78, density: 0.32, variation: 0.28 } },
      { note: "High-intensity stress-like MRI descriptor", stats: { brightness: 178, contrast: 58, symmetry: 0.36, density: 0.72, variation: 0.64 } },
      { note: "Low-energy calm MRI descriptor", stats: { brightness: 108, contrast: 18, symmetry: 0.84, density: 0.26, variation: 0.22 } },
    ],
    fields: [
      { key: "brightness", min: 0, max: 255 },
      { key: "contrast", min: 0, max: 128 },
      { key: "symmetry", min: 0, max: 1 },
      { key: "density", min: 0, max: 1 },
      { key: "variation", min: 0, max: 1 },
    ],
    valenceWeights: { brightness: -0.06, contrast: -0.14, symmetry: 0.32, density: -0.12, variation: -0.08 },
    arousalWeights: { brightness: 0.08, contrast: 0.24, symmetry: -0.08, density: 0.24, variation: 0.2 },
  },
  speech: {
    label: "Speech",
    kind: "audio",
    previewId: "speech-preview",
    metaId: "speech-meta",
    inputId: "speech-file",
    randomId: "speech-random",
    clearId: "speech-clear",
    playerId: "speech-player",
    examples: [
      { note: "Warm positive vocal sample", stats: { pitch: 0.78, energy: 0.72, pace: 0.58, brightness: 0.76, dynamics: 0.52 } },
      { note: "Tense high-arousal vocal sample", stats: { pitch: 0.88, energy: 0.84, pace: 0.74, brightness: 0.32, dynamics: 0.76 } },
      { note: "Slow subdued vocal sample", stats: { pitch: 0.34, energy: 0.24, pace: 0.28, brightness: 0.26, dynamics: 0.22 } },
    ],
    fields: [
      { key: "pitch", min: 0, max: 1 },
      { key: "energy", min: 0, max: 1 },
      { key: "pace", min: 0, max: 1 },
      { key: "brightness", min: 0, max: 1 },
      { key: "dynamics", min: 0, max: 1 },
    ],
    valenceWeights: { pitch: 0.12, energy: 0.18, pace: 0.06, brightness: 0.22, dynamics: -0.08 },
    arousalWeights: { pitch: 0.18, energy: 0.28, pace: 0.26, brightness: 0.04, dynamics: 0.16 },
  },
  face: {
    label: "Face",
    kind: "image",
    previewId: "face-preview",
    metaId: "face-meta",
    inputId: "face-file",
    randomId: "face-random",
    clearId: "face-clear",
    examples: [
      { note: "Open positive face descriptor", stats: { brightness: 176, contrast: 42, symmetry: 0.82, density: 0.44, variation: 0.34 } },
      { note: "Angry face descriptor", stats: { brightness: 108, contrast: 68, symmetry: 0.48, density: 0.72, variation: 0.66 } },
      { note: "Sad low-energy face descriptor", stats: { brightness: 96, contrast: 34, symmetry: 0.74, density: 0.24, variation: 0.22 } },
    ],
    fields: [
      { key: "brightness", min: 0, max: 255 },
      { key: "contrast", min: 0, max: 128 },
      { key: "symmetry", min: 0, max: 1 },
      { key: "density", min: 0, max: 1 },
      { key: "variation", min: 0, max: 1 },
    ],
    valenceWeights: { brightness: 0.12, contrast: -0.12, symmetry: 0.22, density: 0.08, variation: -0.14 },
    arousalWeights: { brightness: 0.04, contrast: 0.2, symmetry: -0.06, density: 0.26, variation: 0.24 },
  },
};

const EMOTION_META = {
  Happy: { emoji: "😊", color: "#f5b041" },
  Calm: { emoji: "😌", color: "#5dade2" },
  Angry: { emoji: "😠", color: "#ec7063" },
  Sad: { emoji: "😢", color: "#7f8c8d" },
  Surprised: { emoji: "😲", color: "#af7ac5" },
  Neutral: { emoji: "🧠", color: "#7f7bff" },
};

const numericInputRefs = {};
const fileState = {
  mri: { file: null, sample: null, stats: null, previewUrl: null },
  speech: { file: null, sample: null, stats: null, previewUrl: null },
  face: { file: null, sample: null, stats: null, previewUrl: null },
};

const themeToggle = document.getElementById("theme-toggle");
const predictAllBtn = document.getElementById("predict-all");
const clearAllBtn = document.getElementById("clear-all");
const emotionBadge = document.getElementById("emotion-badge");
const emotionLabel = document.getElementById("emotion-label");
const emotionSummary = document.getElementById("emotion-summary");
const confidenceValue = document.getElementById("confidence-value");
const valenceValue = document.getElementById("valence-value");
const arousalValue = document.getElementById("arousal-value");
const confidenceBar = document.getElementById("confidence-bar");
const valenceBar = document.getElementById("valence-bar");
const arousalBar = document.getElementById("arousal-bar");
const contributionBody = document.getElementById("contribution-body");
const bgWave = document.getElementById("bg-wave");

// Webcam Elements (Recommendation 5)
const faceWebcamBtn = document.getElementById("face-webcam");
const webcamContainer = document.getElementById("webcam-container");
const webcamVideo = document.getElementById("webcam-video");
const webcamCanvas = document.getElementById("webcam-canvas");
const faceCaptureBtn = document.getElementById("face-capture");
let webcamStream = null;

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("multimodal-theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("multimodal-theme");
  if (saved) {
    setTheme(saved);
    return;
  }
  setTheme("light");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * ratio;
  canvas.height = canvas.clientHeight * ratio;
}

function drawWave(ctx, time, intensity, colorStops) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  colorStops.forEach(([stop, color]) => gradient.addColorStop(stop, color));
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();
  for (let x = 0; x <= width; x += 8) {
    const y =
      height / 2 +
      Math.sin(x * 0.012 + time) * (height * 0.18 * intensity) +
      Math.sin(x * 0.02 + time * 1.35) * (height * 0.08);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animateBackground() {
  if (!bgWave) return;
  const ctx = bgWave.getContext("2d");
  const frame = (time) => {
    const t = time * 0.001;
    drawWave(ctx, t, 0.9, [
      [0, "rgba(127, 123, 255, 0.18)"],
      [0.5, "rgba(76, 201, 240, 0.28)"],
      [1, "rgba(94, 234, 212, 0.18)"],
    ]);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function initCanvas() {
  if (!bgWave) return;
  resizeCanvas(bgWave);
  window.addEventListener("resize", () => resizeCanvas(bgWave));
}

function renderNumericFields() {
  Object.entries(NUMERIC_MODALITIES).forEach(([key, modality]) => {
    const container = document.getElementById(`${key}-fields`);
    numericInputRefs[key] = [];
    container.innerHTML = "";
    modality.fields.forEach((field) => {
      const wrapper = document.createElement("div");
      wrapper.className = "input-field";
      wrapper.innerHTML = `
        <label>
          <span class="field-name">${field.label}</span>
          <span class="field-range">${field.min} to ${field.max}</span>
        </label>
        <input
          type="number"
          min="${field.min}"
          max="${field.max}"
          step="${field.step}"
          placeholder="Enter ${field.label.toLowerCase()}"
        />
      `;
      const input = wrapper.querySelector("input");
      numericInputRefs[key].push({ field, input });
      container.appendChild(wrapper);
    });
  });
}

function normalizeField(field, value) {
  return (value - field.min) / (field.max - field.min || 1);
}

function scoreAxis(weights, values, fields) {
  if (!fields.length) return 0.5;
  let sum = 0;
  let weightSum = 0;
  fields.forEach((field) => {
    const weight = weights[field.key] || 0;
    const centered = normalizeField(field, values[field.key]) - 0.5;
    sum += centered * weight;
    weightSum += Math.abs(weight);
  });
  if (!weightSum) return 0.5;
  return clamp(0.5 + sum / weightSum, 0, 1);
}

function mapEmotion(valence, arousal) {
  if (valence >= 0.65 && arousal >= 0.58) return "Happy";
  if (valence >= 0.62 && arousal < 0.48) return "Calm";
  if (valence < 0.42 && arousal >= 0.58) return "Angry";
  if (valence < 0.42 && arousal < 0.48) return "Sad";
  if (arousal >= 0.72) return "Surprised";
  return "Neutral";
}

function scoreModality(modality, stats, fields) {
  const valence = scoreAxis(modality.valenceWeights, stats, fields);
  const arousal = scoreAxis(modality.arousalWeights, stats, fields);
  const label = mapEmotion(valence, arousal);
  const confidence = clamp(0.58 + Math.abs(valence - 0.5) * 0.3 + Math.abs(arousal - 0.5) * 0.3, 0.55, 0.97);
  return { label, valence, arousal, confidence };
}

function collectNumericModality(key) {
  const modality = NUMERIC_MODALITIES[key];
  const refs = numericInputRefs[key];
  const values = {};
  const activeFields = [];
  refs.forEach(({ field, input }) => {
    const raw = input.value.trim();
    if (raw === "") return;
    const numeric = Number(raw);
    if (!Number.isFinite(numeric)) return;
    values[field.key] = numeric;
    activeFields.push(field);
  });
  if (!activeFields.length) return null;
  return {
    modality: modality.label,
    ...scoreModality(modality, values, activeFields),
  };
}

function formatStats(stats) {
  return Object.entries(stats)
    .map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(2) : value}`)
    .join(" • ");
}

function setImagePreview(previewId, url) {
  const preview = document.getElementById(previewId);
  preview.innerHTML = url ? `<img src="${url}" alt="preview" />` : `<span class="muted">No input selected.</span>`;
}

function setMeta(metaId, text, visible = true) {
  const box = document.getElementById(metaId);
  if (!visible) {
    box.hidden = true;
    box.textContent = "";
    return;
  }
  box.hidden = false;
  box.textContent = text;
}

function clearFileState(key) {
  const config = FILE_MODALITIES[key];
  const state = fileState[key];
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }
  fileState[key] = { file: null, sample: null, stats: null, previewUrl: null };
  const input = document.getElementById(config.inputId);
  input.value = "";
  if (config.kind === "audio") {
    const player = document.getElementById(config.playerId);
    player.hidden = true;
    player.removeAttribute("src");
    player.load();
    document.getElementById(config.previewId).textContent = "No speech input selected.";
  } else {
    document.getElementById(config.previewId).innerHTML = `<span class="muted">No ${key} input selected.</span>`;
  }
  setMeta(config.metaId, "", false);
}

function randomExampleForFile(key) {
  const config = FILE_MODALITIES[key];
  const sample = config.examples[Math.floor(Math.random() * config.examples.length)];
  clearFileState(key);
  fileState[key].sample = sample;
  fileState[key].stats = sample.stats;
  if (config.kind === "audio") {
    document.getElementById(config.previewId).textContent = `Simulated speech sample loaded.`;
  } else {
    document.getElementById(config.previewId).innerHTML = `<div class="muted">Simulated ${config.label} sample loaded.</div>`;
  }
  setMeta(config.metaId, `${sample.note} • ${formatStats(sample.stats)}`);
}

function randomExampleForNumeric(key) {
  const refs = numericInputRefs[key];
  const example = NUMERIC_MODALITIES[key].examples[Math.floor(Math.random() * NUMERIC_MODALITIES[key].examples.length)];
  refs.forEach(({ field, input }) => {
    input.value = example[field.key];
  });
}

function genericImageStats(imageData) {
  const { data, width, height } = imageData;
  let sum = 0;
  let sq = 0;
  let left = 0;
  let right = 0;
  let count = 0;
  let diffSum = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      sum += gray;
      sq += gray * gray;
      if (x < width / 2) left += gray;
      else right += gray;
      if (x > 0) {
        const prevIdx = (y * width + (x - 1)) * 4;
        const prevGray = 0.299 * data[prevIdx] + 0.587 * data[prevIdx + 1] + 0.114 * data[prevIdx + 2];
        diffSum += Math.abs(gray - prevGray);
      }
      count += 1;
    }
  }
  const mean = sum / count;
  const variance = sq / count - mean * mean;
  const contrast = Math.sqrt(Math.max(variance, 0));
  const symmetry = 1 - clamp(Math.abs(left - right) / Math.max(left + right, 1), 0, 1);
  const density = clamp(diffSum / count / 64, 0, 1);
  const variation = clamp(contrast / 64, 0, 1);
  return { mean, contrast, symmetry, density, variation };
}

async function analyzeImageFile(file) {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = 160;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, 160, 160);
  const stats = genericImageStats(ctx.getImageData(0, 0, 160, 160));
  return {
    previewUrl: url,
    stats: {
      brightness: stats.mean,
      contrast: stats.contrast,
      symmetry: stats.symmetry,
      density: stats.density,
      variation: stats.variation,
    },
  };
}

async function analyzeAudioFile(file) {
  const seed = hashString(`${file.name}-${file.size}`);
  const fallback = {
    pitch: ((seed % 60) + 20) / 100,
    energy: ((seed % 55) + 25) / 100,
    pace: ((seed % 50) + 20) / 100,
    brightness: ((seed % 70) + 10) / 100,
    dynamics: ((seed % 65) + 15) / 100,
  };
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) return fallback;
  let ctx;
  try {
    ctx = new AudioContextClass();
    const buffer = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(buffer.slice(0));
    const channel = audioBuffer.getChannelData(0);
    const step = Math.max(1, Math.floor(channel.length / 6000));
    let sq = 0;
    let crossings = 0;
    let prev = channel[0] || 0;
    let ampSum = 0;
    let count = 0;
    let max = -1;
    let min = 1;
    for (let i = 0; i < channel.length; i += step) {
      const value = channel[i];
      sq += value * value;
      ampSum += Math.abs(value);
      if ((value >= 0) !== (prev >= 0)) crossings += 1;
      prev = value;
      max = Math.max(max, value);
      min = Math.min(min, value);
      count += 1;
    }
    const rms = Math.sqrt(sq / Math.max(count, 1));
    const pace = clamp((crossings / Math.max(count, 1)) * 5, 0, 1);
    const brightness = clamp((ampSum / Math.max(count, 1)) * 2.4, 0, 1);
    const pitch = clamp((audioBuffer.sampleRate / 48000) * 0.4 + pace * 0.6, 0, 1);
    const dynamics = clamp((max - min) / 2, 0, 1);
    return {
      pitch,
      energy: clamp(rms * 4, 0, 1),
      pace,
      brightness,
      dynamics,
    };
  } catch {
    return fallback;
  } finally {
    if (ctx) {
      ctx.close().catch(() => {});
    }
  }
}

async function handleFileChange(key) {
  const config = FILE_MODALITIES[key];
  const input = document.getElementById(config.inputId);
  const file = input.files?.[0];
  if (!file) {
    clearFileState(key);
    return;
  }

  clearFileState(key);
  fileState[key].file = file;

  if (config.kind === "audio") {
    const player = document.getElementById(config.playerId);
    const previewUrl = URL.createObjectURL(file);
    fileState[key].previewUrl = previewUrl;
    player.src = previewUrl;
    player.hidden = false;
    document.getElementById(config.previewId).textContent = file.name;
    const stats = await analyzeAudioFile(file);
    fileState[key].stats = stats;
    setMeta(config.metaId, `Uploaded audio • ${formatStats(stats)}`);
    return;
  }

  const result = await analyzeImageFile(file);
  fileState[key].previewUrl = result.previewUrl;
  fileState[key].stats = result.stats;
  setImagePreview(config.previewId, result.previewUrl);
  setMeta(config.metaId, `Uploaded image • ${formatStats(result.stats)}`);
}

function collectFileModality(key) {
  const config = FILE_MODALITIES[key];
  const state = fileState[key];
  if (!state.stats) return null;
  return {
    modality: config.label,
    ...scoreModality(config, state.stats, config.fields),
  };
}

function resetResult() {
  emotionBadge.textContent = "--";
  emotionBadge.style.background = "linear-gradient(135deg, #7f7bff, #4cc9f0)";
  emotionLabel.textContent = "Awaiting prediction";
  emotionSummary.textContent = "Fill at least one modality and run the combined prediction.";
  confidenceValue.textContent = "0%";
  valenceValue.textContent = "0.00";
  arousalValue.textContent = "0.00";
  confidenceBar.style.width = "0%";
  valenceBar.style.width = "0%";
  arousalBar.style.width = "0%";
  contributionBody.innerHTML = `<tr><td colspan="4" class="muted center-cell">No modalities processed yet.</td></tr>`;
}

function renderResult(combined, contributions) {
  const meta = EMOTION_META[combined.label] || EMOTION_META.Neutral;
  emotionBadge.textContent = meta.emoji;
  emotionBadge.style.background = `linear-gradient(135deg, ${meta.color}, rgba(255,255,255,0.35))`;
  emotionLabel.textContent = combined.label;
  emotionSummary.textContent = `Predicted from ${contributions.length} active modalit${contributions.length === 1 ? "y" : "ies"}: ${contributions.map((item) => item.modality).join(", ")}.`;
  confidenceValue.textContent = `${Math.round(combined.confidence * 100)}%`;
  valenceValue.textContent = combined.valence.toFixed(2);
  arousalValue.textContent = combined.arousal.toFixed(2);
  confidenceBar.style.width = `${Math.round(combined.confidence * 100)}%`;
  valenceBar.style.width = `${Math.round(combined.valence * 100)}%`;
  arousalBar.style.width = `${Math.round(combined.arousal * 100)}%`;

  contributionBody.innerHTML = contributions
    .map(
      (item) => `
        <tr>
          <td>${item.modality}</td>
          <td>${item.label}</td>
          <td>${Math.round(item.confidence * 100)}%</td>
          <td>Used</td>
        </tr>
      `,
    )
    .join("");
}

function clearNumeric(key) {
  numericInputRefs[key].forEach(({ input }) => {
    input.value = "";
  });
}

function clearAll() {
  Object.keys(NUMERIC_MODALITIES).forEach(clearNumeric);
  Object.keys(FILE_MODALITIES).forEach(clearFileState);
  resetResult();
}

function buildSimulatedNumericResult(key) {
  const scored = collectNumericModality(key);
  if (!scored) return null;
  return {
    modality: NUMERIC_MODALITIES[key].label,
    prediction: scored.label,
    confidence: Number(scored.confidence.toFixed(4)),
    probabilities: {
      [scored.label]: Number(scored.confidence.toFixed(4)),
    },
    source: "simulated",
  };
}

function collectFilePayload(key) {
  const state = fileState[key];
  return state.file || null;
}

function buildSimulatedFileResult(key) {
  const scored = collectFileModality(key);
  if (!scored) return null;
  return {
    modality: FILE_MODALITIES[key].label,
    prediction: scored.label,
    confidence: Number(scored.confidence.toFixed(4)),
    probabilities: {
      [scored.label]: Number(scored.confidence.toFixed(4)),
    },
    source: "simulated",
  };
}

async function parsePredictionResponse(response, key) {
  if (response.ok) {
    return response.json();
  }

  let message = `${key} failed`;
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string" && payload.detail.trim()) {
      message = payload.detail;
    }
  } catch {
    // Ignore response parsing errors and fall back to the generic message.
  }

  throw new Error(message);
}

async function predictCombined() {
  const promises = [];
  predictAllBtn.disabled = true;
  predictAllBtn.classList.add("loading");

  Object.keys(NUMERIC_MODALITIES).forEach((key) => {
    const simulated = buildSimulatedNumericResult(key);
    if (simulated) {
      promises.push(Promise.resolve(simulated));
    }
  });

  Object.keys(FILE_MODALITIES).forEach((key) => {
    const state = fileState[key];
    const file = collectFilePayload(key);
    if (file) {
      const fd = new FormData();
      fd.append("file", file);
      promises.push(
        fetch(`/api/${key}/predict`, { method: "POST", body: fd })
          .then((res) => parsePredictionResponse(res, key))
          .then((data) => ({ modality: FILE_MODALITIES[key].label, ...data }))
      );
      return;
    }

    if (state.sample) {
      const simulated = buildSimulatedFileResult(key);
      if (simulated) {
        promises.push(Promise.resolve(simulated));
      }
    }
  });

  if (!promises.length) {
    window.alert("Provide at least one modality input before predicting.");
    predictAllBtn.disabled = false;
    predictAllBtn.classList.remove("loading");
    return;
  }

  try {
    const results = await Promise.all(promises);
    const fusionPayload = { modalities: {} };
    results.forEach((res) => {
      const modKey = res.modality.toLowerCase();
      fusionPayload.modalities[modKey] = {
        prediction: res.prediction || "NEUTRAL",
        confidence: res.confidence || 0.5,
        probabilities: res.probabilities || {},
      };
    });

    const fres = await fetch("/api/fusion/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fusionPayload),
    });
    const fusionData = await parsePredictionResponse(fres, "fusion");

    let emotionLabel = (fusionData.prediction || fusionData.emotion || fusionData.predicted_emotion || fusionData.final_emotion || "NEUTRAL").toUpperCase();
    const emLower = emotionLabel.toLowerCase();
    if (emLower.includes("happy") || emLower.includes("pos")) emotionLabel = "Happy";
    else if (emLower.includes("sad") || emLower.includes("neg") || emLower.includes("tumor") || emLower.includes("glioma")) emotionLabel = "Sad";
    else if (emLower.includes("ang")) emotionLabel = "Angry";
    else if (emLower.includes("surpris")) emotionLabel = "Surprised";
    else if (emLower.includes("calm")) emotionLabel = "Calm";
    else emotionLabel = "Neutral";

    const combined = {
      label: emotionLabel,
      valence: 0.5,
      arousal: 0.5,
      confidence: fusionData.confidence || 0.5,
    };

    const contribs = results.map((r) => ({
      modality: r.modality,
      label: r.prediction,
      confidence: r.confidence || 0.5,
    }));

    renderResult(combined, contribs);
  } catch (err) {
    window.alert(`Prediction failed: ${err.message}`);
  } finally {
    predictAllBtn.disabled = false;
    predictAllBtn.classList.remove("loading");
  }
}

function wireButtons() {
  Object.keys(NUMERIC_MODALITIES).forEach((key) => {
    document.getElementById(`${key}-random`).addEventListener("click", () => randomExampleForNumeric(key));
    document.getElementById(`${key}-clear`).addEventListener("click", () => clearNumeric(key));
  });

  Object.keys(FILE_MODALITIES).forEach((key) => {
    const config = FILE_MODALITIES[key];
    document.getElementById(config.randomId).addEventListener("click", () => randomExampleForFile(key));
    document.getElementById(config.clearId).addEventListener("click", () => clearFileState(key));
    document.getElementById(config.inputId).addEventListener("change", () => {
      handleFileChange(key).catch((error) => {
        clearFileState(key);
        window.alert(`Failed to process ${config.label} input: ${error.message}`);
      });
    });
  });

  predictAllBtn.addEventListener("click", () => {
    predictCombined().catch((error) => {
      window.alert(`Prediction failed: ${error.message}`);
    });
  });

  clearAllBtn.addEventListener("click", clearAll);
  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme;
    setTheme(current === "dark" ? "light" : "dark");
  });

  // Webcam Event Listeners
  if (faceWebcamBtn) {
    faceWebcamBtn.addEventListener("click", toggleWebcam);
  }
  if (faceCaptureBtn) {
    faceCaptureBtn.addEventListener("click", captureFaceSnapshot);
  }
}

async function toggleWebcam() {
  if (webcamStream) {
    stopWebcam();
  } else {
    await startWebcam();
  }
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcamVideo.srcObject = webcamStream;
    webcamContainer.style.display = "flex";
    faceWebcamBtn.textContent = "Stop Webcam";
    faceWebcamBtn.classList.add("active");
  } catch (err) {
    window.alert("Webcam access denied: " + err.message);
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  webcamVideo.srcObject = null;
  webcamContainer.style.display = "none";
  faceWebcamBtn.textContent = "Use Webcam";
  faceWebcamBtn.classList.remove("active");
}

async function captureFaceSnapshot() {
  const context = webcamCanvas.getContext("2d");
  webcamCanvas.width = webcamVideo.videoWidth;
  webcamCanvas.height = webcamVideo.videoHeight;
  context.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
  
  webcamCanvas.toBlob(async (blob) => {
    const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
    await processCapturedFile("face", file);
    stopWebcam();
  }, "image/jpeg", 0.95);
}

async function processCapturedFile(key, file) {
  const config = FILE_MODALITIES[key];
  clearFileState(key);
  fileState[key].file = file;
  const analysis = await analyzeImageFile(file);
  fileState[key].previewUrl = analysis.previewUrl;
  fileState[key].stats = analysis.stats;
  setImagePreview(config.previewId, analysis.previewUrl);
  setMeta(config.metaId, `Webcam Capture • ${formatStats(analysis.stats)}`);
}

function init() {
  initTheme();
  initCanvas();
  animateBackground();
  renderNumericFields();
  wireButtons();
  resetResult();
}

init();

// Theory tab switching
document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.theory-tab-btn');
  const contents = document.querySelectorAll('.theory-tab-content');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => {
        t.classList.remove('active');
        t.classList.add('ghost');
        t.classList.remove('glass');
      });
      btn.classList.add('active', 'glass');
      btn.classList.remove('ghost');
      contents.forEach(c => c.hidden = true);
      const target = document.getElementById('theory-tab-' + btn.dataset.tab);
      if (target) target.hidden = false;
    });
  });
});
