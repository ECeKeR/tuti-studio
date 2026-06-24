import { addLog } from "./logger.js";

let activeEventSource = null;
let activeRepoId = null;

export async function loadModelsStatus() {
  try {
    const res = await fetch("/api/models/status");
    const data = await res.json();
    if (!data.models) return;

    const cloneList = document.getElementById("models-list-clone");
    const presetList = document.getElementById("models-list-preset");
    const designList = document.getElementById("models-list-design");

    if (!cloneList || !presetList || !designList) return;

    // Clear lists
    cloneList.innerHTML = "";
    presetList.innerHTML = "";
    designList.innerHTML = "";

    data.models.forEach(model => {
      const cardHtml = renderModelCard(model);
      if (model.type === "clone") {
        cloneList.insertAdjacentHTML("beforeend", cardHtml);
      } else if (model.type === "preset") {
        presetList.insertAdjacentHTML("beforeend", cardHtml);
      } else if (model.type === "design") {
        designList.insertAdjacentHTML("beforeend", cardHtml);
      }
    });

    if (window.lucide) window.lucide.createIcons();
  } catch (e) {
    console.error("Failed to load models status:", e);
    addLog("Failed to load models status: " + e.message, "error");
  }
}

function renderModelCard(model) {
  const isCached = model.cached;
  const isDownloading = activeRepoId === model.id;
  
  const statusBadge = isCached 
    ? `<span style="font-size:0.7rem; font-weight:700; color:var(--green); background:rgba(0, 230, 118, 0.08); border:1px solid rgba(0, 230, 118, 0.15); padding:3px 8px; border-radius:4px; text-transform:uppercase; letter-spacing:0.04em; display:inline-flex; align-items:center; gap:4px;"><i data-lucide="check-circle-2" style="width:10px;height:10px;"></i> Cached</span>`
    : isDownloading 
      ? `<span style="font-size:0.7rem; font-weight:700; color:var(--cyan); background:rgba(0, 229, 255, 0.08); border:1px solid rgba(0, 229, 255, 0.15); padding:3px 8px; border-radius:4px; text-transform:uppercase; letter-spacing:0.04em; display:inline-flex; align-items:center; gap:4px;"><i data-lucide="loader-2" class="spin" style="width:10px;height:10px;"></i> Downloading</span>`
      : `<span style="font-size:0.7rem; font-weight:700; color:var(--text3); background:rgba(255,255,255,0.04); border:1px solid var(--border); padding:3px 8px; border-radius:4px; text-transform:uppercase; letter-spacing:0.04em; display:inline-flex; align-items:center; gap:4px;"><i data-lucide="cloud-off" style="width:10px;height:10px;"></i> Not Cached</span>`;

  const actionButton = isCached
    ? `<button class="btn btn-ghost btn-sm" disabled style="opacity:0.5; padding:4px 10px; font-size:0.7rem;">Ready</button>`
    : isDownloading
      ? `<button class="btn btn-ghost btn-sm" onclick="cancelModelDownload()" style="padding:4px 10px; font-size:0.7rem; border-color:rgba(255, 82, 82, 0.2); color:var(--red);">Abort</button>`
      : `<button class="btn btn-primary btn-sm" onclick="downloadModel('${model.id}')" style="padding:4px 10px; font-size:0.7rem; background:linear-gradient(135deg, var(--neon-blue) 0%, #004bb8 100%);"><i data-lucide="download" style="width:11px;height:11px;"></i> Download</button>`;

  // Create progress bar placeholder
  const progressBarHtml = isDownloading
    ? `
      <div style="margin-top:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.68rem; color:var(--text2); margin-bottom:4px;">
          <span id="download-filename" style="text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:200px;">Connecting...</span>
          <span id="download-percentage" style="font-family:monospace; font-weight:bold; color:var(--cyan);">0%</span>
        </div>
        <div style="height:4px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden;">
          <div id="download-progress-bar" style="width:0%; height:100%; background:linear-gradient(90deg, var(--neon-blue), var(--cyan)); border-radius:2px; transition: width 0.1s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:var(--text3); margin-top:4px;">
          <span id="download-bytes">0 MB / 0 MB</span>
          <span id="download-status-txt">Starting snapshot...</span>
        </div>
      </div>
    `
    : "";

  return `
    <div class="card-surface" style="display:flex; flex-direction:column; padding:14px; border:1px solid var(--border); border-radius:12px; background:rgba(255,255,255,0.015); gap:8px;">
      <div style="display:flex; justify-content:space-between; align-items:start; gap:12px;">
        <div style="min-width:0; display:flex; flex-direction:column; gap:4px;">
          <span style="font-size:0.85rem; color:var(--text); font-weight:600; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${model.name}</span>
          <span style="font-size:0.68rem; color:var(--cyan); font-family:monospace; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${model.id}</span>
          <span style="font-size:0.75rem; color:var(--text2); line-height:1.4; display:block; margin-top:2px;">${model.description}</span>
        </div>
        <div style="display:flex; flex-direction:column; align-items:end; gap:6px; flex-shrink:0;">
          ${statusBadge}
          <div style="font-size:0.68rem; color:var(--text3); font-family:monospace;">${model.backend.toUpperCase()} · ${model.size_label}</div>
        </div>
      </div>
      
      <div style="display:flex; justify-content:flex-end; border-top:1px solid rgba(255,255,255,0.03); padding-top:10px; margin-top:4px; align-items:center;">
        ${actionButton}
      </div>

      ${progressBarHtml}
    </div>
  `;
}

export function cancelModelDownload() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  const prevRepo = activeRepoId;
  activeRepoId = null;
  loadModelsStatus();
  addLog(`Model download aborted: ${prevRepo}`, "warn");
}

export function downloadModel(repoId) {
  if (activeRepoId) {
    alert("Another download is currently running. Please wait or abort the active download first.");
    return;
  }

  activeRepoId = repoId;
  loadModelsStatus(); // Redraw status list to show progress elements on this card

  addLog(`Downloading Hugging Face model: ${repoId}...`, "info");

  const esc = new EventSource(`/api/models/download/stream?repo_id=${encodeURIComponent(repoId)}`);
  activeEventSource = esc;

  esc.onmessage = function (event) {
    const data = JSON.parse(event.data);

    if (data.status === "downloading") {
      const filenameEl = document.getElementById("download-filename");
      const percentageEl = document.getElementById("download-percentage");
      const barEl = document.getElementById("download-progress-bar");
      const bytesEl = document.getElementById("download-bytes");
      const statusTxtEl = document.getElementById("download-status-txt");

      const downloadedMb = (data.downloaded / (1024 * 1024)).toFixed(1);
      const totalMb = (data.total / (1024 * 1024)).toFixed(1);
      const percentage = data.total > 0 ? Math.round((data.downloaded / data.total) * 100) : 0;

      if (filenameEl) filenameEl.textContent = data.filename || "Downloading model files...";
      if (percentageEl) percentageEl.textContent = `${percentage}%`;
      if (barEl) barEl.style.width = `${percentage}%`;
      if (bytesEl) bytesEl.textContent = `${downloadedMb} MB / ${totalMb} MB`;
      if (statusTxtEl) statusTxtEl.textContent = "Downloading...";
    } else if (data.status === "completed") {
      addLog(`✓ Model ${repoId} downloaded successfully and cached!`, "success");
      esc.close();
      activeEventSource = null;
      activeRepoId = null;
      loadModelsStatus();
    } else if (data.status === "error") {
      addLog(`Failed to download model ${repoId}: ${data.error}`, "error");
      alert(`Model download error:\n${data.error}`);
      esc.close();
      activeEventSource = null;
      activeRepoId = null;
      loadModelsStatus();
    }
  };

  esc.onerror = function (err) {
    console.error("SSE Connection error:", err);
    esc.close();
    activeEventSource = null;
    activeRepoId = null;
    loadModelsStatus();
  };
}

export function initModels() {
  window.loadModelsStatus = loadModelsStatus;
  window.downloadModel = downloadModel;
  window.cancelModelDownload = cancelModelDownload;
}
