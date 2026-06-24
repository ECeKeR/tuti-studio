import { state } from "../state.js";
import { addLog } from "./logger.js";
import { updateSettingsRecap } from "./sidebar.js";

export function setVoiceMode(mode) {
  document.getElementById("tab-preset").classList.toggle("active", mode === "preset");
  document.getElementById("tab-clone").classList.toggle("active", mode === "clone");
  document.getElementById("tab-design").classList.toggle("active", mode === "design");
  document.getElementById("preset-options").style.display = mode === "preset" ? "" : "none";
  document.getElementById("clone-options").style.display = mode === "clone" ? "" : "none";
  document.getElementById("design-options").style.display = mode === "design" ? "" : "none";
  
  const sftContainer = document.getElementById("sft-status-container");
  if (sftContainer) {
    if (mode === "preset") {
      const speakerEl = document.getElementById("speaker-input");
      const selectedOpt = speakerEl && speakerEl.options[speakerEl.selectedIndex];
      const isSFT = selectedOpt && selectedOpt.getAttribute("data-sft") === "true";
      sftContainer.style.display = isSFT ? "flex" : "none";
    } else {
      sftContainer.style.display = "none";
    }
  }
  updateSettingsRecap();
}

export async function handleFileUpload(input) {
  if (!input.files.length) return;
  const file = input.files[0];
  const zone = document.getElementById("upload-zone");
  if (!zone) return;
  
  const sftContainer = document.getElementById("sft-status-container");
  if (sftContainer) sftContainer.style.display = "none";
  
  zone.innerHTML = `<i data-lucide="loader-2"></i><div class="upload-zone-label">Uploading ${file.name}...</div>`;
  if (window.lucide) window.lucide.createIcons();

  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/project/upload_ref", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    document.getElementById("ref-audio-input").value = data.filePath;
    zone.className = "upload-zone upload-success";
    zone.innerHTML = `<i data-lucide="check-circle-2"></i><div class="upload-zone-label">${file.name}</div><div class="upload-zone-sub">Denoised &amp; ready for cloning</div>`;
    if (window.lucide) window.lucide.createIcons();
    addLog("Voice sample uploaded: " + file.name, "success");
  } catch (e) {
    zone.innerHTML = `<i data-lucide="upload-cloud"></i><div class="upload-zone-label">Upload failed — try again</div><div class="upload-zone-sub">${e.message}</div>`;
    if (window.lucide) window.lucide.createIcons();
    addLog("Upload failed: " + e.message, "error");
  }
}

export async function loadTimbres() {
  try {
    const res = await fetch("/api/timbre/list");
    const d = await res.json();
    
    const timbreList = document.getElementById("timbre-list");
    const libraryList = document.getElementById("timbre-list-library");
    const finetuneSelect = document.getElementById("finetune-select-voice");
    
    if (!d.timbres || d.timbres.length === 0) {
      const emptyMsg = '<div style="font-size:0.75rem;color:var(--text3);padding:10px;">No saved voices yet.</div>';
      if (timbreList) timbreList.innerHTML = emptyMsg;
      if (libraryList) libraryList.innerHTML = emptyMsg;
      if (finetuneSelect) {
        finetuneSelect.innerHTML = '<option value="">-- No Saved Voices Loaded --</option>';
      }
      return;
    }
    
    const renderItem = (t, showLoadAction = false) => {
      const lucideIcon = t.voice_mode === 'design' ? 'wand-2' : t.voice_mode === 'clone' ? 'mic-2' : 'user';
      const iconColor = t.voice_mode === 'design' ? 'var(--green)' : t.voice_mode === 'clone' ? 'var(--neon-blue)' : 'var(--cyan)';
      const typeLabel = t.voice_mode === 'design' ? 'Design' : t.voice_mode === 'clone' ? 'Clone' : 'Preset';
      const subText = t.voice_mode === 'design' ? t.design_prompt : t.voice_mode === 'clone' ? t.ref_audio.split('/').pop() : `Speaker: ${t.speaker}`;
      
      const sftBadge = t.sft_enabled ? `
        <div style="display:flex; align-items:center; margin-top:2px;">
          <span style="font-size:0.58rem; font-weight:700; color:var(--cyan); background:rgba(0, 229, 255, 0.08); border:1px solid rgba(0, 229, 255, 0.15); padding:1px 6px; border-radius:4px; text-transform:uppercase; letter-spacing:0.04em; display:inline-flex; align-items:center; gap:3px;">
            <i data-lucide="zap" style="width:8px;height:8px;"></i> Fine-Tuning
          </span>
        </div>
      ` : '';

      return `
        <div class="card-surface" style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; border:1px solid var(--border); border-radius:10px; background:rgba(255,255,255,0.015); gap:12px; width:100%;">
          <div style="display:flex; align-items:center; gap:10px; cursor:pointer; min-width:0; flex:1;" onclick="loadTimbre('${t.name}')">
            <span style="display:flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:6px; background:rgba(255,255,255,0.03); border:1px solid var(--border); flex-shrink:0;">
              <i data-lucide="${lucideIcon}" style="color:${iconColor}; width:16px; height:16px;"></i>
            </span>
            <div style="min-width:0; display:flex; flex-direction:column; gap:2px;">
              <span style="font-size:0.82rem; color:var(--text); font-weight:600; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${t.name}</span>
              <span style="font-size:0.68rem; color:var(--text2); text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${typeLabel} · ${subText}</span>
              ${sftBadge}
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:4px; flex-shrink:0;">
            ${showLoadAction ? `<button class="btn btn-ghost btn-sm" onclick="loadTimbre('${t.name}'); switchTab('studio');" style="padding:4px 8px; font-size:0.7rem;">Use</button>` : ""}
            <button class="btn btn-ghost btn-sm btn-icon" onclick="deleteTimbre('${t.name}')" style="padding:4px; opacity:0.5; border:none; background:transparent" title="Delete">
              <i data-lucide="trash-2" style="width:13px;height:13px; color:var(--red);"></i>
            </button>
          </div>
        </div>
      `;
    };
    
    if (timbreList) {
      timbreList.innerHTML = d.timbres.map(t => renderItem(t, false)).join("");
    }
    if (libraryList) {
      libraryList.innerHTML = d.timbres.map(t => renderItem(t, true)).join("");
    }
    
    if (finetuneSelect) {
      finetuneSelect.innerHTML = '<option value="">-- Select Saved Voice --</option>' +
        d.timbres.map(t => `<option value="${t.name}">${t.name} (${t.voice_mode})</option>`).join("");
    }

    const speakerInput = document.getElementById("speaker-input");
    if (speakerInput) {
      const currentVal = speakerInput.value;
      const sftVoices = d.timbres.filter(t => t.sft_enabled || t.voice_mode === "clone");
      let optionsHtml = `
        <option value="ethan">Ethan (M)</option>
        <option value="uncle_fu">Uncle Fu (M)</option>
        <option value="vivian">Vivian (F)</option>
        <option value="serena">Serena (F)</option>
        <option value="ono_anna">Ono Anna (F)</option>
        <option value="sohee">Sohee (F)</option>
      `;
      sftVoices.forEach(v => {
        const label = v.sft_enabled ? `${v.name} (Custom SFT)` : `${v.name} (Custom Voice)`;
        optionsHtml += `<option value="${v.name}" data-sft="true" data-model="${v.model_size || '1.7B'}">${label}</option>`;
      });
      speakerInput.innerHTML = optionsHtml;
      if (currentVal && speakerInput.querySelector(`option[value="${currentVal}"]`)) {
        speakerInput.value = currentVal;
      }
    }
    
    if (window.lucide) window.lucide.createIcons();
  } catch (e) {
    console.error("Failed to load timbres:", e);
  }
}

let pendingPayload = null;

export async function saveCurrentTimbre() {
  const isClone = document.getElementById("tab-clone").classList.contains("active");
  const isDesign = document.getElementById("tab-design").classList.contains("active");
  const voiceMode = isDesign ? "design" : isClone ? "clone" : "preset";
  
  let payload = {
    voice_mode: voiceMode,
  };
  
  if (voiceMode === "design") {
    const prompt = document.getElementById("design-prompt-input").value.trim();
    if (!prompt) {
      addLog("Please describe a voice first before saving.", "warn");
      return;
    }
    payload.design_prompt = prompt;
    payload.model_size = document.getElementById("model-size-design-input").value;
  } else if (voiceMode === "clone") {
    const refAudio = document.getElementById("ref-audio-input").value.trim();
    if (!refAudio) {
      addLog("Please upload a reference voice sample first before saving.", "warn");
      return;
    }
    payload.ref_audio = refAudio;
    payload.ref_text = "";
    payload.clone_alpha = parseFloat(document.getElementById("clone-alpha-input").value) || 1.0;
    payload.clone_topk = parseInt(document.getElementById("clone-topk-input").value, 10) || 3;
    // Model always fixed for interpolation method
    payload.model_size = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16";
  } else {
    payload.speaker = document.getElementById("speaker-input").value;
    payload.model_size = document.getElementById("model-size-input").value;
  }
  
  const defaultName = voiceMode === "preset" ? payload.speaker : voiceMode === "clone" ? "cloned_voice" : "designed_voice";
  
  pendingPayload = payload;

  const modal = document.getElementById("save-voice-modal");
  const nameInput = document.getElementById("save-voice-name-input");
  if (modal && nameInput) {
    nameInput.value = defaultName;
    modal.style.display = "flex";
    nameInput.focus();
  }
}

export async function confirmSaveCurrentTimbre() {
  const nameInput = document.getElementById("save-voice-name-input");
  if (!nameInput) return;
  const name = nameInput.value.trim();
  if (!name) {
    addLog("Voice name is required.", "warn");
    return;
  }

  if (!pendingPayload) return;
  const payload = { ...pendingPayload, name };

  try {
    const res = await fetch("/api/timbre/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    addLog(`Voice "${name}" saved to registry.`, "success");
    closeSaveVoiceModal();
    loadTimbres();
  } catch (e) {
    addLog("Save failed: " + e.message, "error");
  }
}

export function closeSaveVoiceModal() {
  const modal = document.getElementById("save-voice-modal");
  if (modal) {
    modal.style.display = "none";
  }
  pendingPayload = null;
}

export async function loadTimbre(name) {
  try {
    const res = await fetch("/api/timbre/list");
    const d = await res.json();
    const t = d.timbres.find(x => x.name === name);
    if (!t) return;

    if (t.sft_enabled) {
      setVoiceMode("preset");
      const speakerInput = document.getElementById("speaker-input");
      if (speakerInput) {
        speakerInput.value = t.name;
      }
      
      const modelSizeInput = document.getElementById("model-size-input");
      if (modelSizeInput) {
        const customModelId = t.model_size === "0.6B"
          ? "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
          : "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16";
        modelSizeInput.value = customModelId;
      }

      const sftContainer = document.getElementById("sft-status-container");
      const sftActiveName = document.getElementById("sft-active-name");
      if (sftContainer && sftActiveName) {
        sftContainer.style.display = "flex";
        sftActiveName.innerText = t.name;
      }
    } else if (t.voice_mode === "design") {
      setVoiceMode("design");
      document.getElementById("design-prompt-input").value = t.design_prompt || "";
      if (t.model_size) {
        const el = document.getElementById("model-size-design-input");
        if (el) el.value = t.model_size.startsWith("mlx-community/") ? t.model_size : (t.model_size === "0.6B" ? "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-bf16" : "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16");
      }
    } else if (t.voice_mode === "clone") {
      setVoiceMode("clone");
      document.getElementById("ref-audio-input").value = t.ref_audio || "";

      // Restore alpha slider
      const alphaVal = t.clone_alpha !== undefined ? t.clone_alpha : 1.0;
      const alphaSlider = document.getElementById("slider-clone-alpha");
      const alphaInput = document.getElementById("clone-alpha-input");
      const alphaLabel = document.getElementById("val-clone-alpha");
      if (alphaSlider) alphaSlider.value = alphaVal;
      if (alphaInput) alphaInput.value = alphaVal;
      if (alphaLabel) alphaLabel.textContent = Number(alphaVal).toFixed(2);

      // Restore top-k select
      const topkVal = t.clone_topk !== undefined ? String(t.clone_topk) : "3";
      const topkSelect = document.getElementById("clone-topk-input");
      if (topkSelect) topkSelect.value = topkVal;
      
      const zone = document.getElementById("upload-zone");
      if (zone && t.ref_audio) {
        zone.className = "upload-zone upload-success";
        const fileName = t.ref_audio.split("/").pop();
        zone.innerHTML = `<i data-lucide="check-circle-2"></i><div class="upload-zone-label">${fileName}</div><div class="upload-zone-sub">Loaded from saved voice</div>`;
        if (window.lucide) window.lucide.createIcons();
      }

      // Check and show SFT adapter active state
      const sftContainer = document.getElementById("sft-status-container");
      const sftActiveName = document.getElementById("sft-active-name");
      if (t.sft_enabled) {
        if (sftContainer && sftActiveName) {
          sftContainer.style.display = "flex";
          sftActiveName.innerText = t.name;
        }
      } else {
        if (sftContainer) {
          sftContainer.style.display = "none";
        }
      }
    } else {
      setVoiceMode("preset");
      if (t.speaker) document.getElementById("speaker-input").value = t.speaker;
      if (t.model_size) {
        const el = document.getElementById("model-size-input");
        if (el) el.value = t.model_size.startsWith("mlx-community/") ? t.model_size : (t.model_size === "0.6B" ? "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16" : "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16");
      }
    }
    
    updateSettingsRecap();
    addLog(`Voice "${name}" loaded.`, "success");
  } catch (e) {
    addLog("Load failed: " + e.message, "error");
  }
}

export async function deleteTimbre(name) {
  try {
    const res = await fetch(`/api/timbre/delete?name=${encodeURIComponent(name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    if (!res.ok) {
      const errText = await res.text();
      let errMsg = "HTTP error " + res.status;
      try {
        const errJson = JSON.parse(errText);
        if (errJson.error) errMsg = errJson.error;
      } catch(_) {}
      throw new Error(errMsg);
    }
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    addLog(`Voice "${name}" deleted.`, "success");
    loadTimbres();
  } catch (e) {
    addLog("Delete failed: " + e.message, "error");
  }
}

let lossHistory = [];
let activeEventSource = null;

export function drawLossChart() {
  const canvas = document.getElementById("loss-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  
  const w = rect.width;
  const h = rect.height;
  
  ctx.clearRect(0, 0, w, h);
  
  const placeholder = document.getElementById("loss-placeholder");
  if (lossHistory.length < 2) {
    if (placeholder) placeholder.style.display = lossHistory.length === 0 ? "block" : "none";
    if (lossHistory.length === 1) {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, 4, 0, 2 * Math.PI);
      ctx.fillStyle = "var(--cyan)";
      ctx.fill();
    }
    return;
  }
  
  if (placeholder) placeholder.style.display = "none";
  
  // Draw grid
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  
  const minLoss = Math.min(...lossHistory) * 0.9;
  const maxLoss = Math.max(...lossHistory) * 1.1;
  const lossRange = maxLoss - minLoss || 1;
  
  // Glowing background under curve
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < lossHistory.length; i++) {
    const x = (w / (lossHistory.length - 1)) * i;
    const y = h - ((lossHistory[i] - minLoss) / lossRange) * (h - 20) - 10;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  
  const gradient = ctx.createLinearGradient(0, 0, 0, h);
  gradient.addColorStop(0, "rgba(0, 229, 255, 0.12)");
  gradient.addColorStop(1, "rgba(0, 229, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.fill();
  
  // Draw stroke
  ctx.beginPath();
  for (let i = 0; i < lossHistory.length; i++) {
    const x = (w / (lossHistory.length - 1)) * i;
    const y = h - ((lossHistory[i] - minLoss) / lossRange) * (h - 20) - 10;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.strokeStyle = "var(--cyan)";
  ctx.lineWidth = 2.5;
  ctx.shadowColor = "rgba(0, 229, 255, 0.35)";
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

export function abortVoiceFinetune() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  
  const selectEl = document.getElementById("finetune-select-voice");
  const voiceName = selectEl ? selectEl.value : "Unknown";
  const terminalEl = document.getElementById("sim-terminal");
  const statusEl = document.getElementById("sim-status");
  const btnEl = document.getElementById("btn-start-finetune");
  const btnAbort = document.getElementById("btn-abort-finetune");
  
  if (statusEl) {
    statusEl.innerText = "ABORTED";
    statusEl.style.color = "var(--red)";
    statusEl.style.background = "rgba(255, 82, 82, 0.08)";
  }
  
  if (btnEl) {
    btnEl.disabled = false;
    btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
  }
  if (btnAbort) {
    btnAbort.disabled = true;
  }
  
  if (terminalEl) {
    const lineDiv = document.createElement("div");
    lineDiv.style.display = "flex";
    lineDiv.style.alignItems = "flex-start";
    lineDiv.style.gap = "8px";
    lineDiv.style.fontSize = "0.78rem";
    lineDiv.style.padding = "4px 8px";
    lineDiv.style.borderRadius = "6px";
    lineDiv.style.background = "rgba(255, 82, 82, 0.03)";
    lineDiv.style.borderColor = "rgba(255, 82, 82, 0.08)";
    lineDiv.style.borderStyle = "solid";
    lineDiv.style.borderWidth = "1px";
    lineDiv.style.marginTop = "4px";
    
    lineDiv.innerHTML = `
      <i data-lucide="x-circle" style="width: 13px; height: 13px; color: var(--red); margin-top: 2px; flex-shrink: 0;"></i>
      <div style="flex: 1; word-break: break-word;">
        <span style="font-weight: 700; color: var(--red); font-size: 0.72rem; margin-right: 4px; text-transform: uppercase;">[ABORTED]</span>
        <span style="color: var(--text);">Fine-tuning manually aborted by user. Training loop halted.</span>
      </div>
    `;
    terminalEl.appendChild(lineDiv);
    terminalEl.scrollTop = terminalEl.scrollHeight;
  }
  
  addLog(`Fine-tuning aborted for: ${voiceName}`, "warn");
  if (window.lucide) window.lucide.createIcons();
}

let currentPreparedSegments = [];

export function playSegmentAudio(url) {
  const audio = new Audio(url);
  audio.play().catch(err => console.error("Failed to play segment audio:", err));
}

export function cancelFinetuneReview() {
  currentPreparedSegments = [];
  const reviewPanel = document.getElementById("finetune-review-panel");
  if (reviewPanel) {
    reviewPanel.style.display = "none";
  }
  const btnEl = document.getElementById("btn-start-finetune");
  if (btnEl) {
    btnEl.disabled = false;
    btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
  }
  if (window.lucide) window.lucide.createIcons();
}

export async function startVoiceFinetune() {
  const selectEl = document.getElementById("finetune-select-voice");
  const btnEl = document.getElementById("btn-start-finetune");

  if (!selectEl) return;

  const voiceName = selectEl.value;
  if (!voiceName) {
    alert("Please select a saved voice profile first.");
    return;
  }

  // Set analyzing status on the main button
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Analyzing Audio...';
  }
  if (window.lucide) window.lucide.createIcons();

  addLog(`Running Whisper alignment & data auditing analysis for voice: ${voiceName}...`, "info");

  try {
    const res = await fetch(`/api/timbre/finetune/prepare?name=${encodeURIComponent(voiceName)}`);
    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }

    currentPreparedSegments = data.segments || [];
    
    // Render the table
    const tbody = document.getElementById("finetune-segments-tbody");
    if (tbody) {
      tbody.innerHTML = "";
      if (currentPreparedSegments.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 20px; color: var(--text3);">No segments could be extracted.</td></tr>`;
      } else {
        currentPreparedSegments.forEach(seg => {
          const tr = document.createElement("tr");
          tr.style.borderBottom = "1px solid rgba(255,255,255,0.03)";
          
          const statusColor = seg.status === "Green" ? "var(--green)" : "var(--amber)";
          const statusBg = seg.status === "Green" ? "rgba(0, 230, 118, 0.08)" : "rgba(255, 179, 0, 0.08)";
          
          tr.innerHTML = `
            <td style="padding: 12px; color: var(--text2); font-family: monospace;">${seg.id}</td>
            <td style="padding: 12px;">
              <button class="btn btn-ghost" style="padding: 4px 8px; font-size: 0.75rem; border-color: rgba(255,255,255,0.08);" onclick="playSegmentAudio('${seg.audio_url}')">
                <i data-lucide="play" style="width: 12px; height: 12px; margin-right: 4px;"></i> Play
              </button>
            </td>
            <td style="padding: 12px;">
              <input type="text" id="seg-text-${seg.id}" value="${seg.text.replace(/"/g, '&quot;')}" style="width: 100%; background: rgba(0,0,0,0.2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; color: var(--text); font-size: 0.8rem;" />
            </td>
            <td style="padding: 12px; text-align: center; font-family: monospace; font-weight: bold; color: ${statusColor};">${seg.confidence}%</td>
            <td style="padding: 12px; text-align: center;">
              <span style="font-size: 0.7rem; font-weight: 700; color: ${statusColor}; background: ${statusBg}; padding: 2px 8px; border-radius: 4px; text-transform: uppercase;">
                ${seg.status === "Green" ? "Aligned" : "Suspicious"}
              </span>
            </td>
          `;
          tbody.appendChild(tr);
        });
      }
    }

    // Show the review panel
    const reviewPanel = document.getElementById("finetune-review-panel");
    if (reviewPanel) {
      reviewPanel.style.display = "flex";
      reviewPanel.scrollIntoView({ behavior: "smooth" });
    }

    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
    }
    if (window.lucide) window.lucide.createIcons();

    addLog(`Data preparation complete. Please review the segmented script in the table below.`, "success");

  } catch (err) {
    alert("Preparation failed: " + err.message);
    addLog("Preparation failed: " + err.message, "error");
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
    }
    if (window.lucide) window.lucide.createIcons();
  }
}

export async function confirmAndStartFinetune() {
  const selectEl = document.getElementById("finetune-select-voice");
  const terminalEl = document.getElementById("sim-terminal");
  const statusEl = document.getElementById("sim-status");
  const btnEl = document.getElementById("btn-start-finetune");
  const btnAbort = document.getElementById("btn-abort-finetune");
  const reviewPanel = document.getElementById("finetune-review-panel");

  if (!selectEl || !terminalEl || !statusEl) return;

  const voiceName = selectEl.value;
  if (!voiceName) return;

  // Save edits
  currentPreparedSegments.forEach(seg => {
    const inputEl = document.getElementById("seg-text-" + seg.id);
    if (inputEl) {
      seg.text = inputEl.value;
    }
  });

  addLog(`Saving verified transcripts for ${voiceName}...`, "info");

  try {
    const saveRes = await fetch("/api/timbre/finetune/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: voiceName, segments: currentPreparedSegments })
    });
    const saveData = await saveRes.json();
    if (saveData.error) {
      throw new Error(saveData.error);
    }
  } catch (err) {
    alert("Failed to save segment edits: " + err.message);
    addLog("Failed to save segment edits: " + err.message, "error");
    return;
  }

  // Hide review panel
  if (reviewPanel) {
    reviewPanel.style.display = "none";
  }

  // Reset loss history
  lossHistory = [];
  const lossDisplay = document.getElementById("loss-display");
  if (lossDisplay) lossDisplay.innerText = "--";
  drawLossChart();

  // Clear terminal and set running status
  terminalEl.innerHTML = "";
  statusEl.innerText = "RUNNING";
  statusEl.style.color = "var(--cyan)";
  statusEl.style.background = "rgba(0, 229, 255, 0.08)";
  
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Fine-Tuning...';
  }
  if (btnAbort) {
    btnAbort.disabled = false;
  }
  if (window.lucide) window.lucide.createIcons();

  const modelIdEl = document.getElementById("finetune-model-id");
  const modelId = modelIdEl ? modelIdEl.value : "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16";

  const lr = document.getElementById("finetune-lr").value;
  const steps = document.getElementById("finetune-steps").value;
  const rank = document.getElementById("finetune-rank").value;
  const alpha = document.getElementById("finetune-alpha").value;

  addLog(`Starting SFT fine-tuning (LR: ${lr}, Steps: ${steps}, Rank: ${rank}, Alpha: ${alpha}) for: ${voiceName}`, "info");

  // Create EventSource stream connect
  const esc = new EventSource(`/api/timbre/finetune/stream?name=${encodeURIComponent(voiceName)}&model=${encodeURIComponent(modelId)}&lr=${encodeURIComponent(lr)}&steps=${encodeURIComponent(steps)}&r=${encodeURIComponent(rank)}&alpha=${encodeURIComponent(alpha)}`);
  activeEventSource = esc;

  esc.onmessage = function (event) {
    const line = event.data.trim();
    if (!line) return;
    const lineDiv = document.createElement("div");
    
    lineDiv.style.display = "flex";
    lineDiv.style.alignItems = "flex-start";
    lineDiv.style.gap = "8px";
    lineDiv.style.fontSize = "0.78rem";
    lineDiv.style.padding = "6px 8px";
    lineDiv.style.borderRadius = "6px";
    lineDiv.style.background = "rgba(255,255,255,0.015)";
    lineDiv.style.border = "1px solid rgba(255,255,255,0.03)";

    let icon = "info";
    let color = "var(--text2)";
    let prefix = "INFO";
    let messageText = line;

    if (line.startsWith("SUCCESS:")) {
      icon = "check-circle-2";
      color = "var(--green)";
      prefix = "SUCCESS";
      messageText = line.substring(8).trim();
      lineDiv.style.background = "rgba(0, 230, 118, 0.03)";
      lineDiv.style.borderColor = "rgba(0, 230, 118, 0.08)";
    } else if (line.startsWith("WARN:")) {
      icon = "alert-triangle";
      color = "var(--amber)";
      prefix = "WARN";
      messageText = line.substring(5).trim();
      lineDiv.style.background = "rgba(255, 179, 0, 0.03)";
      lineDiv.style.borderColor = "rgba(255, 179, 0, 0.08)";
    } else if (line.startsWith("ERROR:")) {
      icon = "x-circle";
      color = "var(--red)";
      prefix = "ERROR";
      messageText = line.substring(6).trim();
      lineDiv.style.background = "rgba(255, 82, 82, 0.03)";
      lineDiv.style.borderColor = "rgba(255, 82, 82, 0.08)";
    } else if (line.startsWith("TRAIN:")) {
      icon = "activity";
      color = "var(--cyan)";
      prefix = "TRAIN";
      messageText = line.substring(6).trim();
      lineDiv.style.background = "rgba(0, 229, 255, 0.02)";
      lineDiv.style.borderColor = "rgba(0, 229, 255, 0.06)";

      const lossMatch = messageText.match(/loss:\s*([0-9.]+)/i);
      if (lossMatch) {
        const val = parseFloat(lossMatch[1]);
        lossHistory.push(val);
        if (lossDisplay) lossDisplay.innerText = val.toFixed(4);
        drawLossChart();
      }
    } else if (line.startsWith("INFO:")) {
      icon = "info";
      color = "var(--text2)";
      prefix = "INFO";
      messageText = line.substring(5).trim();
    }

    lineDiv.innerHTML = `
      <i data-lucide="${icon}" style="width: 13px; height: 13px; color: ${color}; margin-top: 2px; flex-shrink: 0;"></i>
      <div style="flex: 1; word-break: break-word;">
        <span style="font-weight: 700; color: ${color}; font-size: 0.72rem; margin-right: 4px; text-transform: uppercase;">[${prefix}]</span>
        <span style="color: var(--text);">${messageText}</span>
      </div>
    `;

    terminalEl.appendChild(lineDiv);
    terminalEl.scrollTop = terminalEl.scrollHeight;

    if (window.lucide) window.lucide.createIcons();

    if (line.includes("is now SFT-enabled")) {
      setTimeout(() => {
        esc.close();
        activeEventSource = null;
        statusEl.innerText = "FINISHED";
        statusEl.style.color = "var(--green)";
        statusEl.style.background = "rgba(0, 200, 83, 0.08)";
        
        if (btnEl) {
          btnEl.disabled = false;
          btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
        }
        if (btnAbort) {
          btnAbort.disabled = true;
        }
        if (window.lucide) window.lucide.createIcons();
        addLog(`Fine-tuning successfully completed for: ${voiceName}`, "success");
        loadTimbres();
      }, 500);
    }
  };

  esc.onerror = function (err) {
    console.error("EventSource connection error:", err);
    esc.close();
    activeEventSource = null;
    
    statusEl.innerText = "FINISHED";
    statusEl.style.color = "var(--green)";
    statusEl.style.background = "rgba(0, 200, 83, 0.08)";
    
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = '<i data-lucide="zap"></i> Start Fine-Tuning';
    }
    if (btnAbort) {
      btnAbort.disabled = true;
    }
    if (window.lucide) window.lucide.createIcons();
  };
}

const lrValues = ["1e-6", "5e-6", "1e-5", "2e-5", "5e-5", "1e-4"];

export function adjustFinetuneLR(direction) {
  const el = document.getElementById("finetune-lr");
  if (!el) return;
  let idx = lrValues.indexOf(el.value);
  if (idx === -1) idx = 2; // default 1e-5
  idx = Math.max(0, Math.min(lrValues.length - 1, idx + direction));
  el.value = lrValues[idx];
}

export function adjustFinetuneSteps(change) {
  const el = document.getElementById("finetune-steps");
  if (!el) return;
  let val = parseInt(el.value) || 120;
  val = Math.max(10, Math.min(2000, val + change));
  el.value = val;
}

export function adjustFinetuneRank(change) {
  const el = document.getElementById("finetune-rank");
  if (!el) return;
  let val = parseInt(el.value) || 16;
  val = Math.max(4, Math.min(128, val + change));
  el.value = val;
}

export function adjustFinetuneAlpha(change) {
  const el = document.getElementById("finetune-alpha");
  if (!el) return;
  let val = parseInt(el.value) || 16;
  val = Math.max(4, Math.min(128, val + change));
  el.value = val;
}

export function initVoices() {
  window.setVoiceMode = setVoiceMode;
  window.handleFileUpload = handleFileUpload;
  window.saveCurrentTimbre = saveCurrentTimbre;
  window.loadTimbre = loadTimbre;
  window.deleteTimbre = deleteTimbre;
  window.confirmSaveCurrentTimbre = confirmSaveCurrentTimbre;
  window.closeSaveVoiceModal = closeSaveVoiceModal;
  window.startVoiceFinetune = startVoiceFinetune;
  window.abortVoiceFinetune = abortVoiceFinetune;
  window.playSegmentAudio = playSegmentAudio;
  window.cancelFinetuneReview = cancelFinetuneReview;
  window.confirmAndStartFinetune = confirmAndStartFinetune;
  window.adjustFinetuneLR = adjustFinetuneLR;
  window.adjustFinetuneSteps = adjustFinetuneSteps;
  window.adjustFinetuneRank = adjustFinetuneRank;
  window.adjustFinetuneAlpha = adjustFinetuneAlpha;
}
