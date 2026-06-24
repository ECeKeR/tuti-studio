import { state } from "../state.js";
import { addLog, setStatus } from "./logger.js";
import { switchTab } from "./sidebar.js";
import { updateStitchBar, playTake, stopAll } from "./player.js";

let lastSavedDataStr = "";
let debounceTimer = null;

// ── GLOBAL MLX GENERATION LOCK (frontend tarafı) ──────────────────────
// MLX tek seferinde bir üretim yapabilir. Üretim devam ederken
// tüm Generate butonları devre dışı bırakılır.
let isGenerating = false;

export function setGeneratingState(busy) {
  isGenerating = busy;
  // Tüm generate butonlarını güncelle
  const btnGenerateTakes = document.getElementById("btn-generate-takes");
  const btnGenerateAll   = document.getElementById("btn-generate-all");
  const btnCompile       = document.getElementById("btn-compile");
  if (btnGenerateTakes) btnGenerateTakes.disabled = busy || (state.selectedSeg >= 0 && state.project?.segments[state.selectedSeg]?.status === "generating");
  if (btnGenerateAll)   btnGenerateAll.disabled   = busy;
  if (btnCompile)       btnCompile.disabled        = busy;
}

// ── CENTRAL GENERATION POLLING WATCHER ────────────────────────────────
// Tek bir poll döngüsü tüm generating segment'leri izler. Sunucu artık
// üretimi worker thread'de yapıp 202 ile hemen döndüğü için, frontend
// ilerlemeyi per-take polling ile takip eder.
//
// watchedSegments: { idx -> { takeCount } }  — take sayısı değiştiğinde
//   re-render tetiklenir, böylece çalan audio DOM'dan kopmaz.
// allDoneResolver: Generate All tamamlandığında resolve edilen Promise.
let pollTimer = null;
const watchedSegments = new Map();       // idx -> last seen take count
let allDoneResolver = null;              // { resolve, idxList }

function _renderSelectedIfGenerating() {
  // Sadece seçili segment generating ise ve take sayısı değiştiyse editor'ü
  // yeniden çiz → kademeli take kartları. Aynı take sayısında re-render yok
  // (çalan audio kesilmesin).
  if (state.selectedSeg < 0 || !state.project) return;
  const seg = state.project.segments[state.selectedSeg];
  if (!seg || seg.status !== "generating") return;
  const last = watchedSegments.get(state.selectedSeg);
  const cur = seg.takes ? seg.takes.length : 0;
  if (last !== cur) {
    watchedSegments.set(state.selectedSeg, cur);
    renderEditor();
  }
}

async function _pollTick() {
  try {
    const res = await fetch("/api/project/state");
    if (res.ok) {
      const fresh = await res.json();
      if (fresh && fresh.segments) {
        state.project = fresh;

        // Generating segment'lerin take sayısı değiştiyse editor'ü yeniden çiz
        _renderSelectedIfGenerating();

        // Tamamlanan/error olan watched segment'leri işle
        for (const idx of [...watchedSegments.keys()]) {
          const seg = fresh.segments[idx];
          if (!seg) { watchedSegments.delete(idx); continue; }
          if (seg.status === "completed") {
            addLog(`Chunk #${idx + 1} done — Take #${seg.selected_take + 1} auto-selected (score: ${seg.takes[seg.selected_take].score.toFixed(3)})`, "success");
            watchedSegments.delete(idx);
            if (idx === state.selectedSeg) renderEditor();
            renderSegments();
            updateStitchBar();
          } else if (seg.status === "error") {
            addLog(`Chunk #${idx + 1} error: ${seg.error_msg}`, "error");
            watchedSegments.delete(idx);
            if (idx === state.selectedSeg) renderEditor();
            renderSegments();
            updateStitchBar();
          } else if (seg.status === "generating") {
            // take sayısını güncelle (renderer bir sonraki tick'te kullanır)
            watchedSegments.set(idx, seg.takes ? seg.takes.length : 0);
          }
        }
      }
    }
  } catch (e) {
    console.warn("Poll tick failed:", e);
  }

  // Watcher boşaldıysa: poll'u durdur, UI'ı rahatlat, Generate All resolve
  if (watchedSegments.size === 0) {
    stopPolling();
    setGeneratingState(false);
    renderSegments();
    if (state.selectedSeg >= 0) renderEditor();
    updateStitchBar();
    const anyBusy = state.project?.segments.some(s => s.status === "generating" || s.status === "pending");
    if (!anyBusy) setStatus("active", `${state.project.segments.filter(s => s.status === "completed").length} / ${state.project.segments.length} done`);

    // Generate All bekleyen varsa resolve et
    if (allDoneResolver) {
      const r = allDoneResolver;
      allDoneResolver = null;
      r.resolve();
    }
    return;
  }

  // Devam: 900ms sonra tekrar
  pollTimer = setTimeout(_pollTick, 900);
}

export function startPolling(idx) {
  watchedSegments.set(idx, 0);
  // İlk poll tick'i hemen başlat (eğer zaten çalışmıyorsa)
  if (!pollTimer) {
    pollTimer = setTimeout(_pollTick, 400);  // 400ms ilk poll — worker'ın başlamasına izin ver
  }
}

export function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

// Generate All / compileProject için: tüm watched segment'ler bitene kadar bekle
export function watchSegmentsUntilDone(idxList) {
  for (const idx of idxList) watchedSegments.set(idx, 0);
  if (!pollTimer) {
    pollTimer = setTimeout(_pollTick, 400);
  }
  return new Promise((resolve) => {
    allDoneResolver = { resolve };
  });
}

export function debouncedAutoSave() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    autoSaveParams();
  }, 500);
}

export async function autoSaveParams() {
  if (state.selectedSeg < 0 || !state.project) return;
  
  const checkSeedLock = document.getElementById("check-seed-lock");
  const inputLockedSeed = document.getElementById("input-locked-seed");
  const lockedSeedVal = (checkSeedLock && checkSeedLock.checked && inputLockedSeed.value) 
    ? parseInt(inputLockedSeed.value, 10) 
    : null;

  const body = {
    idx: state.selectedSeg,
    text: document.getElementById("edit-text").value.trim(),
    stress: parseFloat(document.getElementById("slider-stress").value),
    speed: parseFloat(document.getElementById("slider-speed").value),
    pause_after: parseFloat(document.getElementById("slider-pause").value),
    tts_instruct: document.getElementById("edit-instruct").value.trim(),
    pitch: parseFloat(document.getElementById("slider-pitch").value),
    temperature: parseFloat(document.getElementById("slider-temperature").value),
    intonation_trend: document.getElementById("select-intonation").value,
    vowel_stretching: document.getElementById("select-vowel").value,
    thinking_enabled: document.getElementById("check-thinking").checked,
    locked_seed: lockedSeedVal,
    lora_strength: parseFloat(document.getElementById("slider-lora-strength").value)
  };

  const currentDataStr = JSON.stringify(body);
  if (currentDataStr === lastSavedDataStr) {
    return;
  }
  
  lastSavedDataStr = currentDataStr;

  const oldStatus = state.project.segments[state.selectedSeg]?.status;
  const oldTakesCount = state.project.segments[state.selectedSeg]?.takes?.length || 0;

  try {
    const res = await fetch("/api/segment/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: currentDataStr
    });
    const s = await res.json();
    state.project = s;
    renderSegments();
    
    const newSeg = state.project.segments[state.selectedSeg];
    if (newSeg) {
      if (newSeg.status !== oldStatus || (newSeg.takes?.length || 0) !== oldTakesCount) {
        renderTakes(newSeg);
        renderChunks(newSeg);
      }
    }
  } catch (e) {
    console.error("Auto-save failed:", e);
  }
}


export async function compileProject() {
  const text = document.getElementById("script-input").value.trim();
  if (!text) { addLog("Please paste a script first!", "warn"); return; }

  const isClone = document.getElementById("tab-clone").classList.contains("active");
  const isDesign = document.getElementById("tab-design").classList.contains("active");
  const speaker = document.getElementById("speaker-input").value;
  const model_size = isDesign
    ? document.getElementById("model-size-design-input").value
    : isClone
      ? document.getElementById("model-size-clone-input").value
      : document.getElementById("model-size-input").value;
  const tone = document.getElementById("tone-input").value;
  const n_takes = document.getElementById("takes-input").value;
  const backend = document.getElementById("backend-input").value;
  const ref_audio = isClone ? document.getElementById("ref-audio-input").value.trim() : "";
  const ref_text = isClone ? document.getElementById("ref-text-input").value.trim() : "";
  const voice_mode = isDesign ? "design" : isClone ? "clone" : "preset";
  const design_prompt = isDesign ? document.getElementById("design-prompt-input").value.trim() : "";
  const llm_backend = document.getElementById("llm-backend-input").value;
  const ollama_model = document.getElementById("ollama-model-input").value;

  // Clone-specific: Speaker Space Interpolation params
  const clone_alpha = isClone
    ? parseFloat(document.getElementById("clone-alpha-input").value)
    : 1.0;
  const clone_topk = isClone
    ? parseInt(document.getElementById("clone-topk-input").value, 10)
    : 3;
  const clone_speaker = isClone
    ? document.getElementById("clone-speaker-input").value
    : "ryan";

  setStatus("busy", "Compiling...");
  const btn = document.getElementById("btn-compile");
  btn.disabled = true;
  addLog("Building speech plan...");

  try {
    const isNew = !state.project || !!state.newProjectName;
    const url = isNew ? "/api/project/init" : "/api/project/append";

    const payloadStr = JSON.stringify({ 
      text, speaker, model_size, tone, n_takes, backend, ref_audio, ref_text, voice_mode, 
      design_prompt, llm_backend, ollama_model,
      clone_alpha, clone_topk, clone_speaker,
      project_name: state.newProjectName || "" 
    });
    addLog("Sending payload: " + payloadStr, "info");

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payloadStr
    });
    
    if (!res.ok) {
      const txt = await res.text();
      addLog(`HTTP ${res.status}: ${txt}`, "error");
      throw new Error(`HTTP ${res.status}`);
    }

    const s = await res.json();
    if (s.error) throw new Error(s.error);
    const oldLength = state.project ? state.project.segments.length : 0;
    state.project = s;
    state.newProjectName = null; // Clear custom name on success
    state.selectedSeg = oldLength; // select the first newly appended segment
    addLog(`Speech plan compiled: ${s.segments.length - oldLength} segments appended.`, "success");
    showWorkspace();
    setStatus("active", `${s.segments.length} segments`);
    loadProjectHistory();
    switchTab("studio");

    // Automatically batch generate pending segments
    const pending = state.project.segments.map((seg, i) => ({ seg, i })).filter(({ seg }) => seg.status !== "completed");
    if (pending.length > 0) {
      addLog(`Auto-generating ${pending.length} new segments...`);
      const pendingIds = pending.map(({ i }) => i);
      for (const i of pendingIds) {
        await generateSegment(i);
      }

      // Now wait for all of them to finish using the central poll watcher
      await watchSegmentsUntilDone(pendingIds);
      
      // Auto-stitch if all segments completed
      const allDone = state.project.segments.every(seg => seg.status === "completed");
      if (allDone) {
        addLog("All segments done — auto-stitching...", "success");
        try {
          const speedSlider = document.getElementById("slider-output-speed");
          const autoStitchSpeed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
          const stitchRes = await fetch("/api/stitch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ output_speed: autoStitchSpeed })
          });
          const stitchData = await stitchRes.json();
          if (stitchData.error) throw new Error(stitchData.error);
          state.project = stitchData;
          addLog("✓ Audio compiled & mastered! Ready to play.", "success");
          if (state.masterAudio) state.masterAudio.pause();
          state.masterAudio = new Audio(stitchData.final_audio_url);
          import("./player.js").then(m => m.bindPlayerEvents());
          import("./player.js").then(m => m.updateStitchBar());
          renderSegments();
        } catch (e) {
          addLog("Auto-stitch failed: " + e.message, "error");
        }
      }
    }
  } catch (e) {
    addLog("Compile failed: " + e.message, "error");
    setStatus("idle", "Error");
  } finally {
    setGeneratingState(false);
  }
}

export function showCreatePanel() {
  state.selectedSeg = -1;
  const editorPanel = document.getElementById("editor-panel");
  if (editorPanel) {
    editorPanel.classList.remove("visible");
    editorPanel.style.display = "none";
  }
  const emptyState = document.getElementById("empty-selection-state");
  if (emptyState) emptyState.style.display = "none";

  const createPanel = document.getElementById("create-panel");
  if (createPanel) createPanel.style.display = "flex";

  const ws = document.getElementById("workspace");
  if (ws) ws.style.display = "none";

  const bottomBar = document.getElementById("bottom-bar");
  if (bottomBar) bottomBar.style.display = "none";

  const sidebarActions = document.getElementById("sidebar-project-actions");
  if (sidebarActions) sidebarActions.style.display = "none";

  renderSegments();
}

export function showWorkspace() {
  document.getElementById("workspace").style.display = "flex";
  document.getElementById("bottom-bar").style.display = "flex";
  document.getElementById("sidebar-project-actions").style.display = "block";
  
  const createPanel = document.getElementById("create-panel");
  if (createPanel) createPanel.style.display = "none";

  // Set custom project/video title in workspace header
  const titleEl = document.getElementById("workspace-title");
  if (titleEl && state.project && state.project.name) {
    titleEl.textContent = state.project.name;
  } else if (titleEl) {
    titleEl.textContent = "Speech Segments";
  }

  renderSegments();
  if (state.selectedSeg >= 0) {
    renderEditor();
  } else {
    const editorPanel = document.getElementById("editor-panel");
    if (editorPanel) {
      editorPanel.classList.remove("visible");
      editorPanel.style.display = "none";
    }
    const emptyState = document.getElementById("empty-selection-state");
    if (emptyState) emptyState.style.display = "flex";
  }
  updateStitchBar();
}

export function renderSegments() {
  const area = document.getElementById("segments-area");
  if (!area) return;
  if (!state.project) {
    area.innerHTML = "";
    return;
  }
  area.innerHTML = "";
  state.project.segments.forEach((seg, i) => {
    const card = document.createElement("div");
    card.className = "seg-card " + seg.status + (i === state.selectedSeg ? " active" : "");
    card.onclick = () => selectSegment(i);

    let statusLabel = "Pending";
    let statusClass = "pending";
    if (seg.status === "generating") { statusLabel = "Generating..."; statusClass = "generating"; }
    else if (seg.status === "completed") { statusLabel = `Take #${seg.selected_take + 1} selected`; statusClass = "completed"; }
    else if (seg.status === "error") { statusLabel = "Error"; statusClass = "error"; }

    card.innerHTML = `
      <div class="seg-head">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="seg-num">Seg ${i + 1}</span>
          <span class="emotion-tag">${seg.emotion}</span>
          ${seg.status === "generating" ? `<div class="wave-anim"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div>` : ""}
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="status-tag ${statusClass}">${statusLabel}</span>
          <button class="btn btn-ghost btn-sm btn-icon" onclick="event.stopPropagation(); deleteSegment(${i})" style="padding: 2px; border: none; background: transparent; opacity: 0.5;" title="Delete segment">
            <i data-lucide="trash-2" style="width: 14px; height: 14px; color: var(--red);"></i>
          </button>
        </div>
      </div>
      <div class="seg-text">${seg.text}</div>
    `;
    area.appendChild(card);
  });

  // Append the "+" segment card
  const addCard = document.createElement("div");
  addCard.className = "seg-card add-seg-card";
  addCard.innerHTML = `
    <i data-lucide="plus" style="width: 14px; height: 14px; color: var(--text3);"></i>
    <span>Add New Segment</span>
  `;
  addCard.onclick = (e) => {
    e.stopPropagation();
    addNewSegment();
  };
  area.appendChild(addCard);

  if (window.lucide) window.lucide.createIcons();
}

export function renderEditor() {
  if (!state.project || state.selectedSeg < 0) {
    console.log("[Studio UI] renderEditor skipped: selectedSeg =", state.selectedSeg);
    return;
  }
  const seg = state.project.segments[state.selectedSeg];
  console.log("[Studio UI] renderEditor active for segment:", state.selectedSeg);
  
  document.getElementById("editor-panel").style.display = "flex";
  document.getElementById("editor-panel").classList.add("visible");
  const createPanel = document.getElementById("create-panel");
  if (createPanel) createPanel.style.display = "none";
  
  const emptyState = document.getElementById("empty-selection-state");
  if (emptyState) emptyState.style.display = "none";

  document.getElementById("editor-title").textContent = `Segment #${state.selectedSeg + 1}`;
  document.getElementById("editor-emotion").textContent = seg.emotion;
  document.getElementById("edit-text").value = seg.text;
  document.getElementById("edit-instruct").value = seg.tts_instruct;
  document.getElementById("slider-stress").value = seg.stress;
  document.getElementById("val-stress").textContent = Number(seg.stress).toFixed(2);
  document.getElementById("slider-speed").value = seg.speed;
  document.getElementById("val-speed").textContent = Number(seg.speed).toFixed(2) + "×";
  document.getElementById("slider-pause").value = seg.pause_after;
  document.getElementById("val-pause").textContent = Number(seg.pause_after).toFixed(2) + "s";

  const pitch = seg.pitch !== undefined ? seg.pitch : 0.0;
  document.getElementById("slider-pitch").value = pitch;
  document.getElementById("val-pitch").textContent = (pitch > 0 ? "+" : "") + Number(pitch).toFixed(2);

  const temp = seg.temperature !== undefined ? seg.temperature : 0.6;
  document.getElementById("slider-temperature").value = temp;
  document.getElementById("val-temperature").textContent = Number(temp).toFixed(2);

  const voiceMode = (state.project.config && state.project.config.voice_mode) || "preset";
  const isClone = voiceMode === "clone";

  const isSft = !!(state.project.config && state.project.config.is_sft);
  const loraStrengthWrap = document.getElementById("lora-strength-wrap");
  if (loraStrengthWrap) {
    loraStrengthWrap.style.display = (isSft && !isClone) ? "flex" : "none";
  }
  const loraStr = seg.lora_strength !== undefined ? seg.lora_strength : 0.75;
  const sliderLora = document.getElementById("slider-lora-strength");
  if (sliderLora) sliderLora.value = loraStr;
  const valLora = document.getElementById("val-lora-strength");
  if (valLora) valLora.textContent = Number(loraStr).toFixed(2);

  const instructWrap = document.getElementById("instruct-wrap");
  if (instructWrap) instructWrap.style.display = "block";

  const advSettingsWrap = document.getElementById("advanced-settings-wrap");
  if (advSettingsWrap) advSettingsWrap.style.display = "flex";

  document.getElementById("select-intonation").value = seg.intonation_trend || "stable";
  document.getElementById("select-vowel").value = seg.vowel_stretching || "normal";
  document.getElementById("check-thinking").checked = !!seg.thinking_enabled;

  const hasSeedLock = seg.locked_seed !== undefined && seg.locked_seed !== null;
  document.getElementById("check-seed-lock").checked = hasSeedLock;
  const seedInput = document.getElementById("input-locked-seed");
  seedInput.value = hasSeedLock ? seg.locked_seed : "";
  seedInput.disabled = !hasSeedLock;

  const isGenerating = seg.status === "generating";
  document.getElementById("btn-generate-takes").disabled = isGenerating;
  const btnSaveParams = document.getElementById("btn-save-params");
  if (btnSaveParams) btnSaveParams.disabled = isGenerating;

  // Set baseline for auto-save to avoid immediately triggering a save
  const baselineBody = {
    idx: state.selectedSeg,
    text: seg.text.trim(),
    stress: parseFloat(seg.stress),
    speed: parseFloat(seg.speed),
    pause_after: parseFloat(seg.pause_after),
    tts_instruct: seg.tts_instruct.trim(),
    pitch: parseFloat(pitch),
    temperature: parseFloat(temp),
    intonation_trend: seg.intonation_trend || "stable",
    vowel_stretching: seg.vowel_stretching || "normal",
    thinking_enabled: !!seg.thinking_enabled,
    locked_seed: hasSeedLock ? parseInt(seg.locked_seed, 10) : null,
    lora_strength: parseFloat(loraStr)
  };
  lastSavedDataStr = JSON.stringify(baselineBody);

  renderTakes(seg);
  renderChunks(seg);
}

// Take URL map — DOM attribute yerine JS'de sakla (encoding sorunlarını önler)
const takeUrlMap = {};

export function renderTakes(seg) {
  console.log("[Studio UI] renderTakes called for segment:", seg.idx, "takes count:", seg.takes ? seg.takes.length : 0, "status:", seg.status);
  const grid = document.getElementById("takes-grid");
  if (!grid) {
    console.warn("[Studio UI] takes-grid element not found!");
    return;
  }
  grid.innerHTML = "";
  if (!seg.takes || seg.takes.length === 0) {
    console.log("[Studio UI] No takes found for segment:", seg.idx);
    if (seg.status === "generating") {
      grid.innerHTML = `<div style="color:var(--text3);font-size:0.82rem;padding:8px 0">Generating takes — this may take 30-90s per segment…</div>`;
    } else {
      grid.innerHTML = `<div style="color:var(--text3);font-size:0.82rem;padding:8px 0">No takes yet. Click "Generate Takes" above.</div>`;
    }
    return;
  }

  seg.takes.forEach((take, i) => {
    const sel = seg.selected_take === i;
    const confPct = Math.round(take.metrics.pronunciation_confidence * 100);
    const energyPct = Math.round(take.metrics.energy * 100);
    const pitchPct = Math.round((take.metrics.pitch_variety || take.metrics.pitch_stability || 0) * 100);

    // URL'yi DOM'a değil JS map'e yaz — HTML encoding sorunlarını önler
    const key = `${state.selectedSeg}-${i}`;
    const projId = state.project?.id || 'temp';
    takeUrlMap[key] = take.audio_url || `/api/audio?path=pipeline_work/project_${projId}/seg_${String(state.selectedSeg).padStart(3,"0")}/take_${i}.wav`;

    const card = document.createElement("div");
    card.className = "take-card" + (sel ? " selected" : "");
    card.innerHTML = `
      <div class="take-head">
        <span class="take-title">Take #${i + 1}</span>
        <span class="take-score">Score: ${take.score.toFixed(3)}</span>
      </div>
      <div class="take-transcript">"${take.transcription}"</div>
      <div class="metric">
        <div class="metric-info"><span>Pronunciation</span><span>${confPct}%</span></div>
        <div class="metric-bar"><div class="metric-fill" style="width:${confPct}%"></div></div>
      </div>
      <div class="metric">
        <div class="metric-info"><span>Energy</span><span>${energyPct}%</span></div>
        <div class="metric-bar"><div class="metric-fill" style="width:${energyPct}%"></div></div>
      </div>
      <div class="metric">
        <div class="metric-info"><span>Pitch Variety</span><span>${pitchPct}%</span></div>
        <div class="metric-bar"><div class="metric-fill" style="width:${pitchPct}%"></div></div>
      </div>
      <div class="take-actions">
        <button class="play-btn" data-seg="${state.selectedSeg}" data-take="${i}">
          <i data-lucide="play" id="play-icon-${state.selectedSeg}-${i}"></i>
        </button>
        <button class="btn btn-ghost btn-sm" onclick="selectTake(${state.selectedSeg}, ${i})" ${sel ? "disabled" : ""}>
          ${sel ? "✓ Selected" : "Use This"}
        </button>
      </div>
    `;
    grid.appendChild(card);
  });

  // Event delegation — cloneNode yok, doğrudan listener ekle
  // Her render'da listener override edilmemesi için removeEventListener ile temizle
  if (grid._playHandler) {
    grid.removeEventListener("click", grid._playHandler);
  }
  grid._playHandler = (e) => {
    const btn = e.target.closest(".play-btn");
    if (!btn) return;
    const segIdx = parseInt(btn.dataset.seg, 10);
    const takeIdx = parseInt(btn.dataset.take, 10);
    const url = takeUrlMap[`${segIdx}-${takeIdx}`];
    if (!url) { console.warn("Take URL bulunamadı:", segIdx, takeIdx); return; }
    playTake(segIdx, takeIdx, url);
  };
  grid.addEventListener("click", grid._playHandler);

  if (window.lucide) window.lucide.createIcons();
}


export async function selectSegment(i) {
  stopAll();
  if (state.selectedSeg >= 0 && state.selectedSeg !== i) {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }
    await autoSaveParams();
  }
  state.selectedSeg = i;
  renderSegments();
  renderEditor();
}

export async function deleteSegment(idx) {
  try {
    const res = await fetch(`/api/segment/delete?idx=${idx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idx })
    });
    const s = await res.json();
    if (s.error) throw new Error(s.error);
    state.project = s;
    if (state.selectedSeg === idx) {
      if (state.project.segments && state.project.segments.length > 0) {
        state.selectedSeg = Math.min(idx, state.project.segments.length - 1);
      } else {
        state.selectedSeg = -1;
      }
    } else if (state.selectedSeg > idx) {
      state.selectedSeg -= 1;
    }
    addLog(`Segment #${idx + 1} deleted.`, "success");
    renderSegments();
    if (state.selectedSeg >= 0) {
      renderEditor();
    } else {
      const editorPanel = document.getElementById("editor-panel");
      if (editorPanel) {
        editorPanel.classList.remove("visible");
        editorPanel.style.display = "none";
      }
      const emptyState = document.getElementById("empty-selection-state");
      if (emptyState) emptyState.style.display = "flex";
    }
    updateStitchBar();
  } catch (e) {
    addLog("Delete failed: " + e.message, "error");
  }
}

export async function generateSegment(idx) {
  if (!state.project) return;
  const seg = state.project.segments[idx];
  if (seg.status === "generating") return;

  // Set status and disable buttons IMMEDIATELY to prevent double clicks/race conditions
  state.project.segments[idx].status = "generating";
  setStatus("busy", "Generating...");
  setGeneratingState(true);
  renderSegments();
  if (idx === state.selectedSeg) renderEditor();

  // Force immediate save of any pending changes first!
  if (debounceTimer) {
    clearTimeout(debounceTimer);
    debounceTimer = null;
  }
  await autoSaveParams();

  addLog(`Generating ${state.project.config.n_takes} takes for Chunk #${idx + 1}...`);

  try {
    // Sunucu artık job'u worker queue'ya koyup 202 ile hemen döner.
    // Üretim arka planda sürer; ilerleme pollGeneration ile takip edilir.
    const res = await fetch("/api/segment/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idx })
    });
    if (res.status === 409) {
      // Zaten generating — sessizce kabul et, polling devam etsin
      addLog(`Chunk #${idx + 1} already generating.`, "warn");
      startPolling(idx);
      return;
    }
    if (res.status === 503) {
      // Artık sunucu 503 dönmüyor (queue modeli), ama yine de handle et
      const errData = await res.json().catch(() => ({}));
      addLog(`MLX meşgul: ${errData.error || "Başka bir segment üretiliyor"}`, "warn");
      state.project.segments[idx].status = "pending";
      setGeneratingState(false);
      return;
    }
    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      const errMsg = errData ? (errData.error || errData.error_msg) : null;
      throw new Error(errMsg || `HTTP ${res.status}`);
    }

    // 202 accepted: state'i güncelle ve poll watcher'ı başlat
    const s = await res.json();
    state.project = s;

    // startPolling — watcher tüm bitenleri loglar, UI günceller, finally'de
    // setGeneratingState(false) yapar.
    startPolling(idx);
  } catch (e) {
    addLog(`Generate API error: ${e.message}`, "error");
    state.project.segments[idx].status = "error";
    setGeneratingState(false);
    renderSegments();
    if (idx === state.selectedSeg) renderEditor();
    updateStitchBar();
  }
}

export async function addNewSegment() {
  if (!state.project) return;
  addLog("Adding new segment...", "info");
  try {
    const res = await fetch("/api/segment/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const s = await res.json();
    if (s.error) throw new Error(s.error);
    state.project = s;
    const newIdx = s.segments.length - 1;
    state.selectedSeg = newIdx; // Auto-select the newly created segment
    addLog(`✓ Blank Segment #${newIdx + 1} added.`, "success");
    renderSegments();
    renderEditor();
    updateStitchBar();
  } catch (e) {
    addLog("Failed to add segment: " + e.message, "error");
  }
}

export async function selectTake(segIdx, takeIdx) {
  try {
    const res = await fetch("/api/segment/select_take", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seg_idx: segIdx, take_idx: takeIdx })
    });
    const s = await res.json();
    state.project = s;
    addLog(`Chunk #${segIdx + 1} → Take #${takeIdx + 1} selected.`, "success");
    renderSegments();
    renderEditor();
    updateStitchBar();
  } catch (e) { addLog("Selection failed: " + e.message, "error"); }
}

export function renderChunks(seg) {
  const section = document.getElementById("chunks-section");
  const container = document.getElementById("chunks-container");
  if (!section || !container) return;
  container.innerHTML = "";

  if (!seg.chunks || seg.chunks.length === 0 || !seg.takes || seg.takes.length === 0) {
    if (seg.status === "generating") {
      section.style.display = "block";
      container.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px; font-size:0.82rem; color:var(--neon-blue); font-weight:500; padding:8px 0;">
          <div class="wave-anim" style="margin:0"><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div></div>
          ⚡ Processing Chunks & Aligning Takes...
        </div>
      `;
    } else {
      section.style.display = "none";
    }
    return;
  }

  section.style.display = "block";
  seg.chunks.forEach((chunk) => {
    const chip = document.createElement("div");
    chip.className = "chunk-chip";
    
    let buttonsHtml = "";
    for (let t = 0; t < seg.takes.length; t++) {
      const isActive = chunk.active_take === t;
      const isOverridden = chunk.override_take === t;
      buttonsHtml += `
        <button class="mini-btn ${isActive ? "active" : ""} ${isOverridden ? "overridden" : ""}" 
                onclick="overrideChunk(${state.selectedSeg}, ${chunk.idx}, ${t})" 
                title="Force Take #${t + 1}">
          T${t + 1}
        </button>
      `;
    }

    const isAutoActive = chunk.override_take === null || chunk.override_take === undefined;
    buttonsHtml += `
      <button class="mini-btn auto-btn ${isAutoActive ? "active" : ""}" 
              onclick="overrideChunk(${state.selectedSeg}, ${chunk.idx}, null)" 
              title="Reset to Auto (Viterbi path)">
        Auto
      </button>
    `;

    chip.innerHTML = `
      <span class="chunk-text-span">${chunk.text}</span>
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:4px;">
        <span style="font-size:0.68rem; color:var(--text3);">Take:</span>
        <div class="mini-btn-group">
          ${buttonsHtml}
        </div>
      </div>
    `;
    container.appendChild(chip);
  });
}

export async function overrideChunk(segIdx, chunkIdx, takeIdx) {
  try {
    const res = await fetch("/api/segment/override_chunk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seg_idx: segIdx, chunk_idx: chunkIdx, take_idx: takeIdx })
    });
    const data = await res.json();
    if (data.error) {
      addLog(data.error, "error");
      return;
    }
    state.project = data;
    addLog(`Prosodic chunk #${chunkIdx + 1} updated.`, "success");
    renderSegments();
    renderEditor();
    updateStitchBar();
  } catch (e) {
    addLog("Chunk override failed: " + e.message, "error");
  }
}

export async function loadProjectHistory() {
  const list = document.getElementById("project-history-list");
  if (!list) return;
  try {
    const res = await fetch("/api/project/list");
    const d = await res.json();
    if (!d.projects || d.projects.length === 0) {
      list.innerHTML = `<div style="font-size:0.72rem;color:var(--text3);padding:8px 0;">No past sessions found.</div>`;
      return;
    }
    list.innerHTML = "";
    d.projects.forEach(p => {
      const el = document.createElement("div");
      el.className = "project-history-card";
      el.style.cssText = "display:flex; justify-content:space-between; align-items:center; padding:6px 10px; border:1px solid var(--border); border-radius:6px; background:rgba(255,255,255,0.01); gap:8px; width:100%;";
      
      const date = new Date(p.updated_at).toLocaleDateString(undefined, {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"});
      
      el.innerHTML = `
        <div style="flex:1; min-width:0; cursor:pointer;" onclick="loadProjectSession(${p.id})">
          <div style="font-size:0.78rem; color:var(--text); font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${p.name}</div>
          <div style="font-size:0.62rem; color:var(--text3);">${date}</div>
        </div>
        <button class="btn btn-ghost btn-sm btn-icon" onclick="deleteProjectSession(event, ${p.id})" style="padding:2px; border:none; background:transparent; opacity:0.4;" title="Delete session">
          <i data-lucide="trash-2" style="width:11px; height:11px; color:var(--red);"></i>
        </button>
      `;
      list.appendChild(el);
    });
    if (window.lucide) window.lucide.createIcons();
  } catch (e) {
    console.error("Failed to load project history:", e);
  }
}

export async function loadProjectSession(id) {
  addLog(`Loading project session #${id}...`);
  try {
    const res = await fetch(`/api/project/load?id=${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id })
    });
    const s = await res.json();
    if (s.error) throw new Error(s.error);
    state.project = s;
    state.selectedSeg = 0;
    addLog(`Loaded project "${s.config.speaker} — ${s.segments.length} segments".`, "success");
    showWorkspace();
    setStatus("active", `${s.segments.length} segments`);
  } catch (e) {
    addLog("Load failed: " + e.message, "error");
  }
}

export async function deleteProjectSession(event, id) {
  event.stopPropagation();
  try {
    const res = await fetch(`/api/project/delete?id=${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id })
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    addLog("Project session deleted.", "success");
    loadProjectHistory();
    
    // If the active project is deleted, reset the active workspace
    if (state.project && state.project.id === id) {
      state.project = null;
      state.selectedSeg = -1;
      showCreatePanel();
      document.getElementById("bottom-bar").style.display = "none";
      setStatus("idle", "Idle");
    }
  } catch (e) {
    addLog("Delete failed: " + e.message, "error");
  }
}

export function showNewProjectModal() {
  const modal = document.getElementById("new-project-modal");
  const nameInput = document.getElementById("new-project-name-input");
  if (modal && nameInput) {
    nameInput.value = "";
    modal.style.display = "flex";
    nameInput.focus();
  }
}

export function closeNewProjectModal() {
  const modal = document.getElementById("new-project-modal");
  if (modal) {
    modal.style.display = "none";
  }
}

export async function confirmCreateProject() {
  const nameInput = document.getElementById("new-project-name-input");
  if (!nameInput) return;
  const name = nameInput.value.trim();
  if (!name) {
    addLog("Please enter a video/project name.", "warn");
    return;
  }
  
  state.project = null; // Clear active project state
  state.selectedSeg = -1;
  state.newProjectName = name;
  closeNewProjectModal();
  
  // Clear the script field and focus it
  const scriptInput = document.getElementById("script-input");
  if (scriptInput) {
    scriptInput.value = "";
    scriptInput.placeholder = `Write narration script for "${name}" here...`;
    scriptInput.focus();
  }
  
  showCreatePanel();
  addLog(`New video project created: "${name}". Write the script and compile the speech plan.`, "info");
}

export function initStudio() {
  window.selectSegment = selectSegment;
  window.selectTake = selectTake;
  window.overrideChunk = overrideChunk;
  window.loadProjectSession = loadProjectSession;
  window.deleteProjectSession = deleteProjectSession;
  window.deleteSegment = deleteSegment;
  window.showCreatePanel = showCreatePanel;
  window.showNewProjectModal = showNewProjectModal;
  window.closeNewProjectModal = closeNewProjectModal;
  window.confirmCreateProject = confirmCreateProject;
  window.addNewSegment = addNewSegment;
}
