import { state } from "./state.js";
import { setStatus, addLog } from "./components/logger.js";
import { initSidebar, switchTab } from "./components/sidebar.js";
import { initSettings } from "./components/settings.js";
import { initVoices, loadTimbres } from "./components/voices.js";
import { initStudio, compileProject, showWorkspace, showCreatePanel, renderSegments, renderEditor, generateSegment, loadProjectHistory, autoSaveParams, debouncedAutoSave, watchSegmentsUntilDone } from "./components/studio.js";
import { initPlayer, updateStitchBar, bindPlayerEvents, stopAll } from "./components/player.js";
import { initModels } from "./components/models.js";

/* ═══════════════════════════════════════════════════════
   INIT & DOM CONTENT LOADED
   ═══════════════════════════════════════════════════════ */
window.addEventListener("DOMContentLoaded", () => {
  // Detect macOS and add class to body for custom title bar spacing
  const isMac = navigator.userAgent.toLowerCase().includes('mac') || navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  if (isMac) {
    document.body.classList.add("mac-os");
  }

  if (window.lucide) {
    window.lucide.createIcons();
  }

  // Initialize components and mount their handlers to window
  initSidebar();
  initSettings();
  initVoices();
  initStudio();
  initPlayer();
  initModels();

  // Bind local controls
  bindSliders();
  bindButtons();

  // Load server state and load timbres list
  loadServerState();
  loadTimbres();
  loadProjectHistory();

  // Default to studio tab
  switchTab("studio");
});

/* ═══════════════════════════════════════════════════════
   SLIDERS BINDING
   ═══════════════════════════════════════════════════════ */
function bindSliders() {
  const stress = document.getElementById("slider-stress");
  if (stress) {
    stress.addEventListener("input", e => {
      const val = document.getElementById("val-stress");
      if (val) val.textContent = Number(e.target.value).toFixed(2);
    });
    stress.addEventListener("change", autoSaveParams);
  }

  const speed = document.getElementById("slider-speed");
  if (speed) {
    speed.addEventListener("input", e => {
      const val = document.getElementById("val-speed");
      if (val) val.textContent = Number(e.target.value).toFixed(2) + "×";
    });
    speed.addEventListener("change", autoSaveParams);
  }

  const pitch = document.getElementById("slider-pitch");
  if (pitch) {
    pitch.addEventListener("input", e => {
      const val = document.getElementById("val-pitch");
      if (val) {
        const num = Number(e.target.value);
        val.textContent = (num > 0 ? "+" : "") + num.toFixed(2);
      }
    });
    pitch.addEventListener("change", autoSaveParams);
  }

  const temp = document.getElementById("slider-temperature");
  if (temp) {
    temp.addEventListener("input", e => {
      const val = document.getElementById("val-temperature");
      if (val) val.textContent = Number(e.target.value).toFixed(2);
    });
    temp.addEventListener("change", autoSaveParams);
  }

  const pause = document.getElementById("slider-pause");
  if (pause) {
    pause.addEventListener("input", e => {
      const val = document.getElementById("val-pause");
      if (val) val.textContent = Number(e.target.value).toFixed(2) + "s";
    });
    pause.addEventListener("change", autoSaveParams);
  }

  const loraStrength = document.getElementById("slider-lora-strength");
  if (loraStrength) {
    loraStrength.addEventListener("input", e => {
      const val = document.getElementById("val-lora-strength");
      if (val) val.textContent = Number(e.target.value).toFixed(2);
    });
    loraStrength.addEventListener("change", autoSaveParams);
  }

  const outputSpeed = document.getElementById("slider-output-speed");
  if (outputSpeed) {
    outputSpeed.addEventListener("input", e => {
      const val = document.getElementById("val-output-speed");
      if (val) val.textContent = Number(e.target.value).toFixed(2) + "×";
    });
  }

  // Bind dropdowns and other inputs
  const selectIntonation = document.getElementById("select-intonation");
  if (selectIntonation) selectIntonation.addEventListener("change", autoSaveParams);

  const selectVowel = document.getElementById("select-vowel");
  if (selectVowel) selectVowel.addEventListener("change", autoSaveParams);

  const checkThinking = document.getElementById("check-thinking");
  if (checkThinking) checkThinking.addEventListener("change", autoSaveParams);

  const editText = document.getElementById("edit-text");
  if (editText) {
    editText.addEventListener("input", debouncedAutoSave);
    editText.addEventListener("change", autoSaveParams);
    editText.addEventListener("blur", autoSaveParams);
  }

  const editInstruct = document.getElementById("edit-instruct");
  if (editInstruct) {
    editInstruct.addEventListener("input", debouncedAutoSave);
    editInstruct.addEventListener("change", autoSaveParams);
    editInstruct.addEventListener("blur", autoSaveParams);
  }
}


/* ═══════════════════════════════════════════════════════
   BUTTON BINDINGS & ORCHESTRATION
   ═══════════════════════════════════════════════════════ */
function bindButtons() {
  // Compile project action
  const btnCompile = document.getElementById("btn-compile");
  if (btnCompile) {
    btnCompile.addEventListener("click", compileProject);
  }

  // Generate takes for selected segment
  const btnGenTakes = document.getElementById("btn-generate-takes");
  if (btnGenTakes) {
    btnGenTakes.addEventListener("click", () => {
      if (state.selectedSeg < 0) return;
      generateSegment(state.selectedSeg);
    });
  }

  // Generate all segments action (Batch generate)
  const btnGenerateAll = document.getElementById("btn-generate-all");
  if (btnGenerateAll) {
    btnGenerateAll.addEventListener("click", async () => {
      if (!state.project) return;
      btnGenerateAll.disabled = true;
      const pending = state.project.segments.map((seg, i) => ({ seg, i })).filter(({ seg }) => seg.status !== "completed");
      addLog(`Batch enqueuing ${pending.length} segments...`);

      // Auto-save currently active inputs first
      await autoSaveParams();

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
          const stitchSpeedSlider = document.getElementById("slider-output-speed");
          const stitchSpeed = stitchSpeedSlider ? parseFloat(stitchSpeedSlider.value) : 1.0;
          const stitchRes = await fetch("/api/stitch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ output_speed: stitchSpeed })
          });
          const stitchData = await stitchRes.json();
          if (stitchData.error) throw new Error(stitchData.error);
          state.project = stitchData;
          addLog("✓ Audio compiled & mastered! Ready to play.", "success");
          if (state.masterAudio) state.masterAudio.pause();
          state.masterAudio = new Audio(stitchData.final_audio_url);
          bindPlayerEvents();
          updateStitchBar();
          renderSegments();
        } catch (e) {
          addLog("Auto-stitch failed: " + e.message, "error");
        }
      }
      btnGenerateAll.disabled = false;
    });
  }


  // Close editor panel
  const btnCloseEditor = document.getElementById("btn-close-editor");
  if (btnCloseEditor) {
    btnCloseEditor.addEventListener("click", () => {
      state.selectedSeg = -1;
      const editorPanel = document.getElementById("editor-panel");
      if (editorPanel) {
        editorPanel.classList.remove("visible");
        editorPanel.style.display = "none";
      }
      const emptyState = document.getElementById("empty-selection-state");
      if (emptyState) emptyState.style.display = "flex";
      renderSegments();
    });
  }

  // Save parameters fallback (button is hidden but kept for event compatibility)
  const btnSaveParams = document.getElementById("btn-save-params");
  if (btnSaveParams) {
    btnSaveParams.addEventListener("click", autoSaveParams);
  }

  // Bind seed lock checkbox
  const checkSeedLock = document.getElementById("check-seed-lock");
  const inputLockedSeed = document.getElementById("input-locked-seed");
  if (checkSeedLock && inputLockedSeed) {
    checkSeedLock.addEventListener("change", e => {
      inputLockedSeed.disabled = !e.target.checked;
      if (e.target.checked && !inputLockedSeed.value) {
        inputLockedSeed.value = "42"; // Default seed
      }
      autoSaveParams();
    });
    inputLockedSeed.addEventListener("input", debouncedAutoSave);
    inputLockedSeed.addEventListener("change", autoSaveParams);
  }

  // Bind paralinguistic injectors
  document.querySelectorAll(".btn-inject").forEach(btn => {
    btn.addEventListener("click", e => {
      e.preventDefault();
      const injectVal = btn.getAttribute("data-inject");
      const textarea = document.getElementById("edit-text");
      if (textarea && injectVal) {
        const start = textarea.selectionStart;
        const end = textarea.selectionEnd;
        const val = textarea.value;
        textarea.value = val.substring(0, start) + injectVal + val.substring(end);
        textarea.focus();
        textarea.selectionStart = textarea.selectionEnd = start + injectVal.length;
        autoSaveParams();
      }
    });
  });

  // Reset/New project action
  document.querySelectorAll(".action-new-project").forEach(btn => {
    btn.addEventListener("click", () => {
      if (window.showNewProjectModal) {
        window.showNewProjectModal();
      }
    });
  });

  // Manual Stitch and Master action
  const btnStitch = document.getElementById("btn-stitch");
  if (btnStitch) {
    btnStitch.addEventListener("click", async () => {
      const speedSlider = document.getElementById("slider-output-speed");
      const outputSpeed = speedSlider ? parseFloat(speedSlider.value) : 1.0;
      const speedLabel = outputSpeed !== 1.0 ? ` @ ${outputSpeed.toFixed(2)}× speed` : "";
      addLog(`Stitching: zero-crossing cut + pitch anchor + LUFS mastering${speedLabel}...`);
      btnStitch.disabled = true;
      try {
        const res = await fetch("/api/stitch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ output_speed: outputSpeed })
        });
        const s = await res.json();
        if (s.error) throw new Error(s.error);
        state.project = s;
        addLog(`Audio compiled and mastered${speedLabel}! Ready to play.`, "success");
        if (state.masterAudio) state.masterAudio.pause();
        state.masterAudio = new Audio(s.final_audio_url);
        bindPlayerEvents();
        updateStitchBar();
        renderSegments();
      } catch (e) {
        addLog("Stitch failed: " + e.message, "error");
        btnStitch.disabled = false;
      }
    });
  }

  // Project Download & Export action
  const btnDownload = document.getElementById("btn-download");
  if (btnDownload) {
    btnDownload.addEventListener("click", async (e) => {
      e.preventDefault(); // Default link download'unu engelle
      addLog("Exporting files... Requesting target folder selection.");
      try {
        // 1. Wails Go API üzerinden klasör seçtir
        const dirRes = await fetch("/api/wails/select_folder");
        const dirData = await dirRes.json();
        if (dirData.error) throw new Error(dirData.error);

        const exportDir = dirData.path;
        if (!exportDir) {
          addLog("Export cancelled by user.", "info");
          return;
        }

        addLog(`Folder selected: ${exportDir}. Exporting takes and master...`);

        // 2. Python backend export API'sini çağır
        const exportRes = await fetch("/api/project/export", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ export_dir: exportDir })
        });

        const exportData = await exportRes.json();
        if (exportData.error) throw new Error(exportData.error);

        addLog(`✓ Project successfully exported to: ${exportData.export_dir}`, "success");
        alert(`Tebrikler! Proje başarıyla şu klasöre dışa aktarıldı:\n${exportData.export_dir}`);
      } catch (err) {
        addLog("Export failed: " + err.message, "error");
      }
    });
  }
}

/* ═══════════════════════════════════════════════════════
   SERVER CONNECTION / STATE RESUME
   ═══════════════════════════════════════════════════════ */
async function loadServerState() {
  setStatus("busy", "Connecting...");
  addLog("Connecting to AI Voice Engine...");

  let retries = 30; // 30 retries * 500ms = 15 seconds max wait time
  while (retries > 0) {
    try {
      const res = await fetch("/api/project/state");
      if (res.ok) {
        const s = await res.json();
        addLog("Successfully connected to AI Voice Compiler Engine.", "success");
        setStatus("active", "Connected");
        if (s && s.id) {
          state.project = s;
          showWorkspace();
        } else {
          state.project = null;
          showCreatePanel();
        }
        return;
      }
    } catch (e) {
      // Keep waiting
    }
    retries--;
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  setStatus("idle", "Engine Offline");
  addLog("Failed to connect to AI Voice Engine. Please make sure the Python server is running.", "error");
}

/* ═══════════════════════════════════════════════════════
   CUSTOM VOICE PIPELINE SIMULATION
   ═══════════════════════════════════════════════════════ */
export function simulateCustomVoicePipeline() {
  const btn = document.getElementById("btn-simulate-pipeline");
  const term = document.getElementById("sim-terminal");
  const status = document.getElementById("sim-status");
  if (!term || !status || !btn) return;

  if (btn.disabled) return;
  btn.disabled = true;
  status.textContent = "RUNNING";
  status.style.background = "rgba(255, 179, 0, 0.15)";
  status.style.color = "var(--amber)";
  term.innerHTML = "";

  const highlightCard = (stepIdx) => {
    for (let i = 1; i <= 5; i++) {
      const card = document.getElementById(`cv-step-${i}`);
      if (!card) continue;
      if (i === stepIdx + 1) {
        card.classList.add("active-stage");
        card.classList.remove("done-stage");
      } else if (i < stepIdx + 1) {
        card.classList.add("done-stage");
        card.classList.remove("active-stage");
      } else {
        card.classList.remove("active-stage", "done-stage");
      }
    }
  };

  const writeLine = (text, type = "info") => {
    const line = document.createElement("div");
    line.className = `log-line ${type}`;
    if (type === "success") line.style.color = "var(--green)";
    else if (type === "error") line.style.color = "var(--red)";
    else if (type === "warn") line.style.color = "var(--amber)";
    else line.style.color = "#a5d6ff";

    line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
  };

  const logQueue = [
    { delay: 100, action: () => { highlightCard(0); writeLine("Pipeline initiated. Scanning raw audio files...", "info"); } },
    { delay: 800, action: () => writeLine("ham_kayit_27s.wav loaded. (Noise Ratio: High SNR, Mono)", "info") },
    { delay: 1500, action: () => writeLine("Whisper word-level timestamp segmentation started...", "info") },
    {
      delay: 2200, action: () => {
        writeLine("✓ Segment #1: [0.00s - 5.12s] -> 'Technical roadmap is below.'", "success");
        writeLine("✓ Segment #2: [5.12s - 12.04s] -> 'This guide walks you through shifting...'", "success");
        writeLine("✓ Segment #3: [12.04s - 18.50s] -> 'Stage one: data engineering...'", "success");
        writeLine("✓ Segment #4: [18.50s - 27.00s] -> 'Lightweight fine-tuning parameters.'", "success");
      }
    },
    { delay: 3000, action: () => writeLine("Calling Ollama Ground Truth cleaner (model: qwen3:8b)...", "info") },
    { delay: 4000, action: () => writeLine("✓ Ollama: Raw transcript aligned with original script. Filler words ('um/uh') cleaned. Training texts created.", "success") },

    { delay: 4800, action: () => { highlightCard(1); writeLine("Stage 2: Qwen-TTS-Tokenizer-12Hz audio tokenization started...", "info"); } },
    { delay: 5500, action: () => writeLine("Converting waveform to discrete audio token codes (at 12Hz rate)...", "info") },
    { delay: 6300, action: () => writeLine("✓ Discrete audio tokens generated for 4 segments. Writing JSONL dataset in ChatML format...", "success") },
    { delay: 7000, action: () => writeLine("Training data augmentation: Instruction modifiers [laughter, breath, pitch] injected.", "info") },

    { delay: 7800, action: () => { highlightCard(2); writeLine("Stage 3: Apple Silicon MLX LoRA Fine-Tuning initiated...", "info"); } },
    { delay: 8400, action: () => writeLine("Loaded Base Model: Qwen3-TTS-12Hz-1.7B-Base", "info") },
    { delay: 9000, action: () => writeLine("Training parameters: Learning Rate = 2e-5, Rank = 16, Steps = 200", "warn") },
    { delay: 9800, action: () => writeLine("Target modules mapped: q_proj, v_proj, speaker_embedding", "info") },
    { delay: 10500, action: () => writeLine("Training Step 50/200 - Loss: 2.381 - Step Time: 124ms", "info") },
    { delay: 11200, action: () => writeLine("Training Step 100/200 - Loss: 1.642 - Step Time: 119ms", "info") },
    { delay: 11900, action: () => writeLine("Training Step 150/200 - Loss: 1.104 - Step Time: 121ms", "info") },
    { delay: 12600, action: () => writeLine("Training Step 200/200 - Loss: 0.762 - Step Time: 118ms", "info") },
    { delay: 13300, action: () => writeLine("✓ Training complete. LoRA adapter weights saved: 'pipeline_work/mytrainedvoice_lora.safetensors'", "success") },

    { delay: 14100, action: () => { highlightCard(3); writeLine("Stage 4: Checking CustomVoice integration (Inference Bridge)...", "info"); } },
    { delay: 14800, action: () => writeLine("PEFT adapter merged onto Base model.", "info") },
    { delay: 15500, action: () => writeLine("✓ Model now recognizes 'MyTrainedVoice' as a target adaptation speaker. Verification passed.", "success") },

    { delay: 16300, action: () => { highlightCard(4); writeLine("Stage 5: Quality Control (Acoustic Matching) applied...", "info"); } },
    { delay: 17000, action: () => writeLine("High-pass filter calibrated (>85Hz) to cut background hum & room reverb.", "info") },
    { delay: 17800, action: () => writeLine("✓ Pitch boundaries validated. Out-of-bounds prosodic values normalized.", "success") },
    {
      delay: 18500, action: () => {
        highlightCard(5);
        status.textContent = "COMPLETED";
        status.style.background = "rgba(0, 230, 118, 0.15)";
        status.style.color = "var(--green)";
        btn.disabled = false;
        writeLine("✓ CONGRATULATIONS! Target-Speaker Adaptation pipeline completed successfully.", "success");
        writeLine("Voice Mode: 'MyTrainedVoice' is now ready for use! (Secret Sauce parameters enabled)", "success");
      }
    }
  ];

  logQueue.forEach(item => {
    setTimeout(item.action, item.delay);
  });
}

window.simulateCustomVoicePipeline = simulateCustomVoicePipeline;
