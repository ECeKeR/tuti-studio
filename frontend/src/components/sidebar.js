import { state } from "../state.js";

export function updateSettingsRecap() {
  const recapVoice = document.getElementById("recap-voice");
  const recapBackend = document.getElementById("recap-backend");
  const recapTakes = document.getElementById("recap-takes");
  
  if (!recapVoice || !recapBackend || !recapTakes) return;

  const isClone = document.getElementById("tab-clone").classList.contains("active");
  const isDesign = document.getElementById("tab-design").classList.contains("active");
  const speaker = document.getElementById("speaker-input").value;
  const backend = document.getElementById("backend-input").value;
  const n_takes = document.getElementById("takes-input").value;
  const tone = document.getElementById("tone-input").value;

  let voiceStr = "";
  if (isDesign) {
    voiceStr = "Designed Voice";
  } else if (isClone) {
    voiceStr = "Cloned Voice";
  } else {
    const speakerEl = document.getElementById("speaker-input");
    const selectedOpt = speakerEl && speakerEl.options[speakerEl.selectedIndex];
    const isSFT = selectedOpt && selectedOpt.getAttribute("data-sft") === "true";
    
    const sftContainer = document.getElementById("sft-status-container");
    const sftActiveName = document.getElementById("sft-active-name");
    
    if (isSFT) {
      if (sftContainer && sftActiveName) {
        sftContainer.style.display = "flex";
        sftActiveName.innerText = speaker;
      }
      voiceStr = `SFT: ${speaker}`;
      
      // Auto-update preset model size input to custom voice model if needed
      const modelSizeInput = document.getElementById("model-size-input");
      if (modelSizeInput) {
        const modelSizeAttr = selectedOpt.getAttribute("data-model");
        const customModelId = modelSizeAttr && modelSizeAttr.startsWith("mlx-community/") 
          ? modelSizeAttr 
          : (modelSizeAttr === "0.6B"
              ? "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
              : "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16");
        if (modelSizeInput.value !== customModelId) {
          modelSizeInput.value = customModelId;
        }
      }
    } else {
      if (sftContainer) {
        sftContainer.style.display = "none";
      }
      const capSpeaker = speaker.charAt(0).toUpperCase() + speaker.slice(1);
      voiceStr = `Preset: ${capSpeaker}`;
    }
  }

  recapVoice.textContent = voiceStr;
  recapBackend.textContent = backend.toUpperCase() + " Engine";
  recapTakes.textContent = `${n_takes} Takes / ${tone.charAt(0).toUpperCase() + tone.slice(1)}`;
}

export function switchTab(tab) {
  state.activeTab = tab;
  
  // Toggle nav buttons active state
  document.getElementById("nav-studio").classList.toggle("active", tab === "studio");
  document.getElementById("nav-voices").classList.toggle("active", tab === "voices");
  document.getElementById("nav-settings").classList.toggle("active", tab === "settings");
  const navModels = document.getElementById("nav-models");
  if (navModels) navModels.classList.toggle("active", tab === "models");

  // Toggle tab contents
  document.getElementById("tab-content-studio").style.display = tab === "studio" ? "flex" : "none";
  document.getElementById("tab-content-voices").style.display = tab === "voices" ? "block" : "none";
  document.getElementById("tab-content-settings").style.display = tab === "settings" ? "block" : "none";
  const tabModels = document.getElementById("tab-content-models");
  if (tabModels) tabModels.style.display = tab === "models" ? "block" : "none";

  if (tab === "studio") {
    updateSettingsRecap();
  } else if (tab === "models") {
    if (window.loadModelsStatus) {
      window.loadModelsStatus();
    }
  }
}

export function initSidebar() {
  window.switchTab = switchTab;
  
  const list = ["speaker-input", "backend-input", "takes-input", "tone-input"];
  list.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", updateSettingsRecap);
    }
  });

  updateSettingsRecap();
}
