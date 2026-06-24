export async function checkOllama() {
  try {
    const res = await fetch("/api/ollama/models");
    const d = await res.json();
    const sel = document.getElementById("ollama-model-input");
    const status = document.getElementById("ollama-status");
    if (!sel || !status) return;

    if (d.online && d.models.length > 0) {
      let chosen = null;
      d.models.forEach(m => {
        if (m === "qwen3:8b") chosen = m;
        else if (m.includes("qwen3") && (!chosen || chosen.includes("gemma"))) chosen = m;
        else if (m.includes("gemma") && !chosen) chosen = m;
      });
      if (!chosen && d.models.length > 0) chosen = d.models[0];

      sel.innerHTML = "";
      d.models.forEach(m => {
        const o = document.createElement("option");
        o.value = m; o.textContent = m;
        if (m === chosen) o.selected = true;
        sel.appendChild(o);
      });
      status.innerHTML = `<span class="status-dot active" style="width:6px; height:6px;"></span> ${d.models.length} local models found`;
      status.style.color = "var(--green)";
    } else {
      status.innerHTML = `<span class="status-dot busy" style="width:6px; height:6px;"></span> ` + (d.online ? "Ollama online, no models found." : "Ollama offline");
      status.style.color = d.online ? "var(--amber)" : "var(--red)";
    }
  } catch(e) {
    const status = document.getElementById("ollama-status");
    if (status) {
      status.innerHTML = `<span class="status-dot busy" style="width:6px; height:6px; background:var(--red); box-shadow:0 0 6px var(--red);"></span> Cannot reach Ollama`;
      status.style.color = "var(--red)";
    }
  }
}

export function initSettings() {
  const llmBackend = document.getElementById("llm-backend-input");
  if (llmBackend) {
    llmBackend.addEventListener("change", e => {
      const show = e.target.value === "ollama";
      const row = document.getElementById("ollama-row");
      if (row) row.style.display = show ? "" : "none";
      if (show) checkOllama();
    });
  }
  checkOllama();
}
