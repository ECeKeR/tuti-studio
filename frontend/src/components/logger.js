export function setStatus(stateName, text) {
  const dot = document.getElementById("status-dot");
  if (dot) {
    dot.className = "status-dot " + (stateName === "busy" ? "busy" : stateName === "active" ? "active" : "");
  }
  const txt = document.getElementById("status-text");
  if (txt) {
    txt.textContent = text;
  }
}

export function addLog(msg, level = "info") {
  const panel = document.getElementById("log-panel");
  const diagnosticsPanel = document.getElementById("log-panel-diagnostics");
  
  const appendTo = (p, limit) => {
    if (!p) return;
    const el = document.createElement("div");
    el.className = "log-line " + level;
    const t = new Date().toLocaleTimeString();
    el.textContent = `[${t}] ${msg}`;
    p.appendChild(el);
    p.scrollTop = p.scrollHeight;
    while (p.children.length > limit) p.removeChild(p.firstChild);
  };

  appendTo(panel, 60);
  appendTo(diagnosticsPanel, 150);
}
