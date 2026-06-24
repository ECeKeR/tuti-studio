import { state } from "../state.js";
import { addLog } from "./logger.js";

export function fmtTime(s) {
  if (!s || isNaN(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec < 10 ? "0" : ""}${sec}`;
}

export function playTake(segIdx, takeIdx, url) {
  const key = `${segIdx}-${takeIdx}`;
  const icon = document.getElementById(`play-icon-${key}`);

  if (state.audioCache[key] && !state.audioCache[key].paused) {
    state.audioCache[key].pause();
    state.audioCache[key].currentTime = 0;
    icon?.setAttribute("data-lucide", "play");
    if (window.lucide) window.lucide.createIcons();
    return;
  }
  
  stopAll();
  
  if (!state.audioCache[key]) {
    state.audioCache[key] = new Audio(url);
    state.audioCache[key].addEventListener("ended", () => {
      icon?.setAttribute("data-lucide", "play");
      if (window.lucide) window.lucide.createIcons();
    });
  }
  
  state.audioCache[key].play();
  icon?.setAttribute("data-lucide", "pause");
  if (window.lucide) window.lucide.createIcons();
}

export function stopAll() {
  Object.entries(state.audioCache).forEach(([k, a]) => {
    a.pause();
    a.currentTime = 0;
    const ic = document.getElementById(`play-icon-${k}`);
    ic?.setAttribute("data-lucide", "play");
  });
  
  if (state.masterAudio && !state.masterAudio.paused) {
    state.masterAudio.pause();
    document.getElementById("play-icon")?.setAttribute("data-lucide", "play");
  }
  
  if (window.lucide) window.lucide.createIcons();
}

export function updateStitchBar() {
  if (!state.project) return;
  const allDone = state.project.segments.every(s => s.status === "completed");
  const btnStitch = document.getElementById("btn-stitch");
  if (btnStitch) btnStitch.disabled = !allDone;

  const btnPlay = document.getElementById("btn-play");
  const btnDownload = document.getElementById("btn-download");
  const playerSub = document.getElementById("player-sub");
  const stitchStatus = document.getElementById("stitch-status");

  if (state.project.stitched) {
    if (stitchStatus) stitchStatus.textContent = "";
    if (btnPlay) btnPlay.disabled = false;
    if (btnDownload) {
      btnDownload.style.display = "inline-flex";
      btnDownload.href = state.project.final_audio_url;
    }
    if (playerSub) playerSub.textContent = "Click play to listen";
  } else if (allDone) {
    if (stitchStatus) stitchStatus.textContent = "";
    if (btnDownload) btnDownload.style.display = "inline-flex";
  } else {
    const done = state.project.segments.filter(s => s.status === "completed").length;
    if (stitchStatus) stitchStatus.textContent = `${done} / ${state.project.segments.length} segments completed.`;
    if (btnDownload) btnDownload.style.display = "inline-flex";
  }
}

export function bindPlayerEvents() {
  if (!state.masterAudio) return;
  
  state.masterAudio.addEventListener("timeupdate", () => {
    if (!state.masterAudio) return;
    const pct = (state.masterAudio.currentTime / state.masterAudio.duration) * 100;
    const fill = document.getElementById("seek-fill");
    if (fill) fill.style.width = pct + "%";
    const cur = document.getElementById("time-cur");
    if (cur) cur.textContent = fmtTime(state.masterAudio.currentTime);
  });
  
  state.masterAudio.addEventListener("loadedmetadata", () => {
    if (!state.masterAudio) return;
    const dur = document.getElementById("time-dur");
    if (dur) dur.textContent = fmtTime(state.masterAudio.duration);
  });
  
  state.masterAudio.addEventListener("ended", () => {
    const playIcon = document.getElementById("play-icon");
    playIcon?.setAttribute("data-lucide", "play");
    const fill = document.getElementById("seek-fill");
    if (fill) fill.style.width = "0%";
    const cur = document.getElementById("time-cur");
    if (cur) cur.textContent = "0:00";
    if (window.lucide) window.lucide.createIcons();
  });
}

export function initPlayer() {
  window.playTake = playTake;
  
  const btnPlay = document.getElementById("btn-play");
  if (btnPlay) {
    btnPlay.addEventListener("click", () => {
      if (!state.masterAudio) return;
      const playIcon = document.getElementById("play-icon");
      if (state.masterAudio.paused) {
        stopAll();
        state.masterAudio.play();
        playIcon?.setAttribute("data-lucide", "pause");
      } else {
        state.masterAudio.pause();
        playIcon?.setAttribute("data-lucide", "play");
      }
      if (window.lucide) window.lucide.createIcons();
    });
  }

  const seekBg = document.getElementById("seek-bg");
  if (seekBg) {
    seekBg.addEventListener("click", e => {
      if (!state.masterAudio || !state.masterAudio.duration) return;
      const r = e.currentTarget.getBoundingClientRect();
      state.masterAudio.currentTime = ((e.clientX - r.left) / r.width) * state.masterAudio.duration;
    });
  }
}
