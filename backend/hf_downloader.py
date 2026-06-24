# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

import os
import sys
import queue
import logging
import threading
from pathlib import Path
from typing import Generator, Dict, Any, Callable

log = logging.getLogger(__name__)

# Hugging Face Hub constants
try:
    from huggingface_hub import constants as hf_constants
    HF_HUB_CACHE = hf_constants.HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Tuti Models Registry
MODELS_LIST = [
    # ── Clone Models ──
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "name": "Qwen3 TTS 1.7B Base (Centroid Extraction)",
        "type": "clone",
        "backend": "mlx",
        "size_label": "1.7B",
        "description": "Base model used to extract reference speaker timbre centroids (MLX / Apple Silicon)."
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "name": "Qwen3 TTS 1.7B CustomVoice (Inference)",
        "type": "clone",
        "backend": "mlx",
        "size_label": "1.7B",
        "description": "Voice cloning inference model where custom voice centroids are injected (MLX / Apple Silicon)."
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "name": "Qwen3 TTS 0.6B Base (Centroid Extraction)",
        "type": "clone",
        "backend": "mlx",
        "size_label": "0.6B",
        "description": "Base model used to extract reference speaker timbre centroids (MLX / Apple Silicon)."
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "name": "Qwen3 TTS 0.6B CustomVoice (Inference)",
        "type": "clone",
        "backend": "mlx",
        "size_label": "0.6B",
        "description": "Voice cloning inference model where custom voice centroids are injected (MLX / Apple Silicon)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "name": "Qwen3 TTS 1.7B Base (PyTorch)",
        "type": "clone",
        "backend": "pytorch",
        "size_label": "1.7B",
        "description": "Base model for voice cloning centroid extraction on CUDA, CPU, or MPS (PyTorch)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "name": "Qwen3 TTS 1.7B CustomVoice (PyTorch)",
        "type": "clone",
        "backend": "pytorch",
        "size_label": "1.7B",
        "description": "Custom voice cloning inference model on CUDA, CPU, or MPS (PyTorch)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "name": "Qwen3 TTS 0.6B Base (PyTorch)",
        "type": "clone",
        "backend": "pytorch",
        "size_label": "0.6B",
        "description": "Base model for voice cloning centroid extraction on CUDA, CPU, or MPS (PyTorch)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "name": "Qwen3 TTS 0.6B CustomVoice (PyTorch)",
        "type": "clone",
        "backend": "pytorch",
        "size_label": "0.6B",
        "description": "Custom voice cloning inference model on CUDA, CPU, or MPS (PyTorch)."
    },

    # ── Preset / SFT Models ──
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "name": "Qwen3 TTS 1.7B CustomVoice (Preset & SFT)",
        "type": "preset",
        "backend": "mlx",
        "size_label": "1.7B",
        "description": "Standard preset voice engine and base model used for LoRA training (MLX / Apple Silicon)."
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "name": "Qwen3 TTS 0.6B CustomVoice (Preset & SFT)",
        "type": "preset",
        "backend": "mlx",
        "size_label": "0.6B",
        "description": "Standard preset voice engine and base model used for LoRA training (MLX / Apple Silicon)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "name": "Qwen3 TTS 1.7B CustomVoice (Preset/PyTorch)",
        "type": "preset",
        "backend": "pytorch",
        "size_label": "1.7B",
        "description": "Preset voices and SFT training base model (PyTorch)."
    },
    {
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "name": "Qwen3 TTS 0.6B CustomVoice (Preset/PyTorch)",
        "type": "preset",
        "backend": "pytorch",
        "size_label": "0.6B",
        "description": "Preset voices and SFT training base model (PyTorch)."
    },

    # ── Voice Design Models ──
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "name": "Qwen3 TTS 1.7B VoiceDesign (Prompt-based)",
        "type": "design",
        "backend": "mlx",
        "size_label": "1.7B",
        "description": "Prompt-guided voice generation and design engine (MLX / Apple Silicon)."
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-bf16",
        "name": "Qwen3 TTS 0.6B VoiceDesign (Prompt-based)",
        "type": "design",
        "backend": "mlx",
        "size_label": "0.6B",
        "description": "Prompt-guided voice generation and design engine (MLX / Apple Silicon)."
    }
]


def is_model_cached(hf_repo: str) -> bool:
    """Check if a HuggingFace model is fully cached locally."""
    try:
        repo_cache = Path(HF_HUB_CACHE) / ("models--" + hf_repo.replace("/", "--"))
        if not repo_cache.exists():
            return False

        # If blobs contain incomplete downloads, it's not fully cached
        blobs_dir = repo_cache / "blobs"
        if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
            return False

        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.exists():
            return False

        # Look for weight extensions (.safetensors, .bin, .pt)
        for ext in (".safetensors", ".bin", ".pt"):
            if any(snapshots_dir.rglob(f"*{ext}")):
                return True

        return False
    except Exception as e:
        log.warning(f"Error checking cache for {hf_repo}: {e}")
        return False


class TqdmInterceptors:
    """Monkey-patches tqdm temporarily to capture progress updates."""
    def __init__(self, callback: Callable[[int, int, str], None]):
        self.callback = callback
        self._original_tqdm = None
        self._original_auto_tqdm = None
        self._original_hf_tqdm_update = None
        self._patched_modules = {}

    def __enter__(self):
        try:
            import tqdm
            self._original_tqdm = tqdm.tqdm

            callback = self.callback
            class TrackedTqdm(tqdm.tqdm):
                def __init__(self, *args, **kwargs):
                    desc = kwargs.get("desc", "") or ""
                    kwargs["disable"] = False  # Ensure it runs/updates
                    super().__init__(*args, **kwargs)
                    self._filename = desc.split(":")[0].strip() if ":" in desc else desc.strip()

                def update(self, n=1):
                    res = super().update(n)
                    current = getattr(self, "n", 0)
                    total = getattr(self, "total", 0) or 0
                    if total > 0:
                        callback(current, total, self._filename)
                    return res

            tqdm.tqdm = TrackedTqdm
            if hasattr(tqdm, "auto") and hasattr(tqdm.auto, "tqdm"):
                self._original_auto_tqdm = tqdm.auto.tqdm
                tqdm.auto.tqdm = TrackedTqdm

            # Patch globally imported references in sys.modules
            tqdm_attrs = ["tqdm", "base_tqdm", "old_tqdm"]
            for name, module in list(sys.modules.items()):
                if "huggingface" in name or name.startswith("tqdm"):
                    for attr in tqdm_attrs:
                        if hasattr(module, attr):
                            val = getattr(module, attr)
                            if val is self._original_tqdm or val is self._original_auto_tqdm:
                                self._patched_modules[f"{name}.{attr}"] = (module, attr, val)
                                setattr(module, attr, TrackedTqdm)

            # Patch huggingface_hub specific tqdm
            try:
                from huggingface_hub.utils import tqdm as hf_tqdm_module
                if hasattr(hf_tqdm_module, "tqdm"):
                    hf_tqdm_class = hf_tqdm_module.tqdm
                    self._original_hf_tqdm_update = hf_tqdm_class.update

                    def patched_hf_update(tqdm_self, n=1):
                        res = self._original_hf_tqdm_update(tqdm_self, n)
                        current = getattr(tqdm_self, "n", 0)
                        total = getattr(tqdm_self, "total", 0) or 0
                        desc = getattr(tqdm_self, "desc", "") or ""
                        if total > 0:
                            callback(current, total, desc)
                        return res
                    hf_tqdm_class.update = patched_hf_update
            except Exception:
                pass
        except Exception as e:
            log.warning(f"Failed to enter TqdmInterceptors: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import tqdm
            if self._original_tqdm:
                tqdm.tqdm = self._original_tqdm
            if self._original_auto_tqdm and hasattr(tqdm, "auto"):
                tqdm.auto.tqdm = self._original_auto_tqdm
            for key, (module, attr, original) in self._patched_modules.items():
                setattr(module, attr, original)
            if self._original_hf_tqdm_update:
                try:
                    from huggingface_hub.utils import tqdm as hf_tqdm_module
                    if hasattr(hf_tqdm_module, "tqdm"):
                        hf_tqdm_module.tqdm.update = self._original_hf_tqdm_update
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Failed to clean up TqdmInterceptors: {e}")


def download_model_generator(repo_id: str) -> Generator[Dict[str, Any], None, None]:
    """Downloads the model using huggingface_hub in a thread and yields progress status."""
    q = queue.Queue()
    current_files_progress = {}

    def callback(downloaded: int, total: int, filename: str):
        # Store individual file progresses
        current_files_progress[filename] = (downloaded, total)
        
        # Calculate global sums
        total_downloaded = sum(d for d, _ in current_files_progress.values())
        total_size = sum(t for _, t in current_files_progress.values())
        
        # Only report if we have some data
        if total_size > 0:
            q.put({
                "status": "downloading",
                "downloaded": total_downloaded,
                "total": total_size,
                "filename": filename
            })

    def worker():
        try:
            from huggingface_hub import snapshot_download
            with TqdmInterceptors(callback):
                snapshot_download(
                    repo_id=repo_id,
                    token=None,
                    allow_patterns=["*.safetensors", "*.bin", "*.pt", "*.json", "*.txt", "*.model"]
                )
            q.put({"status": "completed"})
        except Exception as e:
            log.exception(f"Error downloading {repo_id}")
            q.put({"status": "error", "error": str(e)})

    thread = threading.Thread(target=worker, name=f"HFDownload-{repo_id.replace('/', '--')}")
    thread.daemon = True
    thread.start()

    while True:
        try:
            msg = q.get(timeout=0.5)
            yield msg
            if msg["status"] in ("completed", "error"):
                break
        except queue.Empty:
            if not thread.is_alive():
                break
            yield {"status": "ping"}
