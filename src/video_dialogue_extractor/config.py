"""Configuration models and runtime device selection."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os


@dataclass(frozen=True)
class TranscriptionConfig:
    """Runtime settings shared by the CLI and the pipeline."""

    asr_model: str = "paraformer-zh"
    vad_model: str = "fsmn-vad"
    chunk_seconds: int = 300
    speaker_window_seconds: float = 4.0
    speaker_stride_seconds: float = 2.0
    speaker_switch_windows: int = 2
    min_speech_rms: float = 0.008
    device: str = "auto"
    batch_size_seconds: int = 300


def choose_device(requested: str) -> str:
    """Return a usable FunASR/torch device, falling back to CPU."""
    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            capability = torch.cuda.get_device_capability()
            arch = f"sm_{capability[0] * 10 + capability[1]}"
            supported_arches = set(torch.cuda.get_arch_list())
            if supported_arches and arch not in supported_arches:
                logging.warning("PyTorch build does not support %s (%s); using CPU",
                                arch, ",".join(sorted(supported_arches)))
                cuda_ok = False
    except Exception:
        cuda_ok = False
    if requested in {"cuda", "gpu"} and not cuda_ok:
        logging.warning("CUDA requested but unavailable; using CPU")
        return "cpu"
    if requested == "auto":
        return "cuda" if cuda_ok else "cpu"
    return requested


def limit_threads() -> None:
    """Avoid oversubscribing a desktop CPU when GPU is unavailable."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "4")
