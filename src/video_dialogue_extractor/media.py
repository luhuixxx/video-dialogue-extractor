"""FFmpeg media helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile


def duration_seconds(video: str | Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def extract_audio(video: str | Path, start: float, duration: float, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
         "-t", f"{duration:.3f}", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", str(output)],
        check=True,
    )
    return output


def temporary_audio(video: str | Path, start: float, duration: float):
    """Yield a temporary 16 kHz mono WAV and clean it up automatically."""
    handle = tempfile.TemporaryDirectory(prefix="video-extract-")
    path = extract_audio(video, start, duration, Path(handle.name) / "audio.wav")
    return handle, path
