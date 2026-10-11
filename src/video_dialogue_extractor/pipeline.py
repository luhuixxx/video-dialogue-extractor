"""End-to-end transcription pipeline."""

from __future__ import annotations

from pathlib import Path
import logging
import tempfile

import numpy as np
import soundfile as sf

from .asr import FunASRRecognizer
from .config import TranscriptionConfig, choose_device, limit_threads
from .media import duration_seconds, extract_audio
from .output import TranscriptWriter
from .speakers import (SpeakerProfile, VoiceSegmenter, cluster_profiles_from_embeddings,
                       extract_voice_embeddings)

LOGGER = logging.getLogger(__name__)


def transcribe_video(video: str | Path, output: str | Path,
                     profiles: list[SpeakerProfile] | None = None, start: float = 0.0,
                     duration: float | None = None,
                     config: TranscriptionConfig | None = None,
                     speakers: int | None = None, resume: bool = False) -> int:
    """Transcribe a clip and write UTF-8 JSONL/TXT records incrementally."""
    limit_threads()
    config = config or TranscriptionConfig()
    video = Path(video)
    total = duration_seconds(video)
    end = total if duration is None else min(total, start + duration)
    if start < 0 or start >= end:
        raise ValueError(f"invalid clip range: start={start}, end={end}")
    clip_duration = end - start
    device = choose_device(config.device)
    LOGGER.info("processing %.1f-%.1fs on %s", start, end, device)

    with tempfile.TemporaryDirectory(prefix="video-extract-") as temp_dir:
        full_wav = extract_audio(video, start, clip_duration, Path(temp_dir) / "full.wav")
        audio, sample_rate = sf.read(full_wav, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != 16000:
            raise ValueError(f"unexpected extracted sample rate: {sample_rate}")
        embeddings, positions = extract_voice_embeddings(
            audio, config.speaker_window_seconds, config.speaker_stride_seconds,
            config.min_speech_rms, device)
        if profiles is None:
            LOGGER.info("clustering unknown voices")
            profiles = cluster_profiles_from_embeddings(embeddings, positions, number=speakers)
            LOGGER.info("detected %d roles: %s", len(profiles), ", ".join(p.name for p in profiles))
        segmenter = VoiceSegmenter(
            profiles,
            window_seconds=config.speaker_window_seconds,
            stride_seconds=config.speaker_stride_seconds,
            switch_windows=config.speaker_switch_windows,
            min_speech_rms=config.min_speech_rms,
            device=device,
        )
        LOGGER.info("detecting speakers")
        speaker_segments = segmenter.segments_from_embeddings(embeddings, positions)
        recognizer = FunASRRecognizer(config.asr_model, config.vad_model, device,
                                      config.batch_size_seconds)
        pending: dict | None = None
        written = 0
        signature = {"video": str(video.resolve()), "start": round(start, 3),
                     "duration": round(clip_duration, 3), "chunk_seconds": config.chunk_seconds}
        with TranscriptWriter(output, resume=resume, run_signature=signature) as writer:
            offset = 0.0
            chunk_index = 0
            while offset < clip_duration:
                chunk_duration = min(config.chunk_seconds, clip_duration - offset)
                if writer.is_completed(chunk_index):
                    offset += chunk_duration
                    chunk_index += 1
                    continue
                chunk_start = int(round(offset * sample_rate))
                chunk_end = min(len(audio), chunk_start + int(round(chunk_duration * sample_rate)))
                chunk_wav = Path(temp_dir) / f"chunk-{chunk_index:05d}.wav"
                sf.write(chunk_wav, audio[chunk_start:chunk_end], sample_rate, subtype="PCM_16")
                for word in recognizer.transcribe(str(chunk_wav)):
                    global_start = start + offset + word.start
                    global_end = start + offset + max(word.end, word.start)
                    relative_mid = (global_start + global_end) / 2 - start
                    label = segmenter.label_at(relative_mid, speaker_segments)
                    speaker = profiles[label].name if label is not None else "未判定"
                    row = {"start": round(global_start, 3), "end": round(global_end, 3),
                           "speaker": speaker, "text": word.text,
                           "alignment_quality": word.alignment_quality,
                           "engine": f"FunASR {config.asr_model}"}
                    if pending and pending["speaker"] == speaker and global_start - pending["end"] <= 0.8:
                        pending["text"] += word.text
                        pending["end"] = row["end"]
                    else:
                        if pending:
                            writer.write(pending)
                            written += 1
                        pending = row
                if pending:
                    # Flush at every chunk boundary so a terminated run keeps all completed work.
                    writer.write(pending)
                    written += 1
                    pending = None
                writer.commit_chunk(chunk_index)
                LOGGER.info("transcribed %.1f-%.1fs", offset, offset + chunk_duration)
                offset += chunk_duration
                chunk_index += 1
            if pending:
                writer.write(pending)
                written += 1
        return written
