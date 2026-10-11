"""Voice embedding enrollment and dynamic speaker segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    center: np.ndarray


@dataclass(frozen=True)
class SpeakerWindow:
    start: float
    end: float
    label: int
    confidence: float


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    label: int
    confidence: float


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-8)


def load_profiles(path: str | Path) -> list[SpeakerProfile]:
    """Load ``.npz`` enrollment created by ``enroll_speakers.py``."""
    data = np.load(path, allow_pickle=False)
    names = [str(x) for x in data["names"].tolist()]
    centers = _normalize(data["centers"])
    if not names or len(names) != len(centers):
        raise ValueError("enrollment must contain matching names and centers")
    return [SpeakerProfile(name, center) for name, center in zip(names, centers)]


def save_profiles(path: str | Path, names: list[str], centers: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, names=np.asarray(names), centers=_normalize(centers))


def save_profile_metadata(path: str | Path, profiles: Iterable[SpeakerProfile]) -> None:
    rows = [{"name": p.name, "dimensions": int(p.center.size)} for p in profiles]
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_voice_embeddings(audio: np.ndarray, window_seconds: float, stride_seconds: float,
                             min_speech_rms: float, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized embeddings and their start times from speech-like windows."""
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder(device=device)
    sample_rate = 16000
    size = max(1, int(window_seconds * sample_rate))
    step = max(1, int(stride_seconds * sample_rate))
    embeddings: list[np.ndarray] = []
    positions: list[float] = []
    for pos in range(0, max(0, len(audio) - size + 1), step):
        clip = audio[pos:pos + size]
        rms = float(np.sqrt(np.mean(np.square(clip)))) if len(clip) else 0.0
        if rms < min_speech_rms:
            continue
        embedding = encoder.embed_utterance(clip)
        embeddings.append(embedding / max(float(np.linalg.norm(embedding)), 1e-8))
        positions.append(pos / sample_rate)
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32), np.asarray(positions, dtype=np.float32)


def _profiles_from_clusters(embeddings: np.ndarray, labels: np.ndarray,
                            positions: np.ndarray) -> list[SpeakerProfile]:
    """Create stable generic role profiles ordered by first occurrence in the audio."""
    cluster_ids = sorted(set(int(x) for x in labels), key=lambda x: float(positions[labels == x].min()))
    centers = []
    for cluster_id in cluster_ids:
        centers.append(np.mean(embeddings[labels == cluster_id], axis=0))
    return [SpeakerProfile(f"角色{index + 1}", center)
            for index, center in enumerate(_normalize(np.vstack(centers)))]


def cluster_profiles_from_embeddings(embeddings: np.ndarray, positions: np.ndarray,
                                      number: int | None = None, min_speakers: int = 2,
                                      max_speakers: int = 6) -> list[SpeakerProfile]:
    """Cluster precomputed voice embeddings and return generic ``角色N`` profiles."""
    if len(embeddings) == 0:
        raise ValueError("no speech-like audio was found for automatic role clustering")
    if len(embeddings) == 1:
        return [SpeakerProfile("角色1", embeddings[0])]
    if len(np.unique(embeddings, axis=0)) < 2:
        return [SpeakerProfile("角色1", np.mean(embeddings, axis=0))]
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if number is not None and number < 1:
        raise ValueError("--speakers must be at least 1")
    upper = min(max_speakers, len(embeddings) - 1)
    candidates = [min(number, len(embeddings))] if number else list(range(min_speakers, upper + 1))
    candidates = [candidate for candidate in candidates if 1 <= candidate <= len(embeddings)]
    if not candidates:
        candidates = [1]
    best_labels = None
    best_score = float("-inf")
    for candidate in candidates:
        model = KMeans(n_clusters=candidate, n_init=10, random_state=0)
        labels = model.fit_predict(embeddings)
        unique_labels = len(set(int(x) for x in labels))
        score = 0.0 if unique_labels == 1 else float(silhouette_score(embeddings, labels))
        if score > best_score:
            best_score = score
            best_labels = labels
    return _profiles_from_clusters(embeddings, np.asarray(best_labels), positions)


def cluster_profiles(audio: np.ndarray, number: int | None = None,
                     min_speakers: int = 2, max_speakers: int = 6,
                     window_seconds: float = 4.0, stride_seconds: float = 2.0,
                     min_speech_rms: float = 0.008, device: str = "cpu") -> list[SpeakerProfile]:
    """Extract, cluster, and return generic ``角色N`` profiles.

    When ``number`` is omitted, the silhouette score chooses a count between two
    and ``max_speakers``. This is a diarization estimate, so the result should
    still be reviewed for overlapping speech or very short turns.
    """
    embeddings, positions = extract_voice_embeddings(audio, window_seconds, stride_seconds,
                                                     min_speech_rms, device)
    return cluster_profiles_from_embeddings(embeddings, positions, number,
                                            min_speakers, max_speakers)


class VoiceSegmenter:
    """Assign sliding voice embeddings to enrolled profiles with hysteresis."""

    def __init__(self, profiles: list[SpeakerProfile], window_seconds: float = 4.0,
                 stride_seconds: float = 2.0, switch_windows: int = 2,
                 min_speech_rms: float = 0.008, device: str = "cpu") -> None:
        if not profiles:
            raise ValueError("at least one speaker profile is required")
        self.profiles = profiles
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.switch_windows = max(1, switch_windows)
        self.min_speech_rms = min_speech_rms
        self.device = device

    def _encoder(self):
        from resemblyzer import VoiceEncoder

        return VoiceEncoder(device=self.device)

    def windows_from_embeddings(self, embeddings: np.ndarray, positions: np.ndarray,
                                sample_rate: int = 16000) -> list[SpeakerWindow]:
        centers = _normalize(np.vstack([p.center for p in self.profiles]))
        output: list[SpeakerWindow] = []
        for embedding, position in zip(embeddings, positions):
            embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
            scores = centers @ embedding
            order = np.argsort(scores)[::-1]
            confidence = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
            output.append(SpeakerWindow(float(position), float(position + self.window_seconds),
                                        int(order[0]), confidence))
        return output

    def windows(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeakerWindow]:
        embeddings, positions = extract_voice_embeddings(
            audio, self.window_seconds, self.stride_seconds, self.min_speech_rms, self.device)
        return self.windows_from_embeddings(embeddings, positions, sample_rate)

    def segments_from_windows(self, windows: list[SpeakerWindow]) -> list[SpeakerSegment]:
        if not windows:
            return []
        current = windows[0].label
        start = windows[0].start
        confidence = windows[0].confidence
        pending: int | None = None
        pending_count = 0
        result: list[SpeakerSegment] = []
        for window in windows[1:]:
            confidence = max(confidence, window.confidence)
            if window.label == current:
                pending = None
                pending_count = 0
                continue
            if pending == window.label:
                pending_count += 1
            else:
                pending = window.label
                pending_count = 1
            if pending_count >= self.switch_windows:
                result.append(SpeakerSegment(start, window.start, current, confidence))
                start = window.start
                current = window.label
                confidence = window.confidence
                pending = None
                pending_count = 0
        result.append(SpeakerSegment(start, windows[-1].end, current, confidence))
        return result

    def segments_from_embeddings(self, embeddings: np.ndarray, positions: np.ndarray,
                                 sample_rate: int = 16000) -> list[SpeakerSegment]:
        return self.segments_from_windows(self.windows_from_embeddings(embeddings, positions, sample_rate))

    def segments(self, audio: np.ndarray, sample_rate: int = 16000) -> list[SpeakerSegment]:
        return self.segments_from_windows(self.windows(audio, sample_rate))

    def label_at(self, second: float, segments: list[SpeakerSegment]) -> int | None:
        if not segments:
            return None
        for segment in segments:
            if segment.start <= second <= segment.end:
                return segment.label
        return None
