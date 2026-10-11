"""Fast subtitle-band change detection without OCR."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def crop_subtitle_band(frame: np.ndarray) -> np.ndarray:
    height = frame.shape[0]
    return frame[int(height * 0.80):int(height * 0.985), :]


def normalize_band(band: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (640, 160), interpolation=cv2.INTER_AREA)


def has_subtitle(band: np.ndarray) -> bool:
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 150)) > 0.006


def changed(previous: np.ndarray | None, current: np.ndarray, threshold: float) -> bool:
    if previous is None:
        return True
    difference = cv2.absdiff(previous, current)
    return float(np.mean(difference)) >= threshold or float(np.mean(difference > 18)) >= 0.015


def segment_video(video: str | Path, output: str | Path, start: float = 0.0,
                  duration: float | None = None, sample_interval: float = 0.25,
                  change_threshold: float = 5.0) -> list[dict]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be greater than zero")
    if sample_interval <= 0:
        raise ValueError("sample_interval must be greater than zero")
    if change_threshold < 0:
        raise ValueError("change_threshold must be non-negative")
    output = Path(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    end = total / fps if duration is None else min(total / fps, start + duration)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        next_sample = start
        previous = None
        segment_start: float | None = None
        representative: np.ndarray | None = None
        records: list[dict] = []
        index = 0
        while next_sample < end:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp + 1e-6 < next_sample:
                continue
            next_sample += sample_interval
            band = crop_subtitle_band(frame)
            normalized = normalize_band(band)
            if changed(previous, normalized, change_threshold):
                if segment_start is not None and representative is not None:
                    records.append(_save_segment(image_dir, index, segment_start, timestamp,
                                                 representative))
                    index += 1
                segment_start = timestamp if has_subtitle(band) else None
                representative = band.copy() if segment_start is not None else None
            previous = normalized
        if segment_start is not None and representative is not None:
            records.append(_save_segment(image_dir, index, segment_start, end, representative))
    finally:
        cap.release()
    metadata = output / "segments.jsonl"
    with metadata.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return records


def _save_segment(image_dir: Path, index: int, start: float, end: float,
                  image: np.ndarray) -> dict:
    end = max(end, start + 0.01)
    record = {"index": index, "start": round(start, 3), "end": round(end, 3),
              "duration": round(end - start, 3)}
    folder = image_dir / f"{record['duration']:.2f}s"
    folder.mkdir(exist_ok=True)
    path = folder / f"subtitle_{index:06d}.png"
    cv2.imwrite(str(path), image)
    record["image"] = str(path)
    return record
