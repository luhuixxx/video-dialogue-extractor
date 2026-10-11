"""FunASR adapter with timestamp normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    alignment_quality: str = "word"


class FunASRRecognizer:
    def __init__(self, model_name: str = "paraformer-zh", vad_model: str = "fsmn-vad",
                 device: str = "cpu", batch_size_seconds: int = 300) -> None:
        from funasr import AutoModel

        kwargs: dict[str, Any] = {"model": model_name, "vad_model": vad_model,
                                  "disable_update": True, "batch_size_s": batch_size_seconds}
        if device == "cuda":
            kwargs["device"] = "cuda"
        self.model = AutoModel(**kwargs)
        self.batch_size_seconds = batch_size_seconds

    def transcribe(self, wav: str) -> Iterator[Word]:
        result = self.model.generate(input=wav, batch_size_s=self.batch_size_seconds)
        for item in result or []:
            text = (item.get("text") or "").strip()
            timestamps = item.get("timestamp") or []
            if not text:
                continue
            tokens = text.split()
            if timestamps and len(tokens) == len(timestamps):
                for token, (start_ms, end_ms) in zip(tokens, timestamps):
                    yield Word(float(start_ms) / 1000, float(end_ms) / 1000, token, "word")
            else:
                start_ms = item.get("start_time", item.get("start", 0)) or 0
                end_ms = item.get("end_time", item.get("end", 0)) or 0
                if timestamps and isinstance(timestamps[-1], (list, tuple)):
                    end_ms = max(float(end_ms), float(timestamps[-1][1]))
                start = float(start_ms) / 1000
                end = float(end_ms) / 1000
                if end <= start:
                    end = start + max(0.1, len(text) * 0.08)
                yield Word(start, end, text, "segment")
