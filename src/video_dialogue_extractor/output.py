"""Incremental transcript writers and timestamp formatting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TextIO


def timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


class TranscriptWriter:
    def __init__(self, jsonl: str | Path, resume: bool = False,
                 run_signature: dict | None = None) -> None:
        self.jsonl_path = Path(jsonl)
        self.txt_path = self.jsonl_path.with_suffix(".txt")
        self.manifest_path = self.jsonl_path.with_suffix(".manifest.json")
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_signature = run_signature or {}
        self.completed_chunks: set[int] = set()
        if not resume and self.manifest_path.exists():
            self.manifest_path.unlink()
        if resume and self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") != self.run_signature:
                raise ValueError("resume manifest does not match this input range or chunk size")
            self.completed_chunks = {int(x) for x in manifest.get("completed_chunks", [])}
        if resume and self.jsonl_path.exists() and not self.manifest_path.exists():
            raise ValueError("cannot resume without a matching manifest file")
        mode = "a" if resume else "w"
        self._jsonl: TextIO = self.jsonl_path.open(mode, encoding="utf-8")
        self._txt: TextIO = self.txt_path.open(mode, encoding="utf-8")
        self.count = 0

    def write(self, row: dict) -> None:
        self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._txt.write(f"[{timestamp(row['start'])}-{timestamp(row['end'])}] "
                        f"{row['speaker']}：{row['text']}\n")
        self._txt.flush()
        self.count += 1

    def is_completed(self, chunk_index: int) -> bool:
        return chunk_index in self.completed_chunks

    def commit_chunk(self, chunk_index: int) -> None:
        self.completed_chunks.add(chunk_index)
        manifest = {"signature": self.run_signature,
                    "completed_chunks": sorted(self.completed_chunks)}
        temporary = self.manifest_path.with_suffix(".manifest.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def close(self) -> None:
        self._jsonl.close()
        self._txt.close()

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
