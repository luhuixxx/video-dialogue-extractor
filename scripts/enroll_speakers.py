#!/usr/bin/env python3
"""Create generic role enrollment from clean speech intervals.

Example:
  python scripts/enroll_speakers.py video.mp4 artifacts/speakers.npz \
    --interval 3600,3650 --interval 3650,3700

Intervals are assigned generic names in order: 角色1, 角色2, ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
from resemblyzer import preprocess_wav

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_dialogue_extractor.media import extract_audio
from video_dialogue_extractor.speakers import create_voice_encoder, save_profiles


def parse_interval(value: str) -> tuple[float, float]:
    try:
        start, end = (float(part) for part in value.split(",", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use start,end") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("interval end must be greater than start")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--interval", action="append", required=True, type=parse_interval,
                        metavar="START,END", help="one clean speech interval per role")
    args = parser.parse_args()
    encoder = create_voice_encoder()
    names: list[str] = []
    centers: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="video-enroll-") as temp_dir:
        for index, (start, end) in enumerate(args.interval):
            name = f"角色{index + 1}"
            wav = extract_audio(args.video, start, end - start,
                                Path(temp_dir) / f"speaker-{index}.wav")
            audio = preprocess_wav(str(wav))
            if len(audio) < 16000:
                raise ValueError(f"enrollment interval for {name!r} is shorter than one second")
            embeddings = []
            window = 10 * 16000
            for pos in range(0, max(1, len(audio) - window + 1), window):
                clip = audio[pos:pos + window]
                if len(clip) >= 16000:
                    embeddings.append(encoder.embed_utterance(clip))
            if not embeddings:
                embeddings.append(encoder.embed_utterance(audio))
            centers.append(np.mean(np.vstack(embeddings), axis=0))
            names.append(name)
    save_profiles(args.output, names, np.vstack(centers))
    print(f"wrote {len(names)} speaker profiles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
