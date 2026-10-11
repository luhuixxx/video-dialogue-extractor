"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import TranscriptionConfig
from .pipeline import transcribe_video
from .speakers import load_profiles
from .subtitles import segment_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video-extract")
    sub = parser.add_subparsers(dest="command", required=True)
    transcribe = sub.add_parser("transcribe", help="transcribe audio with speaker labels")
    transcribe.add_argument("video")
    transcribe.add_argument("-o", "--output", required=True)
    transcribe.add_argument("--enrollment", help=".npz created by enroll_speakers.py")
    transcribe.add_argument("--speakers", type=int,
                            help="number of roles for automatic clustering; omit to estimate 2-6")
    transcribe.add_argument("--start", type=float, default=0.0)
    transcribe.add_argument("--duration", type=float)
    transcribe.add_argument("--chunk-seconds", type=int, default=300)
    transcribe.add_argument("--resume", action="store_true",
                            help="resume completed chunks using the adjacent manifest")
    transcribe.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    transcribe.add_argument("--log-level", default="INFO")
    subtitles = sub.add_parser("segment-subtitles", help="save changed subtitle bands as images")
    subtitles.add_argument("video")
    subtitles.add_argument("-o", "--output", required=True)
    subtitles.add_argument("--start", type=float, default=0.0)
    subtitles.add_argument("--duration", type=float)
    subtitles.add_argument("--sample-interval", type=float, default=0.25)
    subtitles.add_argument("--change-threshold", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "segment-subtitles":
        if args.start < 0:
            raise SystemExit("--start must be non-negative")
        if args.duration is not None and args.duration <= 0:
            raise SystemExit("--duration must be greater than zero")
        if args.sample_interval <= 0:
            raise SystemExit("--sample-interval must be greater than zero")
        if args.change_threshold < 0:
            raise SystemExit("--change-threshold must be non-negative")
        records = segment_video(args.video, args.output, args.start, args.duration,
                                args.sample_interval, args.change_threshold)
        print(json.dumps({"segments": len(records), "output": str(args.output)}, ensure_ascii=False))
        return 0
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.enrollment and args.speakers:
        raise SystemExit("--enrollment and --speakers are mutually exclusive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be greater than zero")
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be greater than zero")
    if args.speakers is not None and args.speakers < 1:
        raise SystemExit("--speakers must be at least 1")
    profiles = load_profiles(args.enrollment) if args.enrollment else None
    count = transcribe_video(
        args.video, args.output, profiles, args.start, args.duration,
        TranscriptionConfig(chunk_seconds=args.chunk_seconds, device=args.device),
        args.speakers, args.resume,
    )
    print(json.dumps({"rows": count, "jsonl": str(args.output),
                      "text": str(Path(args.output).with_suffix(".txt"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
