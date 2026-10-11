# Video Dialogue Extractor

Video Dialogue Extractor is a source-independent, Chinese-first tool for extracting dialogue from video. It combines speech recognition with dynamic speaker diarization and provides a separate subtitle-frame segmentation workflow.

## What it does

- Transcribes speech with FunASR Paraformer-zh.
- Assigns generic labels such as `角色1`, `角色2`, and `角色3`.
- Supports external voice enrollment for known roles or automatic clustering for an unknown cast.
- Reads audio once, reuses voice embeddings, and writes JSONL/TXT incrementally.
- Resumes completed audio chunks through a matching manifest file.
- Samples a subtitle band and saves a representative image whenever the band changes. OCR is intentionally a separate step.

## Requirements

- Python 3.10 or newer
- FFmpeg available on `PATH` (`ffmpeg` and `ffprobe`)
- A working PyTorch installation
- Enough disk space for the FunASR and Resemblyzer models downloaded from ModelScope on first use

Install the package in an isolated environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On Linux or macOS, use `source .venv/bin/activate` instead. Install a PyTorch build appropriate for the target CPU/GPU before running the pipeline. `--device auto` checks CUDA support and falls back to CPU when the installed build does not support the GPU architecture.

## Transcribe a video

For a video without known speaker samples, automatic clustering estimates between two and six roles:

```powershell
video-extract transcribe input.mp4 `
  --output outputs/transcript.jsonl `
  --chunk-seconds 300 `
  --device auto
```

Use `--speakers N` when the number of roles is known:

```powershell
video-extract transcribe input.mp4 `
  --speakers 3 `
  --output outputs/transcript.jsonl
```

For higher stability on a recurring cast, enroll one clean speech interval per role. Enrollment order becomes `角色1`, `角色2`, and so on:

```powershell
python scripts/enroll_speakers.py input.mp4 artifacts/roles.npz `
  --interval 120,150 `
  --interval 300,330

video-extract transcribe input.mp4 `
  --enrollment artifacts/roles.npz `
  --output outputs/transcript.jsonl
```

Long jobs can resume completed chunks:

```powershell
video-extract transcribe input.mp4 `
  --output outputs/transcript.jsonl `
  --chunk-seconds 300 `
  --resume
```

The command writes:

- `transcript.jsonl`: one UTF-8 JSON object per record
- `transcript.txt`: readable synchronized text
- `transcript.manifest.json`: completed chunk indexes and the input signature

Each JSONL record contains `start`, `end`, `speaker`, `text`, `alignment_quality`, and `engine`. If the voice evidence is insufficient, the speaker is `未判定` rather than an arbitrary role.

Use `--start` and `--duration` for a smoke test or a bounded clip.

## Segment subtitle frames

This workflow detects changes in the lower subtitle band and saves the changed band as PNG. It does not perform OCR:

```powershell
video-extract segment-subtitles input.mp4 `
  --output outputs/subtitle-segments `
  --sample-interval 0.25
```

The output contains `segments.jsonl` and image folders named by segment duration. The images can be passed to an OCR engine or a vision-language model in a later step.

## Project layout

```text
src/video_dialogue_extractor/   Core pipeline and CLI
scripts/enroll_speakers.py      Voice enrollment helper
scripts/segment_subtitles.py    Thin CLI wrapper
tests/                          Fast unit tests
docs/ARCHITECTURE.md             Component responsibilities
docs/TEST_REPORT.md              Local validation record
```

## Development

Run tests and compile checks from the repository root:

```powershell
python -m pytest -q
python -m compileall -q src scripts
```

The test suite uses synthetic data for fast checks. It does not download models or require a sample video. Keep videos, WAV files, enrollment embeddings, transcripts, and screenshots outside version control.

## Accuracy and privacy boundaries

Speaker diarization is an estimate. Overlapping speech, background music, short turns, and similar voices can reduce accuracy; review the output for important material and create fresh enrollment samples when changing videos. The project does not claim a fixed accuracy percentage without a labeled evaluation set.

The repository contains no video, audio, personal voice embedding, or model files. Model downloads and generated outputs stay in local caches and ignored directories.

## License

Released under the MIT License. See [LICENSE](LICENSE).
