# Contributing

## Setup

Use Python 3.10 or newer and install the project in editable mode:

```powershell
python -m pip install -e .
```

FFmpeg is required for media tests and real transcription runs.

## Before opening a change

Run:

```powershell
python -m pytest -q
python -m compileall -q src scripts
```

Keep changes focused, add a regression test for behavior changes, and do not commit videos, audio, transcripts, screenshots, model files, or voice embeddings. Use generic role names in shared code and documentation.

## Pull requests

Describe the user-visible behavior, the validation commands, and any model or hardware assumptions. Accuracy claims should include a labeled evaluation set and its methodology.
