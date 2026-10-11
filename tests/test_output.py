import json

from video_dialogue_extractor.output import TranscriptWriter, timestamp


def test_timestamp_and_incremental_writer(tmp_path):
    assert timestamp(3661.2) == "01:01:01"
    path = tmp_path / "result.jsonl"
    with TranscriptWriter(path) as writer:
        writer.write({"start": 0.0, "end": 1.0, "speaker": "甲", "text": "你好"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "你好"
    assert "甲：你好" in path.with_suffix(".txt").read_text(encoding="utf-8")


def test_writer_manifest_resumes_completed_chunks(tmp_path):
    path = tmp_path / "resume.jsonl"
    signature = {"input": "sample", "chunk_seconds": 10}
    with TranscriptWriter(path, run_signature=signature) as writer:
        writer.write({"start": 0, "end": 1, "speaker": "角色1", "text": "一"})
        writer.commit_chunk(0)
    with TranscriptWriter(path, resume=True, run_signature=signature) as writer:
        assert writer.is_completed(0)
        assert not writer.is_completed(1)
