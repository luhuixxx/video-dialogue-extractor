import numpy as np
import cv2

from video_dialogue_extractor.subtitles import changed, has_subtitle, normalize_band
import pytest


def test_subtitle_change_detection():
    black = np.zeros((200, 400, 3), dtype=np.uint8)
    white = black.copy()
    cv2.putText(white, "字幕", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    assert not has_subtitle(black)
    assert has_subtitle(white)
    first = normalize_band(black)
    second = normalize_band(white)
    assert changed(first, second, 1.0)


@pytest.mark.parametrize("kwargs", [
    {"sample_interval": 0},
    {"sample_interval": -1},
    {"duration": 0},
    {"start": -1},
])
def test_segment_video_rejects_invalid_ranges(tmp_path, kwargs):
    from video_dialogue_extractor.subtitles import segment_video

    with pytest.raises(ValueError):
        segment_video("missing.mp4", tmp_path / "out", **kwargs)
