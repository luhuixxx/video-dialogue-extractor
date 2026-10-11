import numpy as np

from video_dialogue_extractor.speakers import (SpeakerProfile, SpeakerSegment, VoiceSegmenter,
                                               save_profiles, load_profiles, _profiles_from_clusters)


def test_profiles_round_trip(tmp_path):
    path = tmp_path / "profiles.npz"
    save_profiles(path, ["角色1", "角色2"], np.eye(2, dtype=np.float32))
    profiles = load_profiles(path)
    assert [profile.name for profile in profiles] == ["角色1", "角色2"]
    assert np.allclose(np.linalg.norm([profile.center for profile in profiles], axis=1), 1)


def test_label_at_uses_relative_timestamps():
    segmenter = VoiceSegmenter([SpeakerProfile("a", np.ones(2))])
    segments = [SpeakerSegment(0.0, 2.0, 0, 0.5), SpeakerSegment(2.0, 4.0, 0, 0.5)]
    assert segmenter.label_at(1.2, segments) == 0
    assert segmenter.label_at(4.2, segments) is None


def test_cluster_profiles_use_generic_role_names():
    embeddings = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    profiles = _profiles_from_clusters(embeddings, labels, np.asarray([0, 1, 2, 3]))
    assert [profile.name for profile in profiles] == ["角色1", "角色2"]


def test_cluster_profiles_tolerate_identical_embeddings():
    from video_dialogue_extractor.speakers import cluster_profiles_from_embeddings

    embeddings = np.ones((4, 3), dtype=np.float32)
    profiles = cluster_profiles_from_embeddings(embeddings, np.arange(4), number=2)
    assert profiles
