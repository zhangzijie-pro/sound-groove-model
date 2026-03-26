import torch

from speaker_verification.interfaces.global_tracker import GlobalSpeakerTracker
from speaker_verification.interfaces.diar_interface import ChunkInferenceResult, SlotResult


def build_dummy_result(slot_id=1, dim=64):
    slot = SlotResult(
        slot=slot_id,
        name="unknown",
        score=None,
        is_known=False,
        num_frames=100,
        duration_sec=1.0,
        prototype=torch.randn(dim),
    )
    return ChunkInferenceResult(
        num_speakers=1,
        dominant_speaker="unknown",
        dominant_speaker_slot=slot_id,
        activity_ratio=0.8,
        slots=[slot],
        segments=[],
        frame_activity_prob=torch.ones(100),
        local_frame_ids=torch.ones(100, dtype=torch.long),
        global_frame_ids=None,
    )


def test_tracker_create_global_id():
    tracker = GlobalSpeakerTracker(match_threshold=0.0)
    result = build_dummy_result()
    out = tracker.update(result)

    assert len(out.local_to_global) == 1
    assert out.dominant_global_id is not None