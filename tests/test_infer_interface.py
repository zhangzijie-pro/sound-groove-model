import torch

from speaker_verification.interfaces.diar_interface import (
    SlotResult,
    fill_short_inactive_gaps,
    frames_to_segments,
    merge_similar_slots,
    remove_short_active_runs,
    smooth_sequence,
)


def test_smooth_sequence_replaces_short_run():
    seq = torch.tensor([1, 1, 2, 1, 1], dtype=torch.long)
    smoothed = smooth_sequence(seq, min_run=2)
    assert smoothed.tolist() == [1, 1, 1, 1, 1]


def test_remove_short_active_runs_filters_spikes():
    mask = torch.tensor([False, True, False, True, True, True, False])
    filtered = remove_short_active_runs(mask, min_active_frames=2)
    assert filtered.tolist() == [False, False, False, True, True, True, False]


def test_fill_short_inactive_gaps_bridges_brief_pause():
    mask = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    bridged = fill_short_inactive_gaps(mask, max_gap_frames=1)
    assert bridged.tolist() == [True, True, True, True, True]


def test_merge_similar_slots_and_segments():
    slot_results = [
        SlotResult(1, "slot_1", None, False, 10, 1.0, torch.tensor([1.0, 0.0, 0.0])),
        SlotResult(2, "slot_2", None, False, 8, 0.8, torch.tensor([0.99, 0.01, 0.0])),
        SlotResult(3, "slot_3", None, False, 6, 0.6, torch.tensor([0.0, 1.0, 0.0])),
    ]

    merged, merge_map = merge_similar_slots(slot_results, sim_threshold=0.95)
    segments = frames_to_segments(
        torch.tensor([0, 1, 1, 0, 3, 3, 0], dtype=torch.long),
        frame_shift_sec=0.1,
        slot_to_name={1: "speaker_a", 3: "speaker_b"},
        offset_sec=1.0,
    )

    assert len(merged) == 2
    assert merge_map[2] == 1
    assert segments[0].name == "speaker_a"
    assert segments[0].start_sec == 1.1
    assert segments[0].end_sec == 1.3
    assert segments[1].name == "speaker_b"
