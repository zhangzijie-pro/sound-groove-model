import torch

from speaker_verification.speaker_bank import JsonSpeakerBank


def test_speaker_bank_crud_and_open_set_identification(tmp_path):
    bank = JsonSpeakerBank(tmp_path / "speaker_bank.json", similarity_threshold=0.8)

    alice_embedding = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    bob_embedding = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    bank.add_speaker("alice", alice_embedding, display_name="Alice")
    bank.add_speaker("bob", bob_embedding, display_name="Bob", metadata={"team": "ops"})

    assert [item.speaker_id for item in bank.list_speakers()] == ["alice", "bob"]
    assert bank.get_speaker("bob").metadata["team"] == "ops"

    known = bank.identify(torch.tensor([0.95, 0.05, 0.0]))
    unknown = bank.identify(torch.tensor([0.0, 0.0, 1.0]))

    assert known["is_known"] is True
    assert known["speaker_id"] == "alice"
    assert unknown["is_known"] is False
    assert unknown["name"] == "unknown"

    updated = bank.update_speaker("alice", metadata={"tier": "gold"})
    assert updated.metadata["tier"] == "gold"

    assert bank.delete_speaker("bob") is True
    assert bank.delete_speaker("bob") is False
