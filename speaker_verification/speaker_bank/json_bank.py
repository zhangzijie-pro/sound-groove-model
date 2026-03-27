from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SpeakerProfile:
    """Speaker profile stored in a local speaker bank."""

    speaker_id: str
    display_name: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def as_tensor(self) -> torch.Tensor:
        return F.normalize(torch.tensor(self.embedding, dtype=torch.float32), dim=0)


class JsonSpeakerBank:

    def __init__(
        self,
        path: str | Path,
        similarity_threshold: float = 0.78,
        unknown_label: str = "unknown",
    ) -> None:
        self.path = Path(path)
        self.similarity_threshold = float(similarity_threshold)
        self.unknown_label = unknown_label
        self._profiles: Dict[str, SpeakerProfile] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", [])
        self._profiles = {
            item["speaker_id"]: SpeakerProfile(**item)
            for item in profiles
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "similarity_threshold": self.similarity_threshold,
            "profiles": [asdict(profile) for profile in self.list_speakers()],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
        if embedding.dim() != 1:
            raise ValueError(f"Expected 1D embedding, got shape={tuple(embedding.shape)}")
        return F.normalize(embedding.detach().float().cpu(), dim=0)

    def add_speaker(
        self,
        speaker_id: str,
        embedding: torch.Tensor,
        *,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> SpeakerProfile:
        if speaker_id in self._profiles and not overwrite:
            raise ValueError(f"Speaker already exists: {speaker_id}")

        normalized = self._normalize_embedding(embedding)
        now = _utc_now_iso()
        existing = self._profiles.get(speaker_id)
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name or speaker_id,
            embedding=normalized.tolist(),
            metadata=metadata or {},
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        self._profiles[speaker_id] = profile
        self._save()
        return profile

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        return self._profiles.get(speaker_id)

    def list_speakers(self) -> List[SpeakerProfile]:
        return sorted(self._profiles.values(), key=lambda item: item.speaker_id)

    def update_speaker(
        self,
        speaker_id: str,
        *,
        embedding: Optional[torch.Tensor] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SpeakerProfile:
        current = self._profiles.get(speaker_id)
        if current is None:
            raise KeyError(f"Unknown speaker: {speaker_id}")

        normalized = current.embedding
        if embedding is not None:
            normalized = self._normalize_embedding(embedding).tolist()

        profile = SpeakerProfile(
            speaker_id=speaker_id,
            display_name=display_name or current.display_name,
            embedding=normalized,
            metadata=metadata if metadata is not None else current.metadata,
            created_at=current.created_at,
            updated_at=_utc_now_iso(),
        )
        self._profiles[speaker_id] = profile
        self._save()
        return profile

    def delete_speaker(self, speaker_id: str) -> bool:
        if speaker_id not in self._profiles:
            return False
        del self._profiles[speaker_id]
        self._save()
        return True

    def identify(self, embedding: torch.Tensor) -> Dict[str, Any]:
        normalized = self._normalize_embedding(embedding)
        if not self._profiles:
            return {"speaker_id": None, "name": self.unknown_label, "score": None, "is_known": False}

        best_profile: Optional[SpeakerProfile] = None
        best_score = float("-inf")
        for profile in self._profiles.values():
            score = float(torch.dot(normalized, profile.as_tensor()).item())
            if score > best_score:
                best_score = score
                best_profile = profile

        assert best_profile is not None
        if best_score < self.similarity_threshold:
            return {
                "speaker_id": None,
                "name": self.unknown_label,
                "score": round(best_score, 6),
                "is_known": False,
            }

        return {
            "speaker_id": best_profile.speaker_id,
            "name": best_profile.display_name,
            "score": round(best_score, 6),
            "is_known": True,
            "metadata": best_profile.metadata,
        }

