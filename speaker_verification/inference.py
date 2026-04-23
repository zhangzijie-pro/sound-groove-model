import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F

from speaker_verification.audio.features import TARGET_SR, load_wav_mono, wav_to_fbank_infer
from speaker_verification.engine.checkpoint import get_model_state_from_ckpt
from speaker_verification.models.eend_query_model import EENDQueryModel


FRAME_SHIFT_SEC = 0.01


def _torch_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cfg_get(cfg: Mapping[str, Any], dotted_key: str, default: Any) -> Any:
    cur: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_config_file(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}

    try:
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover - only used when OmegaConf is missing.
        raise RuntimeError("OmegaConf is required when loading a config file.") from exc

    cfg = OmegaConf.load(str(path))
    return dict(OmegaConf.to_container(cfg, resolve=True))


def build_model_from_config(cfg: Mapping[str, Any], device: torch.device) -> EENDQueryModel:
    model = EENDQueryModel(
        in_channels=int(_cfg_get(cfg, "model.in_channels", 80)),
        enc_channels=int(_cfg_get(cfg, "model.channels", 512)),
        d_model=int(_cfg_get(cfg, "model.d_model", 256)),
        max_speakers=int(_cfg_get(cfg, "model.max_mix_speakers", 3)),
        assign_scale=float(_cfg_get(cfg, "model.assign_scale", 8.0)),
        decoder_type=str(_cfg_get(cfg, "model.decoder_type", "query")),
        post_ffn_hidden_dim=int(_cfg_get(cfg, "model.post_ffn_hidden_dim", 512)),
        post_ffn_dropout=float(_cfg_get(cfg, "model.post_ffn_dropout", 0.2)),
        decoder_layers=int(_cfg_get(cfg, "model.decoder_layers", 4)),
        decoder_heads=int(_cfg_get(cfg, "model.decoder_heads", 8)),
        decoder_ffn=int(_cfg_get(cfg, "model.decoder_ffn", 512)),
        dropout=float(_cfg_get(cfg, "model.dropout", 0.1)),
    )
    return model.to(device)


def _remove_short_runs(binary_seq: torch.Tensor, min_active_frames: int) -> torch.Tensor:
    out = binary_seq.clone()
    total = out.numel()
    pos = 0
    while pos < total:
        if float(out[pos].item()) < 0.5:
            pos += 1
            continue

        end = pos + 1
        while end < total and float(out[end].item()) >= 0.5:
            end += 1

        if end - pos < min_active_frames:
            out[pos:end] = 0.0
        pos = end
    return out


def _fill_short_gaps(binary_seq: torch.Tensor, max_gap_frames: int) -> torch.Tensor:
    if max_gap_frames <= 0:
        return binary_seq

    out = binary_seq.clone()
    total = out.numel()
    pos = 0
    while pos < total:
        if float(out[pos].item()) >= 0.5:
            pos += 1
            continue

        start = pos
        while pos < total and float(out[pos].item()) < 0.5:
            pos += 1
        end = pos

        if start > 0 and end < total and end - start <= max_gap_frames:
            out[start:end] = 1.0
    return out


def _chunk_starts(num_frames: int, chunk_frames: int, hop_frames: int) -> List[int]:
    if num_frames <= chunk_frames:
        return [0]

    starts = list(range(0, max(1, num_frames - chunk_frames + 1), max(1, hop_frames)))
    last_start = num_frames - chunk_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _num_fbank_frames(duration_sec: float, sample_rate: int = TARGET_SR) -> int:
    samples = int(round(float(duration_sec) * sample_rate))
    frame_length = int(round(0.025 * sample_rate))
    frame_shift = int(round(0.010 * sample_rate))
    if samples < frame_length:
        return 1
    return 1 + (samples - frame_length) // frame_shift


@dataclass
class DiarizationResult:
    recording_id: str
    segments: List[Dict[str, Any]]
    speaker_prob: torch.Tensor
    speaker_activity: torch.Tensor
    frame_activity: torch.Tensor
    frame_shift_sec: float = FRAME_SHIFT_SEC
    sample_rate: int = TARGET_SR

    def to_dict(self, include_frame_outputs: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "recording_id": self.recording_id,
            "segments": self.segments,
            "frame_shift_sec": self.frame_shift_sec,
            "sample_rate": self.sample_rate,
        }
        if include_frame_outputs:
            out["speaker_prob"] = self.speaker_prob.tolist()
            out["speaker_activity"] = self.speaker_activity.tolist()
            out["frame_activity"] = self.frame_activity.tolist()
        return out


class SpeakerDiarizationPipeline:
    """
    Lightweight inference wrapper for the Train_Ali synthetic diarization model.

    Speaker labels are anonymous query tracks, for example SPEAKER_00. They are
    not persistent identities across unrelated recordings.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "outputs_trainali_synth/best.pt",
        *,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
        chunk_sec: Optional[float] = None,
        hop_sec: Optional[float] = None,
        speaker_activity_threshold: Optional[float] = None,
        exist_threshold: Optional[float] = None,
        min_active_frames: Optional[int] = None,
        merge_gap_sec: float = 0.0,
        relabel_speakers: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = _torch_load(self.checkpoint_path, map_location=self.device)
        checkpoint_cfg = ckpt.get("config", {}) if isinstance(ckpt, Mapping) else {}
        file_cfg = _load_config_file(config_path)
        self.config: Dict[str, Any] = dict(file_cfg or checkpoint_cfg or {})

        self.model = build_model_from_config(self.config, self.device)
        self.model.load_state_dict(get_model_state_from_ckpt(ckpt), strict=True)
        self.model.eval()

        self.sample_rate = TARGET_SR
        self.n_mels = int(_cfg_get(self.config, "model.in_channels", 80))
        self.max_speakers = int(_cfg_get(self.config, "model.max_mix_speakers", 3))
        self.chunk_sec = float(chunk_sec if chunk_sec is not None else _cfg_get(self.config, "data.crop_sec", 4.0))
        self.hop_sec = float(hop_sec if hop_sec is not None else self.chunk_sec / 2.0)
        self.speaker_activity_threshold = float(
            speaker_activity_threshold
            if speaker_activity_threshold is not None
            else _cfg_get(self.config, "validate.speaker_activity_threshold", 0.55)
        )
        self.frame_activity_threshold = float(_cfg_get(self.config, "validate.frame_activity_threshold", 0.5))
        self.exist_threshold = float(
            exist_threshold
            if exist_threshold is not None
            else _cfg_get(self.config, "validate.exist_threshold", 0.5)
        )
        self.min_active_frames = int(
            min_active_frames
            if min_active_frames is not None
            else _cfg_get(self.config, "validate.min_active_frames", 5)
        )
        self.merge_gap_frames = int(round(max(0.0, float(merge_gap_sec)) / FRAME_SHIFT_SEC))
        self.relabel_speakers = bool(relabel_speakers)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path = "outputs_trainali_synth/best.pt", **kwargs):
        return cls(checkpoint_path=checkpoint_path, **kwargs)

    @torch.no_grad()
    def predict_file(self, audio_path: str | Path, recording_id: str | None = None) -> DiarizationResult:
        audio_path = Path(audio_path).expanduser().resolve()
        wav = load_wav_mono(str(audio_path), target_sr=self.sample_rate, trim_silence=False)
        return self.predict_wav(wav, recording_id=recording_id or audio_path.stem)

    @torch.no_grad()
    def predict_wav(self, wav_16k: torch.Tensor, recording_id: str = "audio") -> DiarizationResult:
        wav_16k = wav_16k.detach().cpu().to(torch.float32).flatten()
        if wav_16k.numel() < int(0.1 * self.sample_rate):
            wav_16k = F.pad(wav_16k, (0, int(0.1 * self.sample_rate) - wav_16k.numel()))

        fbank = wav_to_fbank_infer(
            wav_16k,
            n_mels=self.n_mels,
            crop_sec=None,
            crop_mode="none",
        ).float()

        total_frames = int(fbank.size(0))
        chunk_frames = max(1, _num_fbank_frames(self.chunk_sec, self.sample_rate))
        hop_frames = max(1, int(round(self.hop_sec / FRAME_SHIFT_SEC)))
        prob_sum = torch.zeros(total_frames, self.max_speakers, dtype=torch.float32)
        prob_count = torch.zeros(total_frames, 1, dtype=torch.float32)

        for start in _chunk_starts(total_frames, chunk_frames, hop_frames):
            end = min(total_frames, start + chunk_frames)
            chunk = fbank[start:end].unsqueeze(0).to(self.device)
            valid_mask = torch.ones(1, chunk.size(1), dtype=torch.bool, device=self.device)

            _, _, exist_logits, diar_logits, activity_logits = self.model(chunk, valid_mask=valid_mask)
            diar_prob = torch.sigmoid(diar_logits[0]).detach().cpu()
            activity_prob = torch.sigmoid(activity_logits[0]).detach().cpu()
            exist_prob = torch.sigmoid(exist_logits[0]).detach().cpu()

            keep_mask = exist_prob >= self.exist_threshold
            if not bool(keep_mask.any().item()):
                keep_mask = diar_prob.max(dim=0).values >= self.speaker_activity_threshold

            gated_prob = diar_prob.clone()
            gated_prob[:, ~keep_mask] = 0.0
            frame_gate = (activity_prob >= self.frame_activity_threshold).float().unsqueeze(-1)
            gated_prob = gated_prob * frame_gate
            prob_sum[start:end] += gated_prob[: end - start]
            prob_count[start:end] += 1.0

        speaker_prob = prob_sum / prob_count.clamp_min(1.0)
        speaker_activity = (speaker_prob >= self.speaker_activity_threshold).float()

        for speaker_idx in range(speaker_activity.size(1)):
            seq = _remove_short_runs(speaker_activity[:, speaker_idx], self.min_active_frames)
            seq = _fill_short_gaps(seq, self.merge_gap_frames)
            speaker_activity[:, speaker_idx] = _remove_short_runs(seq, self.min_active_frames)

        frame_activity = (speaker_activity.sum(dim=-1) > 0).float()
        segments = self._activity_to_segments(recording_id, speaker_prob, speaker_activity)
        return DiarizationResult(
            recording_id=recording_id,
            segments=segments,
            speaker_prob=speaker_prob,
            speaker_activity=speaker_activity,
            frame_activity=frame_activity,
            sample_rate=self.sample_rate,
        )

    def _activity_to_segments(
        self,
        recording_id: str,
        speaker_prob: torch.Tensor,
        speaker_activity: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        total_frames, total_speakers = speaker_activity.shape
        speaker_name_by_slot = self._speaker_name_by_slot(speaker_activity)

        for speaker_idx in range(total_speakers):
            seq = speaker_activity[:, speaker_idx]
            pos = 0
            while pos < total_frames:
                if float(seq[pos].item()) < 0.5:
                    pos += 1
                    continue

                start = pos
                while pos < total_frames and float(seq[pos].item()) >= 0.5:
                    pos += 1
                end = pos

                if end - start >= self.min_active_frames:
                    score = float(speaker_prob[start:end, speaker_idx].mean().item())
                    segments.append(
                        {
                            "recording_id": recording_id,
                            "speaker": speaker_name_by_slot.get(speaker_idx, f"SPEAKER_{speaker_idx:02d}"),
                            "slot": int(speaker_idx),
                            "raw_speaker": f"SLOT_{speaker_idx:02d}",
                            "start": round(start * FRAME_SHIFT_SEC, 3),
                            "end": round(end * FRAME_SHIFT_SEC, 3),
                            "score": round(score, 6),
                        }
                    )

        segments.sort(key=lambda item: (float(item["start"]), item["speaker"]))
        return segments

    def _speaker_name_by_slot(self, speaker_activity: torch.Tensor) -> Dict[int, str]:
        if not self.relabel_speakers:
            return {idx: f"SPEAKER_{idx:02d}" for idx in range(int(speaker_activity.size(1)))}

        active_slots = []
        for speaker_idx in range(int(speaker_activity.size(1))):
            active_frames = torch.where(speaker_activity[:, speaker_idx] > 0.5)[0]
            if active_frames.numel() == 0:
                continue
            first_frame = int(active_frames[0].item())
            total_frames = int(active_frames.numel())
            active_slots.append((first_frame, -total_frames, speaker_idx))

        active_slots.sort()
        return {
            speaker_idx: f"SPEAKER_{new_idx:02d}"
            for new_idx, (_, _, speaker_idx) in enumerate(active_slots)
        }


def write_rttm(result: DiarizationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in result.segments:
            start = float(seg["start"])
            duration = max(0.0, float(seg["end"]) - start)
            f.write(
                "SPEAKER {recording_id} 1 {start:.3f} {duration:.3f} "
                "<NA> <NA> {speaker} <NA> <NA>\n".format(
                    recording_id=seg["recording_id"],
                    start=start,
                    duration=duration,
                    speaker=seg["speaker"],
                )
            )


def write_json(result: DiarizationResult, path: str | Path, *, include_frame_outputs: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(include_frame_outputs=include_frame_outputs), f, ensure_ascii=False, indent=2)


__all__ = [
    "DiarizationResult",
    "SpeakerDiarizationPipeline",
    "build_model_from_config",
    "write_json",
    "write_rttm",
]
