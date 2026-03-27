# Data Schema

This project uses explicit schemas for both processed training manifests and inference payloads.

## Training Manifest

File format: `JSONL`

Each line must contain:

```json
{
  "pt": "mix_pt/sample_000001.pt",
  "source": "cn_celeb2_single",
  "num_speakers": 2,
  "duration_sec": 4.0,
  "sample_rate": 16000
}
```

### Required Fields

- `pt`: relative path to the processed `.pt` pack.

### Recommended Fields

- `source`: corpus or pipeline stage that produced the sample.
- `num_speakers`: number of active speakers in the clip.
- `duration_sec`: clip duration after crop / pad.
- `sample_rate`: source waveform sample rate before feature extraction.

## Processed `.pt` Pack

Each processed training item should contain:

```python
{
  "fbank": FloatTensor[T, 80],
  "target_matrix": FloatTensor[T, K],
  "target_activity": FloatTensor[T],
  "spk_label": int,
  "target_count": int
}
```

### Semantics

- `fbank`: mel-filterbank feature matrix.
- `target_matrix`: frame-level speaker assignment matrix.
- `target_activity`: frame-level activity target.
- `spk_label`: dominant or reference speaker label for speaker-aware supervision.
- `target_count`: number of speakers active in the clip.

## Inference Request

Recommended JSON envelope for service-oriented deployments:

```json
{
  "schema_version": "sv.infer/v1",
  "request_id": "req_001",
  "audio_uri": "s3://bucket/meeting.wav",
  "sample_rate": 16000,
  "chunk_sec": 4.0,
  "speaker_bank_uri": "file://artifacts/speaker_bank.json",
  "options": {
    "activity_threshold": 0.5,
    "min_active_frames": 3,
    "min_slot_run": 3
  }
}
```

## Inference Response

```json
{
  "schema_version": "sv.result/v1",
  "request_id": "req_001",
  "num_speakers": 2,
  "dominant_speaker": "Alice",
  "dominant_speaker_slot": 1,
  "activity_ratio": 0.74,
  "slots": [
    {
      "slot": 1,
      "name": "Alice",
      "score": 0.93,
      "is_known": true,
      "num_frames": 220,
      "duration_sec": 2.2
    }
  ],
  "segments": [
    {
      "slot": 1,
      "name": "Alice",
      "start_sec": 0.2,
      "end_sec": 1.6,
      "duration_sec": 1.4
    }
  ]
}
```

