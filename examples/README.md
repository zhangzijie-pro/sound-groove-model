# Pretrained Diarization Examples

The default checkpoint is:

```text
outputs_trainali_synth/best.pt
```

## Offline Inference

```powershell
python examples\pretrained_inference_example.py --audio path\to\audio.wav
```

Outputs:

```text
outputs_demo/<recording_id>.json
outputs_demo/<recording_id>.rttm
```

`SPEAKER_00`, `SPEAKER_01`, and `SPEAKER_02` are anonymous diarization tracks, not fixed real identities. By default the interface relabels active tracks to compact names and preserves the internal query slot in `raw_speaker`.

Use this flag to inspect raw slots directly:

```powershell
python examples\pretrained_inference_example.py --audio path\to\audio.wav --keep_raw_slots
```

## Realtime Microphone Test

```powershell
python demos\realtime_diarization_test.py --seconds 30
```

Optional device listing:

```powershell
python demos\realtime_diarization_test.py --list_devices
```

If `sounddevice` is missing:

```powershell
pip install sounddevice
```

## Export

```powershell
python scripts\export.py --ckpt outputs_trainali_synth\best.pt --out_dir outputs\export
```
