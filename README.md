# Speaker-Aware End-to-End Diarization

This branch implements a query-based speaker diarization model for the task:

- input: single-channel log-fbank features
- output: frame-level speaker activity masks
- optional export: full-recording RTTM segments

The current system is designed for `AISHELL-4 near` and `VoxConverse`.

## What The Model Does

The model answers three questions jointly:

1. Is there speech in this frame?
2. How many speaker queries are active in this chunk?
3. Which query is speaking at each frame?

The output stack is:

- `activity_logits [B, T]`: explicit speech / non-speech activity
- `exist_logits [B, N]`: whether each speaker query exists in the chunk
- `diar_logits [B, T, N]`: frame-level query activity masks

At inference time, diarization masks are decoded with:

- frame activity gate
- query existence gate
- per-query frame thresholding
- short-run cleanup
- cross-chunk prototype matching

This gives final segments in RTTM form:

`[start, end, speaker_id]`

## Architecture

The model is:

`AcousticEncoder -> Query Decoder / LSTM Attractor Decoder -> Dot-Product Assignment Head`

### 1. Acoustic Encoder

Implemented in `speaker_verification/models/encoder/acoustic_encoder.py`.

The encoder uses:

- convolutional front-end
- Res2-style local aggregation
- WRR / EPA hybrid blocks for wider temporal mixing
- feed-forward adapter before projection to `d_model`

It produces frame embeddings:

`z = [B, T, D]`

### 2. Query / Attractor Decoder

Implemented in `speaker_verification/models/decoder/query_decoder.py` and `speaker_verification/models/decoder/attractor_decoder_lstm.py`.

The default mode is `query`.

The decoder consumes frame embeddings and emits:

- `attractors [B, N, D]`
- `exist_logits [B, N]`

Each attractor corresponds to one speaker query slot.

### 3. Assignment Head

Implemented in `speaker_verification/models/heads/diar_assign.py`.

Frame embeddings and attractors are matched by normalized dot product:

`diar_logits[t, n] = <frame_embed_t, attractor_n>`

### 4. Explicit Activity Head

Implemented in `speaker_verification/models/eend_query_model.py`.

This branch was added to solve a real weakness of the pure query-mask formulation:

- speaker queries may learn *who* is active
- but frame-level speech / silence boundaries can still drift

The activity head predicts:

- `activity_logits [B, T]`

This is used in two places:

- as an auxiliary training target
- as a hard gate during decoding

That makes the model less dependent on query masks alone for silence rejection.

## Training Objective

Implemented in `speaker_verification/loss/multi_task.py`.

The total loss is:

`L = L_pit + λ_act L_activity + λ_exist L_exist + λ_pull L_pull + λ_sep L_sep + λ_smooth L_smooth`

Where:

- `L_pit`: permutation-invariant diarization BCE
- `L_activity`: frame-level speech activity BCE
- `L_exist`: query existence BCE
- `L_pull`: bring active frames toward matched attractors
- `L_sep`: push different active attractors apart
- `L_smooth`: temporal smoothing on diarization logits

This is still one end-to-end optimization graph. The activity head is an auxiliary supervision branch, not an external preprocessing module.

## Why This Model Is Different From `dev`

The older `dev` branch used `ResoWave + REAT`.

That design was closer to:

- strong frame encoder
- fixed-slot mask head
- mask-derived activity / count logic

The current `E2E` branch is different in four important ways:

1. It uses learned speaker queries / attractors instead of only fixed output slots.
2. It optimizes query existence directly instead of inferring everything from raw mask logits.
3. It performs cross-chunk prototype matching at inference time, so local query slots can be mapped into global speaker identities.
4. It now includes an explicit frame activity auxiliary head, which the `dev` branch did not have.

In practice:

- `dev` is simpler and easier to train as a mask-only diarization model.
- `E2E` is closer to EEND-style diarization research and more suitable for paper writing.
- `E2E` also aligns better with future speaker bank / speaker verification integration.

## Is This Really End-to-End?

Short answer: `yes` for diarization, but not in the most extreme raw-waveform sense.

More precisely:

- It is **end-to-end diarization** because the network maps acoustic features directly to diarization outputs without external clustering during training.
- It is **not raw-waveform end-to-end**, because the input is log-fbank rather than raw waveform.
- It is **not fully causal streaming** yet, because training and inference still use overlapping chunks.

So the correct paper wording is:

`end-to-end speaker diarization with query-based attractors and activity-aware decoding`

That wording is technically defensible.

## Data Pipeline

Preprocessing is intentionally minimal.

Raw datasets are expected under:

```text
../datasets/
  AISHELL-4-near/
    Train_Ali_near/
    Test_Ali_near/
  voxconverse/
    dev/
    test/
```

This matches the preprocessor lookup path `processed/../../datasets`.

Run:

```bash
python processed/build_processed_dataset.py
```

This generates:

```text
processed/real_diar_dataset/
  train_manifest.jsonl
  val_manifest.jsonl
  test_manifest.jsonl
  spk2id.json
  dataset_meta.json
  packs/
```

`train` and `val` are split from:

- `Train_Ali_near`
- `voxconverse/dev`

`test` is preserved from:

- `Test_Ali_near`
- `voxconverse/test`

This means the test set remains available for final paper evaluation.

## Train

```bash
python train.py
```

Main config is `configs/experiment.yaml`.

## Evaluate

Full-recording RTTM export and optional DER/JER scoring:

```bash
python scripts/eval_diarization.py --ckpt outputs_e2e/best.pt
```

The evaluation script reads:

- `processed/real_diar_dataset/test_manifest.jsonl`

and writes predicted RTTM files under:

- `outputs_eval/pred_rttm`

## Current Limitations

- Training still uses chunked features rather than full-sequence transformer memory.
- Query PIT can become expensive if `max_mix_speakers` is pushed too high.
- The model is closer to near-real-time than strict low-latency streaming.
- Final paper-quality benchmarking should rely on full-recording RTTM evaluation, not only chunk-level validation DER.

## Recommended Paper Positioning

Use this model as:

- a query-based end-to-end diarization backbone
- with explicit frame activity supervision
- plus cross-chunk prototype matching for global speaker consistency

That is a coherent research story, and it is meaningfully different from the older `dev` branch.
