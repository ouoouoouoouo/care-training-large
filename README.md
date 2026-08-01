# care-training-large

Extending [`care-training`](https://github.com/ouoouoouoouo/care-training) with a
larger audio backbone (HuBERT-large) and a stronger semantic supervision target
(Llama-3.1-8B mean-pool) for improved multimodal SER fusion in
[`merits-l-llama`](https://github.com/ouoouoouoouo/merits-l-llama).

## What is different from the baseline

| Component                    | care-training (baseline)         | care-training-large (this)           |
| ---------------------------- | -------------------------------- | ------------------------------------ |
| Audio backbone               | WavLM-base (95M, 12 layers)      | **HuBERT-large (316M, 24 layers)**   |
| Audio hidden dim             | 768                              | **1024**                             |
| Semantic supervision target  | RoBERTa-base mean-pool (768-d)   | **Llama-3.1-8B mean-pool (4096-d)**  |
| Acoustic supervision         | PASE+ (256-d, unchanged)         | PASE+ (256-d, unchanged)             |
| Dual encoder                 | 6 layers (768-d)                 | 6 layers (1024-d)                    |
| Common encoder + adapters    | 6 layers (768-d)                 | 6 layers (1024-d)                    |
| Semantic projector           | none (direct MSE, 768→768)       | **linear(1024→4096)** on audio side  |
| Training steps               | 200K                             | 200K                                 |
| Effective batch              | 128                              | 128                                  |

The rest — MSP-PODCAST pretraining data, PASE+ acoustic supervision, dual
acoustic/common semantic split, `zero_grad_hook`, AdamW hyperparameters — is
kept identical to the CARE baseline to make ablation results directly attributable
to the backbone / semantic target swap.

## Motivation

Our previous experiments (`merits-l-llama`) established that:

1. Upgrading the text branch from RoBERTa-large to Llama-3.1-8B improves
   Stage III fusion by +2.6 pp (0.8305 → 0.8567 mean).
2. Vanilla WavLM-base and vanilla HuBERT-base performed comparably to CARE on
   Stage III fusion (0.85 range) — with HuBERT slightly better paired with the
   weaker RoBERTa branch, CARE slightly better with the stronger Llama branch.
3. Fusion appears bottlenecked by the *complementarity* between text and audio
   representations, not by absolute audio-branch accuracy.

This repository tests whether a larger audio backbone (HuBERT-large) paired with
a semantic target aligned to the actual downstream text encoder (Llama-3.1-8B)
can meaningfully exceed the current best fusion result
(Llama LoRA + CARE = 0.8567 mean, 0.8746 best).

## Repository layout

```text
care-training-large/
├── patches/
│   ├── care_config.py                       # Local paths for the cluster
│   ├── model_pase_hubert_large.py           # NEW model: HuBERT-large + Llama semantic
│   ├── train_pase_hubert_large.py           # NEW trainer for the above
│   └── apply_hubert_large_patches.sh        # Applies patches into CARE/ tree
├── scripts/
│   ├── extract_msppodcast_llama_mean.py     # Llama-8B mean-pool on 149K MSP-PODCAST
│   └── extract_iemocap_hubert_care_features.py  # Downstream extraction
├── requirements.txt
└── thesis/                                  # LaTeX tables (added after eval)
```

The CARE codebase itself lives at `/home/ouo/care_training/CARE/pretraining/`
(clone of the original CARE release). We modify it via patches instead of
forking the whole thing.

## Pipeline

```
Prep:
  1. Reuse Whisper transcripts from care-training  (already done)
  2. Reuse PASE+ 256-d @ 100Hz features             (already done)
  3. Extract Llama-3.1-8B mean-pool on all 149K   ← new (~4 hours GPU)
                    ↓
Pretrain:
  4. Apply patches to CARE/                         (30 seconds)
  5. Train HuBERT-large + PASE+ + Llama semantic
     on MSP-PODCAST for 200K steps                  ← ~30 hours GPU
                    ↓
Downstream (in merits-l-llama):
  6. Extract IEMOCAP features with new encoder
  7. Downstream Audio Stage I / II
  8. Stage III fusion 5-seed with Llama text
```

## Prerequisites

* GPU: NVIDIA RTX PRO 6000 Blackwell (102 GB) — required for HuBERT-large
  + AdamW state (~85 GB peak). Torch nightly cu128 environment.
* Access to `meta-llama/Meta-Llama-3.1-8B` on Hugging Face.
* `care-training` baseline artifacts:
    - `data/pase_features/` — 256-d PASE+ features @ 100Hz for 149K utts
    - `data/whisper_transcripts/` — Whisper-large-v3 transcripts
    - `/home/ouo/care_training/CARE/` — CARE original clone

## Not included

* WavLM-large variant (would require its own training pass).
* IEMOCAP downstream / Stage III fusion code — that lives in `merits-l-llama`.
* Original CARE weights (available in `care-training/ckpts_faithful/`).

## Reference

Extends CARE (Dutta, "Leveraging Content and Acoustic Representations for
Speech Emotion Recognition") as used in MERITS-L (Dutta and Ganapathy, ICASSP
2025) and consumed by
[`merits-l-llama`](https://github.com/ouoouoouoouo/merits-l-llama).

This is an independent extension. Not an official implementation.
