# Applying care-training-large patches

Patches modify the CARE codebase (from the original CARE repo) to swap in
HuBERT-large + Llama-aligned semantic supervision. They are **additive**:
we keep the original `model_pase.py` / `train_pase.py` intact and add
`model_pase_hubert_large.py` / `train_pase_hubert_large.py` alongside them,
so the baseline CARE reproduction path remains runnable.

## Prerequisites

1. `/home/ouo/care_training/CARE/` — clone of the original CARE repo
2. `/home/ouo/care_training/data/pase_features/` — PASE+ features (from
   care-training baseline)
3. `/home/ouo/care_training_large/data/llama_features/` — Llama-8B mean-pool
   features (produced by `scripts/extract_msppodcast_llama_mean.py`)

## Apply

```bash
cd /home/ouo/care_training_large
bash patches/apply_hubert_large_patches.sh
```

This will:
1. Copy `patches/model_pase_hubert_large.py` into `CARE/pretraining/`
2. Copy `patches/train_pase_hubert_large.py` into `CARE/pretraining/`
3. Overwrite `CARE/pretraining/config.py` with `patches/care_config.py`
4. Print the resulting file layout for verification.

## Run pretrain

After patches are applied:

```bash
cd /home/ouo/care_training/CARE/pretraining
CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \
    /home/ouo/care_training_large/ckpts_hubert_large \
    --batch_size 8 \
    --num_layers 6 \
    --use_conv True \
    --use_pretrained True \
    --pool_fn avg
```

Blackwell 6000 (102 GB) is required — HuBERT-large + Llama-aligned projector +
AdamW state peaks around ~85 GB.
