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

## Smoke test first

`SpeechTextModelHuBERTLarge` is a port of CARE's WavLM-base model, and the two
backbones differ in ways that only show up at runtime (feature_projection
return type, encoder-layer signature, pre- vs post-LN encoder). Run the shape +
gradient check before starting a 200K-step run:

```bash
cd /home/ouo/care_training/CARE/pretraining
CUDA_VISIBLE_DEVICES=4 python smoke_test_hubert_large.py
```

It must print `SMOKE TEST PASSED`. It also reports peak GPU memory at the given
batch size, which is the cheapest way to size `--batch_size`.

## Run pretrain

After patches are applied:

```bash
cd /home/ouo/care_training/CARE/pretraining
CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \
    /home/ouo/care_training_large/ckpts_hubert_large \
    --batch_size 32 \
    --use_conv True \
    --use_pretrained True \
    --pool_fn avg
```

Blackwell 6000 (102 GB) is required — HuBERT-large + Llama-aligned projector +
AdamW state peaks around ~85 GB. If `--batch_size 32` OOMs, lower it: the
effective batch stays pinned at 128 via gradient accumulation, so only speed
changes, not the optimization.

## Paper-faithfulness checklist (CARE)

| Paper | Where | This code |
|---|---|---|
| batch size 128, lr 1e-5, AdamW | Sec IV-D-1 | `--effective_batch 128` (accum derived), `LEARNING_RATE=1e-5` |
| **200,000 steps** = optimizer updates at batch 128 | Sec IV-D-1 | `UPDATES=200_000`, loop counts updates not micro-batches |
| best model by validation | Sec IV-D-1 | `best.pth` on val loss, every 5K updates |
| λ = 1 for L_sem + λ·L_acoust | Sec IV-D-1, Eq. 3 | `--alpha 1.0` applied to the acoustic loss |
| common enc. = first half, acoustic enc. = last half of backbone | Sec IV-A | `--num_layers` defaults to `total_layers // 2` (12 for HuBERT-large) |
| semantic enc. = **last** N layers of the text model | Sec IV-A | `roberta.encoder.layer[-num_layers:]` → `[12:24]` of roberta-large |
| **semantic enc. transformer layers frozen**, conv adapters trained | Fig. 2, Sec III, Table V (CARE-FT) | `common_model` in neither optimizer group |
| conv adapter: kernel 5, ×3 down before / ×3 up after **each** layer, channels = text width | Sec III | `downblock` / `upblock`, `range(num_layers)`, 1024 ch |
| no dim adapters at the speech/text boundary (WavLM-base 768 == RoBERTa-base 768) | Sec III, Fig. 2 | roberta-large is 1024 == HuBERT-large 1024 → `nn.Identity()` |
| PASE+ 256-d @ 50 Hz, 5 s crops | Sec III, IV-D-1 | 249 frames, `opensmile_feats[:, :249, :]` |

**Intentional deviations** (this project's contribution):
- semantic target `y_text` = Llama-3.1-8B mean-pool (4096-d), not RoBERTa-base
  mean-pool (768-d), via `llama_semantic_proj` (1024 → 2048 → 4096)
- backbone = HuBERT-large (24 layers, 1024-d), not WavLM-base (12 layers, 768-d)
- semantic encoder = RoBERTa-large, chosen so the width match (and therefore
  the adapter-free boundary) survives the backbone swap

## Checkpoints

Files written to `<checkpoint_dir>`:

| File | Contents | Retention |
|---|---|---|
| `best.pth` | model state_dict + config, lowest validation loss | always kept — **this is the one downstream uses** |
| `final.pth` | model state_dict + config at 200K updates | always kept |
| `model-<update>.pth` | model state_dict + config, every 10K updates | newest `--keep_last` (default 3) |
| `last.pth` | model + both optimizers + counters + RNG state | rolling, overwritten each eval |

Checkpoints are **state_dicts, not pickled modules**. Load them with the helper,
which rebuilds the architecture from the config stored alongside the weights:

```python
from model_pase_hubert_large import load_pretrained
model = load_pretrained("ckpts_hubert_large/best.pth", device="cuda")
fusion_out, pooled_audio, fusion_out_aud = model.extract_audio_features(wavs)
```

Any `torch.load(path)` in the downstream extractor that expected a whole module
must be switched to this.

To resume after a preemption or crash:

```bash
CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \
    /home/ouo/care_training_large/ckpts_hubert_large \
    --batch_size 32 --resume auto
```

`--resume auto` is a no-op when `last.pth` does not exist, so it is safe to put
in the launch script from the start. The shuffled batch order inside the
interrupted epoch is not replayed, so a resumed run is not bit-identical to an
unbroken one.

## Downstream layer stack changed — update the SUPERB config

The paper aggregates **13 layers × T × 1536**: 6 pairs of (semantic ‖ acoustic)
concatenated to 1536, plus the conv extractor output and the common encoder's 6
layers duplicated 768→1536.

Scaling the backbone rescales this. At `--num_layers 12` with HuBERT-large +
RoBERTa-large it becomes **25 layers × T × 2048**: 12 concatenated pairs, plus
1 conv output + 12 common-encoder layers duplicated 1024→2048. The convex
combination in the downstream head must be resized from 13 to 25 weights and
its input from 1536 to 2048, or feature extraction will silently mis-slice.
Both the trainer and the smoke test print the expected stack size at startup.
