"""HuBERT-large + Llama-aligned CARE trainer.

Extends CARE's train_pase.py to:
  1. Load HuBERT-large (facebook/hubert-large-ll60k) instead of WavLM-base.
  2. Use SpeechTextModelHuBERTLarge with dim adapters (1024<->768) and a
     new semantic projector (1024 -> 4096) for Llama-aligned MSE.
  3. Load Llama-3.1-8B mean-pool features (4096-d) as the semantic target,
     produced by scripts/extract_msppodcast_llama_mean.py in this repo.

Paper-faithful settings (CARE, Sec IV-A / IV-D-1):
  - 200,000 OPTIMIZER UPDATES at effective batch 128. The paper's "steps" are
    updates at batch 128, not micro-batches; accum_steps is derived from
    --effective_batch // --batch_size so this holds at any micro-batch size.
  - lr=1e-5 constant, AdamW (0.9, 0.99), wd=0, no LR scheduler
  - lambda = 1 balancing L_sem and L_acoust (paper Eq. 3, --alpha)
  - Backbone split in half: common encoder = first N layers, acoustic encoder =
    last N layers (paper used 6+6 of WavLM-base; --num_layers defaults to
    half the backbone depth, i.e. 12+12 for HuBERT-large).
  - Semantic encoder's transformer layers FROZEN, conv adapters trained
    (paper Fig. 2 + Table V CARE-FT ablation).
  - zero_grad_hook on common_params (only distil_loss updates them)
  - Best checkpoint selected on validation loss
  - cuDNN disabled + deterministic

Deviation from the paper (this project's contribution): the semantic target
ytext is Llama-3.1-8B mean-pool (4096-d) instead of RoBERTa-base mean-pool
(768-d), and the backbone is HuBERT-large instead of WavLM-base.

Usage (Blackwell 6000, 102 GB):
    cd /home/ouo/care_training/CARE/pretraining
    CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \
        /home/ouo/care_training_large/ckpts_hubert_large \
        --batch_size 32 \
        --use_conv True \
        --use_pretrained True \
        --pool_fn avg
"""
import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import HubertModel, RobertaModel

from dataset_pase import SpeechTextDataset
from model_pase_hubert_large import SpeechTextModelHuBERTLarge

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ---- Logger ----------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---- Seeds + cuDNN (paper-faithful) ----------------------------------------
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

# ---- CLI -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train HuBERT-large + Llama-aligned CARE.")
parser.add_argument("checkpoint_dir", type=str,
                    help="Directory to save checkpoints.")
parser.add_argument("--alpha",         default=1.0, type=float,
                    help="Weight for acoustic loss (kept for parity).")
parser.add_argument("--common_model",  default="roberta", type=str)
parser.add_argument("--batch_size",    required=True, type=int,
                    help="Per-step micro-batch. Gradient accumulation keeps the "
                         "effective batch at --effective_batch regardless.")
parser.add_argument("--effective_batch", default=128, type=int,
                    help="Paper's batch size (CARE Sec IV-D-1 = 128). "
                         "accum_steps is derived as effective_batch // batch_size.")
parser.add_argument("--num_layers",    default=None, type=int,
                    help="Layers per encoder. Default = half the backbone depth, "
                         "matching paper Sec IV-A ('common encoder ... first 6 "
                         "layers of WavLM-base, acoustic encoder ... last 6 "
                         "layers of the same'): 12 for HuBERT-large (24 layers), "
                         "6 for HuBERT-base (12 layers).")
parser.add_argument("--use_conv",      default="True", type=str)
parser.add_argument("--use_pretrained",default="True", type=str)
parser.add_argument("--pool_fn",       default="avg", type=str)
parser.add_argument("--energy_weights",default="False", type=str)
parser.add_argument("--pitch_weights", default="False", type=str)
parser.add_argument("--supervised",    default="False", type=str)
parser.add_argument("--hubert_model_id", default="facebook/hubert-large-ll60k", type=str)
parser.add_argument("--wandb",         default="True", type=str,
                    help='"True" to enable WandB; "False" to skip.')
parser.add_argument("--wandb_project", default="care-training-large", type=str)
parser.add_argument("--wandb_run_name",default=None, type=str)
args = parser.parse_args()

# ---- Hyperparams (paper-faithful) ------------------------------------------
BATCH_SIZE   = args.batch_size
LEARNING_RATE = 1e-5
BETAS        = (0.9, 0.99)
EPS          = 1e-6
WEIGHT_DECAY = 0
MAX_NORM     = 10
# Paper Sec IV-D-1: "trained with a learning rate of 1e-5 and a batch size of
# 128 ... trained for a total of 200,000 steps". A "step" there is an optimizer
# update at effective batch 128, NOT a micro-batch — with accum_steps=4 these
# differ by 4x. The loop below counts optimizer updates.
UPDATES      = 200_000
LOG_EVERY    = 100      # updates, for WandB train curves
VALID_EVERY  = 5_000    # updates
CKPT_EVERY   = 10_000   # updates

# Effective batch is pinned to args.effective_batch (default 128) by deriving
# accum_steps from the micro-batch, so --batch_size only trades speed for memory.
if args.effective_batch % args.batch_size != 0:
    raise SystemExit(
        f"--effective_batch ({args.effective_batch}) must be divisible by "
        f"--batch_size ({args.batch_size}); otherwise the effective batch drifts "
        f"off the paper's {args.effective_batch}."
    )
accum_steps = args.effective_batch // args.batch_size


# ---- Wrapper mirroring CARE's EmotionClassifier ----------------------------
class EmotionClassifier(nn.Module):
    """CARE-style wrapper. Adds the semantic projector output for Llama MSE."""

    def __init__(self, joint_model, output_dim):
        super().__init__()
        self.joint_model = joint_model
        # Kept for interface parity with CARE (unused in this trainer).
        self.fc_text = nn.Linear(768 * 2, 768)
        self.act = nn.ReLU()
        self.fc_text_final = nn.Linear(768, output_dim)

    def forward(self, audio):
        speech_feats, pooled_audio, _, _ = self.joint_model(audio)
        # Project pooled_audio (1024) into Llama space (4096) via joint_model's
        # llama_semantic_proj. Returned so the trainer can MSE it against llama_feats.
        pooled_audio_llama = self.joint_model.llama_semantic_proj(pooled_audio)
        return speech_feats, pooled_audio, pooled_audio_llama


def zero_grad_hook(grad):
    """Zero gradients — used on common_params during opensmile_loss backward."""
    return torch.zeros_like(grad)


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- WandB (optional) ----
    use_wandb = args.wandb.lower() == "true" and _HAS_WANDB
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"hubert-large-llama-b{args.batch_size}",
            dir=args.checkpoint_dir,
            config={
                "hubert_model_id": args.hubert_model_id,
                "common_model":    args.common_model,
                "batch_size":      BATCH_SIZE,
                "accum_steps":     accum_steps,
                "effective_batch": BATCH_SIZE * accum_steps,
                "learning_rate":   LEARNING_RATE,
                "betas":           BETAS,
                "weight_decay":    WEIGHT_DECAY,
                "updates":         UPDATES,
                "lambda_acoust":   args.alpha,
                "num_layers":      args.num_layers,
                "use_conv":        args.use_conv,
                "use_pretrained":  args.use_pretrained,
                "pool_fn":         args.pool_fn,
            },
        )
        logger.info(f"WandB run: {wandb.run.url}")

    # ---- Data ----
    if args.supervised == "True":
        train_dataset = SpeechTextDataset(train=True, supervised=True)
        valid_dataset = SpeechTextDataset(train=False, supervised=True)
    else:
        train_dataset = SpeechTextDataset(train=True, supervised=False)
        valid_dataset = SpeechTextDataset(train=False, supervised=False)

    train_loader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate,
        batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, collate_fn=valid_dataset.collate,
        batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, drop_last=True,
    )

    # ---- Models: HuBERT-large + RoBERTa (common_model) ----
    logger.info(f"Loading HuBERT model: {args.hubert_model_id}")
    if args.use_pretrained == "True":
        hubert_model = HubertModel.from_pretrained(args.hubert_model_id)
    else:
        # Random-init HuBERT (rare use case)
        from transformers import HubertConfig
        hubert_model = HubertModel(HubertConfig())
    total_layers = len(hubert_model.encoder.layers)
    logger.info(f"HuBERT layers: {total_layers}, hidden: {hubert_model.config.hidden_size}")

    # Paper Sec IV-A: common encoder = first half of the backbone, acoustic
    # encoder = last half. For WavLM-base that was 6 + 6; for HuBERT-large it
    # is 12 + 12. Leaving this at 6 would use only the bottom half of the
    # 24-layer model and waste the "large" backbone entirely.
    if args.num_layers is None:
        args.num_layers = total_layers // 2
    if args.num_layers * 2 != total_layers:
        logger.warning(
            f"--num_layers {args.num_layers} uses {args.num_layers*2}/{total_layers} "
            f"backbone layers; paper Sec IV-A splits the backbone in half "
            f"({total_layers//2} + {total_layers//2})."
        )
    logger.info(f"num_layers: {args.num_layers}  "
                f"(common encoder = layers 0-{args.num_layers-1}, "
                f"acoustic encoder = layers {args.num_layers}-{args.num_layers*2-1})")
    if use_wandb:
        # num_layers was resolved after wandb.init(), so backfill the real value.
        wandb.config.update({"num_layers": args.num_layers,
                             "backbone_layers": total_layers}, allow_val_change=True)

    roberta_model = RobertaModel.from_pretrained("roberta-base")

    joint_model = SpeechTextModelHuBERTLarge(
        hubert_model, roberta_model,
        num_layers=args.num_layers,
        common_model=args.common_model,
        use_conv=args.use_conv,
        pool_fn=args.pool_fn,
    )
    emo_model = EmotionClassifier(joint_model, output_dim=3).to(device)

    # ---- Parameter grouping (paper-faithful zero_grad_hook design) ----
    # Naming note: CARE's *code* names do not match the *paper's* encoder names.
    #   paper "common encoder"   == code `audio_model`   (first N HuBERT layers)
    #   paper "acoustic encoder" == code `dual_model`    (next N HuBERT layers)
    #   paper "semantic encoder" == code `common_model`  (RoBERTa layers 6-11)
    #                               + downblock/upblock conv adapters
    #
    # common_params: updated ONLY by distil_loss (semantic gradient)
    # dual_params:   updated by both losses (but common_params get zeroed during
    #                opensmile_loss backward)
    #
    # `common_model` (the RoBERTa transformer layers) is deliberately NOT in
    # either group: paper Fig. 2 — "For the semantic encoder the transformer
    # layers are frozen while the convolutional adapters are trained" — and
    # Sec III — "the transformer layers themselves are not updated during
    # training". Table V's CARE-FT ablation shows updating them hurts. Only the
    # conv adapters (downblock/upblock) and the 1024<->768 dim adapters train.
    dual_params, common_params = [], []
    for name, param in emo_model.named_parameters():
        # Common (conv adapters + dim adapters + semantic projector + audio_model)
        if ("upblock" in name or "downblock" in name or
            "audio_model" in name or
            "audio_to_common" in name or "common_to_audio" in name or
            "llama_semantic_proj" in name):
            param.requires_grad = True
            common_params.append(param)
        # Dual (acoustic encoder + PASE+ head)
        elif "dual_model" in name or "joint_model.linear" in name:
            param.requires_grad = True
            dual_params.append(param)
        else:
            # Frozen: RoBERTa transformer layers (paper Fig. 2), conv feature
            # extractor, feature projection, pos_conv, embeddings, unused heads.
            param.requires_grad = False

    logger.info(f"Trainable: common={sum(p.numel() for p in common_params)/1e6:.2f}M  "
                f"dual={sum(p.numel() for p in dual_params)/1e6:.2f}M")

    dual_optimizer = AdamW(
        dual_params, lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )
    common_optimizer = AdamW(
        common_params, lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )

    total_micro = UPDATES * accum_steps
    n_epochs = total_micro // len(train_loader) + 1
    num_steps = 0        # micro-batches
    num_updates = 0      # optimizer updates == paper's "steps"
    micro_since_step = 0
    train_speechtext_loss = 0.0
    train_opensmile_loss = 0.0
    window_speechtext_loss = 0.0
    window_opensmile_loss = 0.0
    window_n = 0
    best_valid_loss = float("inf")

    logger.info(f"PyTorch: {torch.__version__}   CUDA: {torch.version.cuda}")
    logger.info(f"CUDNN enabled={torch.backends.cudnn.enabled} deterministic={torch.backends.cudnn.deterministic}")
    logger.info(f"# GPUs: {torch.cuda.device_count()}")
    logger.info(f"batch_size: {BATCH_SIZE}   accum_steps: {accum_steps}   effective: {BATCH_SIZE*accum_steps}")
    logger.info(f"iterations per epoch: {len(train_loader)}   # epochs: {n_epochs}")
    logger.info(f"target: {UPDATES} optimizer updates @ effective batch "
                f"{BATCH_SIZE*accum_steps}  (= {total_micro} micro-batches)  [paper Sec IV-D-1]")
    logger.info(f"lambda (acoustic loss weight): {args.alpha}")
    logger.info("*" * 60)

    for epoch in range(n_epochs + 1):
        emo_model.train()
        for i, data in enumerate(train_loader):
            num_steps += 1
            wavs, opensmile_feats, llama_feats, _ = data
            wavs = wavs.to(device)
            opensmile_feats = opensmile_feats[:, :249, :].to(device)

            # Forward: get PASE+ prediction + pooled_audio + Llama-projected pool
            speech_feats, pooled_audio, pooled_audio_llama = emo_model(wavs)

            # llama_feats from dataset: shape (B, 4096) after mean-pool loader
            if len(llama_feats.shape) == 3:
                llama_feats = llama_feats[:, -1, :].to(device).squeeze(1)
            else:
                llama_feats = llama_feats.to(device)

            # Paper Eq. 1-3: L = L_sem + lambda * L_acoust, with lambda = 1.
            distil_loss = nn.MSELoss()(pooled_audio_llama, llama_feats.float())
            opensmile_loss = args.alpha * nn.MSELoss()(speech_feats, opensmile_feats.float())

            train_speechtext_loss += distil_loss.item()
            train_opensmile_loss += opensmile_loss.item()
            window_speechtext_loss += distil_loss.item()
            window_opensmile_loss += opensmile_loss.item()
            window_n += 1

            distil_loss = distil_loss / accum_steps
            opensmile_loss = opensmile_loss / accum_steps

            distil_loss.backward(retain_graph=True)

            # Zero common_params during acoustic backward — CARE's key trick.
            hook_handles = [param.register_hook(zero_grad_hook) for param in common_params]
            opensmile_loss.backward()
            for handle in hook_handles:
                handle.remove()

            # Step only on full accumulation boundaries, tracked globally rather
            # than per-epoch: `(i+1) % accum_steps` would fire a partial update
            # at every epoch boundary (3703 % 4 != 0), scaled as if it had
            # accumulated a full effective batch.
            micro_since_step += 1
            if micro_since_step < accum_steps:
                continue
            micro_since_step = 0
            common_optimizer.step()
            dual_optimizer.step()
            common_optimizer.zero_grad()
            dual_optimizer.zero_grad()
            num_updates += 1

            # ---- WandB: log train loss every LOG_EVERY updates ----
            # Windowed mean is what shows learning; the since-start running mean
            # flattens out and hides everything after ~20K steps, so log both.
            if use_wandb and num_updates % LOG_EVERY == 0:
                wandb.log({
                    "train/distil_loss":         window_speechtext_loss / window_n,
                    "train/opensmile_loss":      window_opensmile_loss / window_n,
                    "train/total_loss":          (window_speechtext_loss + window_opensmile_loss) / window_n,
                    "train/distil_loss_avg":     train_speechtext_loss / num_steps,
                    "train/opensmile_loss_avg":  train_opensmile_loss / num_steps,
                    "epoch":                     epoch,
                    "micro_batches":             num_steps,
                }, step=num_updates)
                window_speechtext_loss = window_opensmile_loss = 0.0
                window_n = 0

            # ---- Log + checkpoint every CKPT_EVERY updates ----
            if num_updates % CKPT_EVERY == 0:
                torch.save(emo_model,
                           os.path.join(args.checkpoint_dir, f"model-{num_updates}.pth"))
                logger.info("*" * 40)
                logger.info(f"Update: {num_updates}/{UPDATES}   (micro-batches: {num_steps})")
                logger.info(f"Speech Text Distillation Loss: {train_speechtext_loss/num_steps}")
                logger.info(f"Speech Opensmile Loss:         {train_opensmile_loss/num_steps}")
                logger.info("*" * 40)

            # ---- Validate every VALID_EVERY updates ----
            if num_updates % VALID_EVERY == 0:
                emo_model.eval()
                with torch.no_grad():
                    valid_speechtext_loss = 0.0
                    valid_opensmile_loss = 0.0
                    valid_steps = 0
                    for j, data in enumerate(valid_loader):
                        valid_steps += 1
                        wavs, opensmile_feats, llama_feats, _ = data
                        wavs = wavs.to(device)
                        opensmile_feats = opensmile_feats[:, :249, :].to(device)
                        speech_feats, pooled_audio, pooled_audio_llama = emo_model(wavs)
                        if len(llama_feats.shape) == 3:
                            llama_feats = llama_feats[:, -1, :].to(device).squeeze(1)
                        else:
                            llama_feats = llama_feats.to(device)
                        d_loss = nn.MSELoss()(pooled_audio_llama, llama_feats.float())
                        o_loss = args.alpha * nn.MSELoss()(speech_feats, opensmile_feats.float())
                        valid_speechtext_loss += d_loss.item()
                        valid_opensmile_loss += o_loss.item()

                    total_valid_loss = (valid_speechtext_loss + valid_opensmile_loss) / max(1, valid_steps)

                    # WandB: log val losses at every eval
                    if use_wandb:
                        wandb.log({
                            "val/distil_loss":    valid_speechtext_loss / valid_steps,
                            "val/opensmile_loss": valid_opensmile_loss / valid_steps,
                            "val/total_loss":     total_valid_loss,
                            "val/best_total_loss": min(total_valid_loss, best_valid_loss),
                        }, step=num_updates)

                    if total_valid_loss < best_valid_loss:
                        # Paper Sec IV-D-1: "the best model parameters based on
                        # validation set performance are chosen for evaluation".
                        torch.save(emo_model,
                                   os.path.join(args.checkpoint_dir, f"model-{num_updates}.pth"))
                        torch.save(emo_model,
                                   os.path.join(args.checkpoint_dir, "best.pth"))
                        logger.info("*" * 40)
                        logger.info(f"Update: {num_updates}   (NEW BEST)")
                        logger.info(f"Val Distillation Loss: {valid_speechtext_loss/valid_steps}")
                        logger.info(f"Val Opensmile Loss:    {valid_opensmile_loss/valid_steps}")
                        logger.info("*" * 40)
                        best_valid_loss = total_valid_loss
                emo_model.train()

            if num_updates >= UPDATES:
                logger.info(f"Reached {UPDATES} optimizer updates "
                            f"({num_steps} micro-batches). Saving final and stopping.")
                torch.save(emo_model, os.path.join(args.checkpoint_dir, "final.pth"))
                if use_wandb:
                    wandb.finish()
                return


if __name__ == "__main__":
    train(args)
