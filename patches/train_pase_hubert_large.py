"""HuBERT-large + Llama-aligned CARE trainer.

Extends CARE's train_pase.py to:
  1. Load HuBERT-large (facebook/hubert-large-ll60k) instead of WavLM-base.
  2. Use SpeechTextModelHuBERTLarge with dim adapters (1024<->768) and a
     new semantic projector (1024 -> 4096) for Llama-aligned MSE.
  3. Load Llama-3.1-8B mean-pool features (4096-d) as the semantic target,
     produced by scripts/extract_msppodcast_llama_mean.py in this repo.

Paper-faithful settings preserved from CARE Sec IV-D-1:
  - 200K steps (not the 800K default)
  - Effective batch 128 (via accum_steps=4 when --batch_size 32)
  - lr=1e-5 constant, AdamW (0.9, 0.99), wd=0, no LR scheduler
  - zero_grad_hook on common_params (only distil_loss updates them)
  - cuDNN disabled + deterministic

Usage (Blackwell 6000, 102 GB):
    cd /home/ouo/care_training/CARE/pretraining
    CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \
        /home/ouo/care_training_large/ckpts_hubert_large \
        --batch_size 32 \
        --num_layers 6 \
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
parser.add_argument("--batch_size",    required=True, type=int)
parser.add_argument("--num_layers",    default=6, type=int)
parser.add_argument("--use_conv",      default="True", type=str)
parser.add_argument("--use_pretrained",default="True", type=str)
parser.add_argument("--pool_fn",       default="avg", type=str)
parser.add_argument("--energy_weights",default="False", type=str)
parser.add_argument("--pitch_weights", default="False", type=str)
parser.add_argument("--supervised",    default="False", type=str)
parser.add_argument("--hubert_model_id", default="facebook/hubert-large-ll60k", type=str)
args = parser.parse_args()

# ---- Hyperparams (paper-faithful) ------------------------------------------
BATCH_SIZE   = args.batch_size
LEARNING_RATE = 1e-5
BETAS        = (0.9, 0.99)
EPS          = 1e-6
WEIGHT_DECAY = 0
MAX_NORM     = 10
STEPS        = 200_000   # paper Sec IV-D-1
accum_steps  = 4         # effective batch = 32*4 = 128 (matches paper)


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
    logger.info(f"HuBERT-large layers: {len(hubert_model.encoder.layers)}, hidden: {hubert_model.config.hidden_size}")

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
    # common_params: updated ONLY by distil_loss (semantic gradient)
    # dual_params:   updated by both losses (but common_params get zeroed during
    #                opensmile_loss backward)
    dual_params, common_params = [], []
    for name, param in emo_model.named_parameters():
        # Common (semantic path + adapters + semantic projector + audio_model)
        if ("upblock" in name or "downblock" in name or
            "audio_model" in name or
            "audio_to_common" in name or "common_to_audio" in name or
            "llama_semantic_proj" in name):
            param.requires_grad = True
            common_params.append(param)
        # Dual (acoustic path + PASE+ head)
        elif "dual_model" in name or "joint_model.linear" in name:
            param.requires_grad = True
            dual_params.append(param)
        else:
            # Everything else (embeddings, unused fc_text head, etc.) is frozen.
            param.requires_grad = False

    logger.info(f"Trainable: common={sum(p.numel() for p in common_params)/1e6:.2f}M  "
                f"dual={sum(p.numel() for p in dual_params)/1e6:.2f}M")

    dual_optimizer = AdamW(
        dual_params, lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )
    common_optimizer = AdamW(
        common_params, lr=LEARNING_RATE, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY,
    )

    n_epochs = STEPS // len(train_loader) + 1
    num_steps = 0
    train_speechtext_loss = 0.0
    train_opensmile_loss = 0.0
    best_valid_loss = float("inf")

    logger.info(f"PyTorch: {torch.__version__}   CUDA: {torch.version.cuda}")
    logger.info(f"CUDNN enabled={torch.backends.cudnn.enabled} deterministic={torch.backends.cudnn.deterministic}")
    logger.info(f"# GPUs: {torch.cuda.device_count()}")
    logger.info(f"batch_size: {BATCH_SIZE}   accum_steps: {accum_steps}   effective: {BATCH_SIZE*accum_steps}")
    logger.info(f"iterations per epoch: {len(train_loader)}   # epochs: {n_epochs}   total steps: {STEPS}")
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

            distil_loss = nn.MSELoss()(pooled_audio_llama, llama_feats.float())
            opensmile_loss = nn.MSELoss()(speech_feats, opensmile_feats.float())

            train_speechtext_loss += distil_loss.item()
            train_opensmile_loss += opensmile_loss.item()

            distil_loss = distil_loss / accum_steps
            opensmile_loss = opensmile_loss / accum_steps

            distil_loss.backward(retain_graph=True)

            # Zero common_params during acoustic backward — CARE's key trick.
            hook_handles = [param.register_hook(zero_grad_hook) for param in common_params]
            opensmile_loss.backward()
            for handle in hook_handles:
                handle.remove()

            if ((i + 1) % accum_steps == 0) or (i + 1 == len(train_loader)):
                common_optimizer.step()
                dual_optimizer.step()
                common_optimizer.zero_grad()
                dual_optimizer.zero_grad()

            # ---- Log every 10K steps ----
            if num_steps % 10000 == 0:
                torch.save(emo_model,
                           os.path.join(args.checkpoint_dir, f"model-{num_steps}.pth"))
                logger.info("*" * 40)
                logger.info(f"Step: {num_steps}")
                logger.info(f"Speech Text Distillation Loss: {train_speechtext_loss/num_steps}")
                logger.info(f"Speech Opensmile Loss:         {train_opensmile_loss/num_steps}")
                logger.info("*" * 40)

            # ---- Validate every 5K steps ----
            if num_steps % 5000 == 0:
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
                        o_loss = nn.MSELoss()(speech_feats, opensmile_feats.float())
                        valid_speechtext_loss += d_loss.item()
                        valid_opensmile_loss += o_loss.item()

                    total_valid_loss = (valid_speechtext_loss + valid_opensmile_loss) / max(1, valid_steps)
                    if total_valid_loss < best_valid_loss:
                        torch.save(emo_model,
                                   os.path.join(args.checkpoint_dir, f"model-{num_steps}.pth"))
                        torch.save(emo_model,
                                   os.path.join(args.checkpoint_dir, "best.pth"))
                        logger.info("*" * 40)
                        logger.info(f"Step: {num_steps}   (NEW BEST)")
                        logger.info(f"Val Distillation Loss: {valid_speechtext_loss/valid_steps}")
                        logger.info(f"Val Opensmile Loss:    {valid_opensmile_loss/valid_steps}")
                        logger.info("*" * 40)
                        best_valid_loss = total_valid_loss
                emo_model.train()

            if num_steps >= STEPS:
                logger.info(f"Reached STEPS={STEPS}. Saving final and stopping.")
                torch.save(emo_model, os.path.join(args.checkpoint_dir, "final.pth"))
                return


if __name__ == "__main__":
    train(args)
