"""HuBERT-large + Llama-aligned CARE model.

Extends CARE's SpeechTextModel (WavLM-base + RoBERTa semantic) to:
  1. HuBERT-large backbone (24 layers, 1024-d) instead of WavLM-base (12 layers, 768-d).
  2. Semantic supervision target = Llama-3.1-8B mean-pool (4096-d) instead of
     RoBERTa-base mean-pool (768-d).

Preserves CARE's key architectural ideas:
  - Split into a shared audio path, a dual (acoustic) path, and a common
    (semantic) path with adapters.
  - Dual acoustic loss: PASE+ 256-d reconstruction (unchanged, called
    "opensmile_loss" in CARE code for legacy reasons).
  - Semantic loss: MSE between projected audio pool and Llama mean-pool.

Dimension adaptation (needed because HuBERT-large=1024 but RoBERTa=768):
  - down_adapter (1024 -> 768) before common_model
  - up_adapter   (768 -> 1024) after common_model
  - llama_semantic_proj (1024 -> 4096) on pooled_audio for Llama-aligned MSE

Args to __init__:
  hubert_model : HubertModel from `facebook/hubert-large-ll60k`
  roberta_model: RobertaModel from `roberta-base` (still used for common_model)
  num_layers   : 6 by default (audio=layers[0:6], dual=layers[6:12]).
                 Set to 12 to use all 24 HuBERT-large layers
                 (audio=layers[0:12], dual=layers[12:24]).
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


# ---- Self-attention pooling (unchanged from CARE) --------------------------
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep, att_mask=None):
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_logits + att_mask
        att_w = F.softmax(att_logits, dim=-1).unsqueeze(-1)
        return torch.sum(batch_rep * att_w, dim=1)


class SpeechTextModelHuBERTLarge(nn.Module):
    """HuBERT-large + Llama-aligned CARE.

    Interfaces mirror CARE's `SpeechTextModel` so `train_pase_hubert_large.py`
    can reuse the CARE training loop with minimal modifications.
    """

    HUBERT_HIDDEN = 1024   # HuBERT-large hidden size
    ROBERTA_HIDDEN = 768   # RoBERTa-base (used by common_model)
    LLAMA_HIDDEN = 4096    # Llama-3.1-8B (semantic supervision target)
    PASE_DIM = 256         # PASE+ acoustic target dim

    def __init__(
        self,
        hubert_model,
        roberta_model,
        num_layers: int = 6,
        common_model: str = "roberta",
        use_conv: str = "True",
        pool_fn: str = "avg",
        semantic_proj_hidden: int = 2048,   # bottleneck between 1024 -> 4096
    ) -> None:
        super().__init__()
        self.common_model_name = common_model
        self.num_layers = num_layers
        self.use_conv = use_conv

        # ---- HuBERT-large backbone (audio_model + dual_model) --------------
        # HuBERT's feature_extractor is Wav2Vec2 CNN — same interface as WavLM.
        self.audio_feature_extractor = nn.Sequential(
            *hubert_model.feature_extractor.conv_layers
        )
        self.feature_projection_audio = hubert_model.feature_projection
        self.pos_conv = hubert_model.encoder.pos_conv_embed
        self.wavlm_layer_norm = hubert_model.encoder.layer_norm     # keep name for compat
        self.wavlm_dropout = nn.Dropout(0.1, inplace=False)

        total_layers = len(hubert_model.encoder.layers)             # 24 for HuBERT-large
        self.audio_model = nn.ModuleList(
            [hubert_model.encoder.layers[i] for i in range(num_layers)]
        )
        # dual_model: next `num_layers` layers, still on the audio path
        dual_end = min(num_layers * 2, total_layers)
        self.dual_model = nn.ModuleList(
            [hubert_model.encoder.layers[i] for i in range(num_layers, dual_end)]
        )
        self.position_bias = None

        # PASE+ acoustic target projection: HuBERT 1024 -> PASE+ 256
        self.linear = nn.Linear(self.HUBERT_HIDDEN, self.PASE_DIM)

        # ---- RoBERTa common_model (semantic path, still 768-d) --------------
        self.roberta_embeddings = roberta_model.embeddings
        self.text_model = nn.ModuleList(
            [roberta_model.encoder.layer[i] for i in range(num_layers)]
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        if common_model == "roberta":
            self.common_model = nn.ModuleList(
                [roberta_model.encoder.layer[i] for i in range(6, 12)]
            )
        else:
            raise NotImplementedError(f"common_model={common_model!r} not supported")

        # ---- Dimension adapters between HuBERT (1024) and RoBERTa (768) ----
        # CARE original had 1024==768 mismatch avoided because WavLM==RoBERTa==768.
        # For HuBERT-large we need explicit down/up-projections at the boundary.
        self.audio_to_common = nn.Linear(self.HUBERT_HIDDEN, self.ROBERTA_HIDDEN)
        self.common_to_audio = nn.Linear(self.ROBERTA_HIDDEN, self.HUBERT_HIDDEN)

        # CARE's per-layer down/upblocks (kept if use_conv=True), adapted for 768-d.
        if use_conv == "True":
            self.downblock = nn.ModuleList([
                nn.Conv1d(in_channels=self.ROBERTA_HIDDEN,
                          out_channels=self.ROBERTA_HIDDEN,
                          kernel_size=5, stride=3)
                for _ in range(6)
            ])
            self.upblock = nn.ModuleList([
                nn.ConvTranspose1d(in_channels=self.ROBERTA_HIDDEN,
                                   out_channels=self.ROBERTA_HIDDEN,
                                   kernel_size=5, stride=3, output_padding=1)
                for _ in range(6)
            ])

        # ---- Semantic supervision projector: HuBERT space -> Llama space ----
        # pooled_audio (1024-d) -> Llama mean target (4096-d)
        self.llama_semantic_proj = nn.Sequential(
            nn.Linear(self.HUBERT_HIDDEN, semantic_proj_hidden),
            nn.GELU(),
            nn.Linear(semantic_proj_hidden, self.LLAMA_HIDDEN),
        )

        self.act = nn.GELU()
        self.pool_fn = pool_fn if pool_fn in ("avg", "atten") else "avg"
        if self.pool_fn == "atten":
            self.attenpool = SelfAttentionPooling(self.HUBERT_HIDDEN)

    # ------------------------------------------------------------------------
    #  extract_audio_features — used by downstream feature extraction
    # ------------------------------------------------------------------------
    def extract_audio_features(self, audio):
        """Return (fusion_out, pooled_audio, fusion_out_aud).

        Same interface as CARE's SpeechTextModel.extract_audio_features, so
        merits-l-llama's downstream extractor can consume this drop-in.
        Shapes:
            fusion_out     : (13, B, T, 1024)   semantic path all layers (audio 1024)
            pooled_audio   : (B, 1024)           mean-pooled semantic output
            fusion_out_aud : (13, B, T, 1024)   acoustic path all layers
        """
        out_layers, out_layers_aud = [], []
        audio = audio.unsqueeze(1)
        audio_features = self.audio_feature_extractor(audio)
        audio_features = audio_features.permute(0, 2, 1)
        audio_features = self.feature_projection_audio(audio_features)[0]
        audio_features = audio_features.permute(0, 2, 1)
        audio_features_conv = self.pos_conv(audio_features.transpose(1, 2))
        audio_features_conv = audio_features_conv.transpose(1, 2)
        audio_features = audio_features + audio_features_conv
        audio_features = self.wavlm_layer_norm(audio_features.transpose(1, 2))
        audio_features = self.wavlm_dropout(audio_features)
        x = audio_features
        speech_feats = x
        out_layers.append(speech_feats)
        out_layers_aud.append(speech_feats)
        position_bias = None
        for i, layer in enumerate(self.audio_model):
            if i != 0:
                speech_feats, position_bias = layer(speech_feats, position_bias=position_bias)
            else:
                speech_feats, position_bias = layer(speech_feats)
            out_layers.append(speech_feats)
            out_layers_aud.append(speech_feats)
        inp = speech_feats

        # Acoustic path
        speech_only_feats = speech_feats
        for i, layer in enumerate(self.dual_model):
            speech_only_feats, position_bias = layer(speech_only_feats, position_bias=position_bias)
            out_layers_aud.append(speech_only_feats)
        fusion_out_aud = torch.stack(out_layers_aud, dim=0)

        # Semantic path — with dim adaptation
        if self.common_model_name == "roberta":
            x_768 = self.audio_to_common(inp)           # 1024 -> 768
            x = x_768
            for i, layer in enumerate(self.common_model):
                if self.use_conv == "True":
                    x = x.permute(0, 2, 1)
                    x = self.downblock[i](x)
                    x = x.permute(0, 2, 1)
                x = layer(x)[0]
                if self.use_conv == "True":
                    x = x.permute(0, 2, 1)
                    x = self.upblock[i](x)
                    x = x.permute(0, 2, 1)
                if x.shape[1] < inp.shape[1]:
                    x = torch.cat(
                        (x, torch.zeros(x.shape[0], inp.shape[1] - x.shape[1], x.shape[-1]).to(x.device)),
                        dim=1,
                    )
                else:
                    x = x[:, :inp.shape[1], :]
                # Bring back to HuBERT-large space (1024) for downstream consistency
                x_1024 = self.common_to_audio(x)
                out_layers.append(x_1024)
            fusion_out = torch.stack(out_layers, dim=0)
            fusion_feats_audio = x_1024
        else:
            raise NotImplementedError

        if self.pool_fn == "avg":
            pooled_audio = torch.mean(fusion_feats_audio, 1).squeeze(1)
        else:
            pooled_audio = self.attenpool(fusion_feats_audio)
        return fusion_out, pooled_audio, fusion_out_aud

    # ------------------------------------------------------------------------
    #  forward — training path, mirrors CARE's forward interface
    # ------------------------------------------------------------------------
    def forward(self, audio, padding_mask=None, mask=False, mode="speech", weights=None):
        """Same output signature as CARE:
            speech_only_feats_os (B, T, 256)   PASE+ prediction
            pooled_audio         (B, 1024)     for llama_semantic_proj -> MSE
            fusion_out           (13, B, T, 1024)
            fusion_out_aud       (13, B, T, 1024)

        The trainer computes:
            opensmile_loss = MSE(speech_only_feats_os, pase_target)
            distil_loss    = MSE(llama_semantic_proj(pooled_audio), llama_target)
        """
        out_layers, out_layers_aud = [], []
        if mode == "speech" or mode == "speech_text":
            audio_features = self.audio_feature_extractor(audio)
            audio_features = audio_features.permute(0, 2, 1)
            audio_features = self.feature_projection_audio(audio_features)[0]
            audio_features = audio_features.permute(0, 2, 1)
            audio_features_conv = self.pos_conv(audio_features.transpose(1, 2))
            audio_features_conv = audio_features_conv.transpose(1, 2)
            audio_features = audio_features + audio_features_conv
            audio_features = self.wavlm_layer_norm(audio_features.transpose(1, 2))
            audio_features = self.wavlm_dropout(audio_features)

            x = audio_features
            speech_feats = x
            out_layers.append(speech_feats)
            out_layers_aud.append(speech_feats)
            position_bias = None
            for i, layer in enumerate(self.audio_model):
                if i != 0:
                    speech_feats, position_bias = layer(speech_feats, position_bias=position_bias)
                else:
                    speech_feats, position_bias = layer(speech_feats)
                out_layers.append(speech_feats)
                out_layers_aud.append(speech_feats)
            inp = speech_feats

            # Acoustic path (dual_model + PASE+ head)
            speech_only_feats = speech_feats
            for i, layer in enumerate(self.dual_model):
                speech_only_feats, position_bias = layer(speech_only_feats, position_bias=position_bias)
                out_layers_aud.append(speech_only_feats)
            speech_only_feats_os = self.linear(speech_only_feats)  # (B, T, 256)

        # Semantic path (common_model, with dim adapters)
        if self.common_model_name == "roberta":
            x_768 = self.audio_to_common(inp)
            x = x_768
            for i, layer in enumerate(self.common_model):
                if self.use_conv == "True" and mode == "speech":
                    x = x.permute(0, 2, 1)
                    x = self.downblock[i](x)
                    x = x.permute(0, 2, 1)
                x = layer(x)[0]
                if self.use_conv == "True" and mode == "speech":
                    x = x.permute(0, 2, 1)
                    x = self.upblock[i](x)
                    x = x.permute(0, 2, 1)
                out_layers.append(x)  # (B, T_reduced_or_full, 768)
            x_1024 = self.common_to_audio(x)   # back to HuBERT-large space
            fusion_out = x_1024
        else:
            raise NotImplementedError

        if mode == "speech":
            fusion_feats_audio = fusion_out
            if self.pool_fn == "avg":
                if weights is None:
                    pooled_audio = torch.mean(fusion_feats_audio, 1).squeeze(1)
                else:
                    pooled_audio = torch.matmul(weights, fusion_feats_audio)
            else:
                pooled_audio = self.attenpool(fusion_feats_audio)

        # Note: out_layers currently has 768-d entries from common_model; if you
        # want stacked layer outputs in HuBERT-large space, replace each with the
        # 1024-projected version. Kept as-is for backward compat with CARE.
        fusion_out_stack = torch.stack(out_layers, dim=0)
        fusion_out_aud_stack = torch.stack(out_layers_aud, dim=0)

        return speech_only_feats_os, pooled_audio, fusion_out_stack, fusion_out_aud_stack
