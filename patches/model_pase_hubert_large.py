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
  - audio_to_common (1024 -> 768) before common_model
  - common_to_audio (768 -> 1024) after common_model
  - llama_semantic_proj (1024 -> 4096) on pooled_audio for Llama-aligned MSE

WavLM -> HuBERT API differences handled here (these are NOT interchangeable):
  - WavLM's feature_projection returns (hidden_states, extract_features);
    HuBERT's returns hidden_states only.
  - WavLM encoder layers take/return `position_bias` (gated relative position
    bias); HuBERT encoder layers have neither.
  - facebook/hubert-large-ll60k sets do_stable_layer_norm=True, so the encoder
    is pre-LN and `encoder.layer_norm` belongs AFTER the layer stack, not
    before it (WavLM-base is post-LN and applies it at the input).

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


def _first(out):
    """Encoder layers return either a tensor or a tuple depending on the
    transformers version / model family. Normalise to a tensor."""
    return out[0] if isinstance(out, (tuple, list)) else out


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

        # hubert-large-ll60k -> True (pre-LN encoder, final LN after the stack)
        # hubert-base-ls960  -> False (post-LN encoder, LN at the input)
        self.do_stable_layer_norm = bool(
            getattr(hubert_model.config, "do_stable_layer_norm", False)
        )

        total_layers = len(hubert_model.encoder.layers)             # 24 for HuBERT-large
        self.audio_model = nn.ModuleList(
            [hubert_model.encoder.layers[i] for i in range(num_layers)]
        )
        # dual_model: next `num_layers` layers, still on the audio path
        dual_end = min(num_layers * 2, total_layers)
        self.dual_model = nn.ModuleList(
            [hubert_model.encoder.layers[i] for i in range(num_layers, dual_end)]
        )

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
        # CARE original had no mismatch because WavLM==RoBERTa==768.
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
    #  Shared HuBERT front-end (conv extractor -> feature projection -> pos conv)
    # ------------------------------------------------------------------------
    def _hubert_frontend(self, audio):
        """(B, T_wav) or (B, 1, T_wav) -> (B, T, 1024), channels-last throughout."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        x = self.audio_feature_extractor(audio)      # (B, C_conv, T)
        x = x.transpose(1, 2)                        # (B, T, C_conv)
        # HuBERT's feature_projection returns a TENSOR (WavLM's returns a tuple).
        x = self.feature_projection_audio(x)         # (B, T, 1024)
        x = _first(x)
        x = x + self.pos_conv(x)                     # pos_conv is channels-last in/out
        if not self.do_stable_layer_norm:
            # post-LN encoder (hubert-base): LN before the layer stack
            x = self.wavlm_layer_norm(x)
        x = self.wavlm_dropout(x)
        return x

    def _run_common(self, inp, out_layers=None, apply_conv=True, project_back=True):
        """Semantic path: 1024 -> 768 -> common_model (+conv blocks) -> 1024.

        `out_layers`, if given, collects the per-layer output already projected
        back to 1024-d and length-matched to `inp`, so the caller can stack it.
        """
        x = self.audio_to_common(inp)                # 1024 -> 768
        for i, layer in enumerate(self.common_model):
            if self.use_conv == "True" and apply_conv:
                x = x.permute(0, 2, 1)
                x = self.downblock[i](x)
                x = x.permute(0, 2, 1)
            x = _first(layer(x))
            if self.use_conv == "True" and apply_conv:
                x = x.permute(0, 2, 1)
                x = self.upblock[i](x)
                x = x.permute(0, 2, 1)
            # Length-match back to the audio path (conv round-trip is exact for
            # T=249 but not for arbitrary T).
            if x.shape[1] < inp.shape[1]:
                pad = torch.zeros(x.shape[0], inp.shape[1] - x.shape[1], x.shape[-1],
                                  dtype=x.dtype, device=x.device)
                x = torch.cat((x, pad), dim=1)
            elif x.shape[1] > inp.shape[1]:
                x = x[:, :inp.shape[1], :]
            if out_layers is not None:
                out_layers.append(self.common_to_audio(x))
        return self.common_to_audio(x) if project_back else x

    def _pool(self, feats):
        if self.pool_fn == "avg":
            return torch.mean(feats, dim=1)
        return self.attenpool(feats)

    # ------------------------------------------------------------------------
    #  extract_audio_features — used by downstream feature extraction
    # ------------------------------------------------------------------------
    def extract_audio_features(self, audio):
        """Return (fusion_out, pooled_audio, fusion_out_aud).

        Same interface as CARE's SpeechTextModel.extract_audio_features, so
        merits-l-llama's downstream extractor can consume this drop-in.
        Shapes:
            fusion_out     : (13, B, T, 1024)   semantic path all layers
            pooled_audio   : (B, 1024)          mean-pooled semantic output
            fusion_out_aud : (13, B, T, 1024)   acoustic path all layers
        """
        x = self._hubert_frontend(audio)
        out_layers, out_layers_aud = [x], [x]

        speech_feats = x
        for layer in self.audio_model:
            speech_feats = _first(layer(speech_feats))
            out_layers.append(speech_feats)
            out_layers_aud.append(speech_feats)
        inp = speech_feats

        # Acoustic path
        speech_only_feats = speech_feats
        for layer in self.dual_model:
            speech_only_feats = _first(layer(speech_only_feats))
            out_layers_aud.append(speech_only_feats)
        if self.do_stable_layer_norm:
            # pre-LN encoder: final LN belongs at the end of the layer stack
            speech_only_feats = self.wavlm_layer_norm(speech_only_feats)
            out_layers_aud[-1] = speech_only_feats
        fusion_out_aud = torch.stack(out_layers_aud, dim=0)

        # Semantic path — with dim adaptation
        fusion_feats_audio = self._run_common(inp, out_layers=out_layers)
        fusion_out = torch.stack(out_layers, dim=0)

        pooled_audio = self._pool(fusion_feats_audio)
        return fusion_out, pooled_audio, fusion_out_aud

    # ------------------------------------------------------------------------
    #  forward — training path, mirrors CARE's forward interface
    # ------------------------------------------------------------------------
    def forward(self, audio, padding_mask=None, mask=False, mode="speech",
                weights=None, return_layers=False):
        """Same output signature as CARE:
            speech_only_feats_os (B, T, 256)   PASE+ prediction
            pooled_audio         (B, 1024)     for llama_semantic_proj -> MSE
            fusion_out           (13, B, T, 1024) or None
            fusion_out_aud       (13, B, T, 1024) or None

        The stacked layer outputs are only built when `return_layers=True`;
        the trainer discards them, and materialising two (13, B, T, 1024)
        tensors inside the autograd graph costs ~0.8 GB at B=32, T=249.

        The trainer computes:
            opensmile_loss = MSE(speech_only_feats_os, pase_target)
            distil_loss    = MSE(llama_semantic_proj(pooled_audio), llama_target)
        """
        if mode not in ("speech", "speech_text"):
            raise NotImplementedError(f"mode={mode!r} not supported")
        if self.common_model_name != "roberta":
            raise NotImplementedError

        x = self._hubert_frontend(audio)
        out_layers = [x] if return_layers else None
        out_layers_aud = [x] if return_layers else None

        speech_feats = x
        for layer in self.audio_model:
            speech_feats = _first(layer(speech_feats))
            if return_layers:
                out_layers.append(speech_feats)
                out_layers_aud.append(speech_feats)
        inp = speech_feats

        # Acoustic path (dual_model + PASE+ head)
        speech_only_feats = speech_feats
        for layer in self.dual_model:
            speech_only_feats = _first(layer(speech_only_feats))
            if return_layers:
                out_layers_aud.append(speech_only_feats)
        if self.do_stable_layer_norm:
            speech_only_feats = self.wavlm_layer_norm(speech_only_feats)
            if return_layers:
                out_layers_aud[-1] = speech_only_feats
        speech_only_feats_os = self.linear(speech_only_feats)  # (B, T, 256)

        # Semantic path (common_model, with dim adapters)
        fusion_out = self._run_common(
            inp, out_layers=out_layers, apply_conv=(mode == "speech")
        )

        if weights is None:
            pooled_audio = self._pool(fusion_out)
        else:
            pooled_audio = torch.matmul(weights, fusion_out)

        fusion_out_stack = torch.stack(out_layers, dim=0) if return_layers else None
        fusion_out_aud_stack = torch.stack(out_layers_aud, dim=0) if return_layers else None

        return speech_only_feats_os, pooled_audio, fusion_out_stack, fusion_out_aud_stack
