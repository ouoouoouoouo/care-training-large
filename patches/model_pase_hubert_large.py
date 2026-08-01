"""HuBERT-large + Llama-aligned CARE model.

Extends CARE's SpeechTextModel (WavLM-base + RoBERTa semantic) to:
  1. HuBERT-large backbone (24 layers, 1024-d) instead of WavLM-base (12 layers,
     768-d), with RoBERTa-large as the matching 1024-d semantic encoder.
  2. Semantic supervision target = Llama-3.1-8B mean-pool (4096-d) instead of
     RoBERTa-base mean-pool (768-d).

Preserves CARE's key architectural ideas:
  - Split into a shared audio path, a dual (acoustic) path, and a common
    (semantic) path with adapters.
  - Dual acoustic loss: PASE+ 256-d reconstruction (unchanged, called
    "opensmile_loss" in CARE code for legacy reasons).
  - Semantic loss: MSE between projected audio pool and Llama mean-pool.

Dimension handling:
  CARE has no dimension adapters at all — WavLM-base and RoBERTa-base are both
  768-d, so the semantic encoder consumes speech representations directly.
  Pairing HuBERT-large (1024) with RoBERTa-base (768) would force two extra
  Linear layers that the paper does not have, so the default text model here is
  roberta-large (1024), which restores the paper's adapter-free boundary.
  `audio_to_common` / `common_to_audio` collapse to nn.Identity whenever the two
  hidden sizes match, and only materialise as real projections if you
  deliberately mix a 1024-d backbone with a 768-d text model.

  - llama_semantic_proj (1024 -> 4096) on pooled_audio for Llama-aligned MSE
    (this one IS a deviation: the paper regresses 768-d RoBERTa mean-pool
    directly, we regress 4096-d Llama mean-pool)

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
  roberta_model: RobertaModel from `roberta-large` (semantic encoder)
  num_layers   : layers per encoder. Paper Sec IV-A splits the backbone in half
                 (common encoder = first N, acoustic encoder = last N), and the
                 semantic encoder takes the LAST N layers of the text model.
                 12 for HuBERT-large + RoBERTa-large:
                   common   = hubert.layers[0:12]
                   acoustic = hubert.layers[12:24]
                   semantic = roberta.layer[12:24]
                 6 reproduces the paper's WavLM-base + RoBERTa-base geometry.
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

    LLAMA_HIDDEN = 4096    # Llama-3.1-8B (semantic supervision target)
    PASE_DIM = 256         # PASE+ acoustic target dim

    def __init__(
        self,
        hubert_model,
        roberta_model,
        num_layers: int = 12,
        common_model: str = "roberta",
        use_conv: str = "True",
        pool_fn: str = "avg",
        semantic_proj_hidden: int = 2048,   # bottleneck between 1024 -> 4096
        text_model_id: str = "roberta-large",
        keep_text_model: bool = False,
    ) -> None:
        super().__init__()
        self.common_model_name = common_model
        self.num_layers = num_layers
        self.use_conv = use_conv

        # Hidden sizes are read from the checkpoints rather than hard-coded, so
        # base/large can be mixed on either side without silent shape errors.
        self.HUBERT_HIDDEN = hubert_model.config.hidden_size      # 1024 for large
        self.TEXT_HIDDEN = roberta_model.config.hidden_size       # 1024 for large
        self.ROBERTA_HIDDEN = self.TEXT_HIDDEN                    # backwards-compat alias

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

        # ---- Semantic encoder (paper) == `common_model` (CARE code naming) ---
        # Paper Sec IV-A: "initialized with the weights of the last 6 layers of
        # a pre-trained RoBERTa base model" -> generally, the LAST num_layers
        # layers of whichever text model is used.
        if common_model != "roberta":
            raise NotImplementedError(f"common_model={common_model!r} not supported")

        text_layers = roberta_model.encoder.layer
        if num_layers > len(text_layers):
            raise ValueError(
                f"num_layers={num_layers} exceeds the text model's "
                f"{len(text_layers)} layers ({text_model_id}); the semantic "
                f"encoder takes the last num_layers layers."
            )
        start = len(text_layers) - num_layers
        self.common_model = nn.ModuleList(
            [text_layers[i] for i in range(start, len(text_layers))]
        )

        self.roberta_embeddings = roberta_model.embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        # CARE keeps a text-side tower for its speech_text mode; this trainer
        # never runs it, and at roberta-large it is ~150M frozen parameters
        # pickled into every torch.save(emo_model) checkpoint.
        self.text_model = nn.ModuleList(
            [text_layers[i] for i in range(num_layers)]
        ) if keep_text_model else None

        # ---- Dimension adapters (absent from CARE, hence Identity by default)
        # CARE has none: WavLM-base and RoBERTa-base are both 768. With
        # HuBERT-large + RoBERTa-large both sides are 1024, so the boundary
        # stays adapter-free. Real projections appear only for a mixed pairing.
        if self.HUBERT_HIDDEN == self.TEXT_HIDDEN:
            self.audio_to_common = nn.Identity()
            self.common_to_audio = nn.Identity()
        else:
            self.audio_to_common = nn.Linear(self.HUBERT_HIDDEN, self.TEXT_HIDDEN)
            self.common_to_audio = nn.Linear(self.TEXT_HIDDEN, self.HUBERT_HIDDEN)
        self.has_dim_adapters = self.HUBERT_HIDDEN != self.TEXT_HIDDEN

        # ---- Conv adapters (paper Sec III): one 1D-conv block before and one
        # after EACH semantic-encoder transformer layer. Kernel 5, /3 downsample
        # before and x3 upsample after, channels matched to the text model.
        if use_conv == "True":
            self.downblock = nn.ModuleList([
                nn.Conv1d(in_channels=self.TEXT_HIDDEN,
                          out_channels=self.TEXT_HIDDEN,
                          kernel_size=5, stride=3)
                for _ in range(num_layers)
            ])
            self.upblock = nn.ModuleList([
                nn.ConvTranspose1d(in_channels=self.TEXT_HIDDEN,
                                   out_channels=self.TEXT_HIDDEN,
                                   kernel_size=5, stride=3, output_padding=1)
                for _ in range(num_layers)
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
        """Semantic encoder: conv-adapted pass through the frozen text layers.

        Dimensions are HUBERT_HIDDEN -> TEXT_HIDDEN -> HUBERT_HIDDEN; both
        adapters are Identity when the two match (the paper's configuration).

        `out_layers`, if given, collects the per-layer output already projected
        back to HUBERT_HIDDEN and length-matched to `inp`, so the caller can
        stack it.
        """
        x = self.audio_to_common(inp)
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
        Shapes, with L = num_layers and H = HUBERT_HIDDEN:
            fusion_out     : (1 + 2L, B, T, H)  conv out + common + semantic
            pooled_audio   : (B, H)             mean-pooled semantic output
            fusion_out_aud : (1 + 2L, B, T, H)  conv out + common + acoustic

        For the paper's L=6, H=768 this is the familiar (13, B, T, 768). For
        HuBERT-large + RoBERTa-large at L=12, H=1024 it is (25, B, T, 1024) —
        the downstream SUPERB-style convex combination must be sized to match.
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


# ---- Checkpoint loading ----------------------------------------------------
def load_pretrained(checkpoint_path, device="cpu", eval_mode=True):
    """Rebuild a trained SpeechTextModelHuBERTLarge from a trainer checkpoint.

    The trainer saves state_dicts rather than pickled modules, so downstream
    feature extraction goes through here instead of `torch.load(path)` directly:

        from model_pase_hubert_large import load_pretrained
        model = load_pretrained("ckpts_hubert_large/best.pth", device="cuda")
        fusion_out, pooled_audio, fusion_out_aud = model.extract_audio_features(wavs)

    Works on best.pth, final.pth, model-<update>.pth and last.pth alike — the
    architecture config travels with the weights, so nothing has to be kept in
    sync by hand on the downstream side.
    """
    from transformers import HubertModel, RobertaModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} is not a trainer checkpoint (expected 'model' and "
            f"'config' keys). Checkpoints written before the state_dict switch "
            f"are pickled modules — load those with torch.load() directly."
        )
    cfg = ckpt["config"]

    hubert = HubertModel.from_pretrained(cfg["hubert_model_id"])
    roberta = RobertaModel.from_pretrained(cfg["text_model_id"])
    model = SpeechTextModelHuBERTLarge(
        hubert, roberta,
        num_layers=cfg["num_layers"],
        common_model=cfg["common_model"],
        use_conv=cfg["use_conv"],
        pool_fn=cfg["pool_fn"],
        text_model_id=cfg["text_model_id"],
        keep_text_model=cfg.get("keep_text_model", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if eval_mode:
        model.eval()
    return model
