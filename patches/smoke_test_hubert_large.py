"""Shape smoke test for SpeechTextModelHuBERTLarge.

Runs one forward + one backward with a random 5 s batch, so every WavLM->HuBERT
API difference (feature_projection return type, encoder layer signature,
stable-layer-norm placement, 1024<->768 adapters) is exercised in ~1 minute
instead of being discovered at step 0 of a 200K-step run.

Usage:
    cd /home/ouo/care_training/CARE/pretraining
    CUDA_VISIBLE_DEVICES=4 python smoke_test_hubert_large.py
    # CPU-only (slower, no GPU needed):
    CUDA_VISIBLE_DEVICES= python smoke_test_hubert_large.py --device cpu
"""
import argparse

import torch
import torch.nn as nn
from transformers import HubertModel, RobertaModel

from model_pase_hubert_large import SpeechTextModelHuBERTLarge

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--batch_size", type=int, default=2)
p.add_argument("--seconds", type=float, default=5.0)
p.add_argument("--num_layers", type=int, default=12,
               help="Half the backbone depth, per paper Sec IV-A (12 for HuBERT-large).")
p.add_argument("--use_conv", default="True")
p.add_argument("--hubert_model_id", default="facebook/hubert-large-ll60k")
p.add_argument("--text_model_id", default="roberta-large")
p.add_argument("--return_layers", action="store_true",
               help="Also build/stack the per-layer outputs (extra memory).")
args = p.parse_args()

device = torch.device(args.device)
B = args.batch_size
T_WAV = int(16000 * args.seconds)
T_PASE = 249          # trainer does opensmile_feats[:, :249, :]
PASE_DIM = 256
LLAMA_DIM = 4096

print(f"device={device}  B={B}  wav={T_WAV} samples ({args.seconds}s)")

hubert = HubertModel.from_pretrained(args.hubert_model_id)
print(f"  hubert layers={len(hubert.encoder.layers)}  hidden={hubert.config.hidden_size}  "
      f"do_stable_layer_norm={hubert.config.do_stable_layer_norm}  "
      f"feat_extract_norm={hubert.config.feat_extract_norm}")
roberta = RobertaModel.from_pretrained(args.text_model_id)
print(f"  text  layers={roberta.config.num_hidden_layers}  hidden={roberta.config.hidden_size}")

model = SpeechTextModelHuBERTLarge(
    hubert, roberta,
    num_layers=args.num_layers, common_model="roberta",
    use_conv=args.use_conv, pool_fn="avg",
    text_model_id=args.text_model_id,
).to(device)
print(f"  dim adapters: {'YES (not paper-faithful)' if model.has_dim_adapters else 'no (paper-faithful)'}")

wavs = torch.randn(B, T_WAV, device=device)
pase_target = torch.randn(B, T_PASE, PASE_DIM, device=device)
llama_target = torch.randn(B, LLAMA_DIM, device=device)

speech_os, pooled_audio, fo, foa = model(wavs, return_layers=args.return_layers)
pooled_llama = model.llama_semantic_proj(pooled_audio)

H = model.HUBERT_HIDDEN
print("\n--- forward shapes ---")
print(f"  speech_only_feats_os : {tuple(speech_os.shape)}   expect ({B}, {T_PASE}, {PASE_DIM})")
print(f"  pooled_audio         : {tuple(pooled_audio.shape)}   expect ({B}, {H})")
print(f"  pooled_audio_llama   : {tuple(pooled_llama.shape)}   expect ({B}, {LLAMA_DIM})")
print(f"  fusion_out           : {tuple(fo.shape) if fo is not None else None}")
print(f"  fusion_out_aud       : {tuple(foa.shape) if foa is not None else None}")
print(f"  downstream layer stack: {1 + 2*args.num_layers} layers x T x {2*H} "
      f"(concat semantic+acoustic); paper's WavLM-base config is 13 x T x 1536")

assert pooled_audio.shape == (B, model.HUBERT_HIDDEN), pooled_audio.shape
assert pooled_llama.shape == (B, LLAMA_DIM), pooled_llama.shape
assert speech_os.shape[0] == B and speech_os.shape[2] == PASE_DIM, speech_os.shape

t_out = speech_os.shape[1]
if t_out != T_PASE:
    raise SystemExit(
        f"\nFRAME MISMATCH: HuBERT produced T={t_out} but the trainer slices the "
        f"PASE target to {T_PASE}. MSE would broadcast-error at step 0.\n"
        f"Fix by matching the dataset's wav length (HuBERT stride is 320: "
        f"{T_PASE} frames <- {(T_PASE + 1) * 320} samples) or by changing the "
        f"[:, :249, :] slice in train_pase_hubert_large.py."
    )

# --- backward, mirroring the trainer's two-loss / zero_grad_hook pattern ---
distil = nn.MSELoss()(pooled_llama, llama_target)
opensmile = nn.MSELoss()(speech_os, pase_target)
distil.backward(retain_graph=True)
opensmile.backward()

def _has_grad(module):
    ps = list(module.parameters())
    return bool(ps) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps)


grad_ok = {
    "audio_model (common enc.)":   _has_grad(model.audio_model),
    "dual_model (acoustic enc.)":  _has_grad(model.dual_model),
    "common_model (semantic enc.)": _has_grad(model.common_model),
    "llama_semantic_proj":         _has_grad(model.llama_semantic_proj),
    "linear (PASE head)":          _has_grad(model.linear),
}
if args.use_conv == "True":
    grad_ok["downblock"] = _has_grad(model.downblock)
    grad_ok["upblock"] = _has_grad(model.upblock)
if model.has_dim_adapters:
    grad_ok["audio_to_common"] = _has_grad(model.audio_to_common)
    grad_ok["common_to_audio"] = _has_grad(model.common_to_audio)
print("\n--- backward: gradients reach ---")
print("  (graph connectivity only — the trainer additionally freezes")
print("   common_model per paper Fig. 2)")
for k, v in grad_ok.items():
    print(f"  {'OK ' if v else 'NO '} {k}")

if device.type == "cuda":
    print(f"\npeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB at B={B}")

missing = [k for k, v in grad_ok.items() if not v]
if missing:
    raise SystemExit(f"\nFAILED: no gradient into {missing}")
print("\nSMOKE TEST PASSED")
