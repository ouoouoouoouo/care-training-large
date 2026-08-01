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
roberta = RobertaModel.from_pretrained("roberta-base")

model = SpeechTextModelHuBERTLarge(
    hubert, roberta,
    num_layers=args.num_layers, common_model="roberta",
    use_conv=args.use_conv, pool_fn="avg",
).to(device)

wavs = torch.randn(B, T_WAV, device=device)
pase_target = torch.randn(B, T_PASE, PASE_DIM, device=device)
llama_target = torch.randn(B, LLAMA_DIM, device=device)

speech_os, pooled_audio, fo, foa = model(wavs, return_layers=args.return_layers)
pooled_llama = model.llama_semantic_proj(pooled_audio)

print("\n--- forward shapes ---")
print(f"  speech_only_feats_os : {tuple(speech_os.shape)}   expect (B, T, 256)")
print(f"  pooled_audio         : {tuple(pooled_audio.shape)}   expect (B, 1024)")
print(f"  pooled_audio_llama   : {tuple(pooled_llama.shape)}   expect (B, 4096)")
print(f"  fusion_out           : {tuple(fo.shape) if fo is not None else None}")
print(f"  fusion_out_aud       : {tuple(foa.shape) if foa is not None else None}")

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

grad_ok = {
    "audio_model":         any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.audio_model.parameters()),
    "dual_model":          any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.dual_model.parameters()),
    "common_model":        any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.common_model.parameters()),
    "audio_to_common":     model.audio_to_common.weight.grad is not None,
    "common_to_audio":     model.common_to_audio.weight.grad is not None,
    "llama_semantic_proj": model.llama_semantic_proj[0].weight.grad is not None,
    "linear (PASE head)":  model.linear.weight.grad is not None,
}
print("\n--- backward: gradients reach ---")
for k, v in grad_ok.items():
    print(f"  {'OK ' if v else 'NO '} {k}")

if device.type == "cuda":
    print(f"\npeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB at B={B}")

missing = [k for k, v in grad_ok.items() if not v]
if missing:
    raise SystemExit(f"\nFAILED: no gradient into {missing}")
print("\nSMOKE TEST PASSED")
