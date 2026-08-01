"""Local paths for the cluster — complete config for CARE dataset_pase.py.

Overwrites /home/ouo/care_training/CARE/pretraining/config.py.

Includes ALL variables that dataset_pase.py imports (even those the CARE
codebase leaves unused / dead after patching). Only the semantic target
(podcast_roberta_feats*) is redirected to our new Llama mean-pool features;
everything else matches the care-training baseline paths.
"""

# ---- MSP-PODCAST audio + splits (from care-training baseline) ----
podcast_audio_folder = "/home/ouo/dataset/MSP_Podcast/Audios"
train_files          = "/home/ouo/care_training/data/trainlist.pkl"
valid_files          = "/home/ouo/care_training/data/vallist.pkl"

# ---- Whisper transcripts ----
podcast_transcripts  = "/home/ouo/care_training/data/whisper_transcripts.json"

# ---- Acoustic supervision target: PASE+ 256-d @ 100Hz (unchanged) ----
podcast_pase_feats      = "/home/ouo/care_training/data/pase_features"
podcast_opensmile_feats = "/home/ouo/care_training/data/pase_features"   # alias, dead in our runs

# ---- Semantic supervision target: Llama-3.1-8B mean-pool (NEW, 4096-d) ----
# Produced by scripts/extract_msppodcast_llama_mean.py in this repo.
podcast_llama_feats     = "/home/ouo/care_training_large/data/llama_features"

# All roberta_* aliases point to the new Llama features so CARE's dataset
# (which reads only these keys) drops in the Llama target unchanged.
podcast_roberta_feats                = podcast_llama_feats
podcast_roberta_feats_whisper        = podcast_llama_feats
podcast_roberta_feats_whisper_sup    = podcast_llama_feats
podcast_roberta_feats_large          = podcast_llama_feats
podcast_roberta_feats_paraphrasings  = podcast_llama_feats
podcast_roberta_feats_all            = podcast_llama_feats
podcast_roberta_feats_supervised     = podcast_llama_feats
podcast_roberta_logits               = "/home/ouo/care_training_large/data/roberta_logits_PLACEHOLDER"

# ---- WavLM tokens / features ----
# dataset_pase.py OPENS these files at __init__ (not just imports the paths),
# so they must actually exist on disk. Point wavlm_tokens_6 to the same real
# wavlm_tokens.txt — the content is not used in our loss (only opensmile_loss
# and distil_loss matter) but the file must be readable.
podcast_wavlm_tokens    = "/home/ouo/care_training/data/wavlm_tokens.txt"
podcast_wavlm_tokens_6  = "/home/ouo/care_training/data/wavlm_tokens.txt"   # reuse same
podcast_wavlm_feats     = "/home/ouo/care_training/data/pase_features"     # reuse pase dir
podcast_wavlm_feats_6   = "/home/ouo/care_training/data/pase_features"     # reuse pase dir

# ---- Quantized energy / pitch (dead code, --energy_weights/--pitch_weights False) ----
# Point to any existing directory so `os.listdir()` (if called) doesn't crash.
quantized_energy_folder = "/home/ouo/care_training/data/pase_features"
quantized_pitch_folder  = "/home/ouo/care_training/data/pase_features"

# ---- Labels (dead in unsupervised mode; needed only for --supervised True) ----
podcast_labels          = "/home/ouo/care_training/data/text_labels.json"
podcast_text_labels     = "/home/ouo/care_training/data/text_labels.json"
