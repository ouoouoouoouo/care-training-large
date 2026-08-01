"""Local paths for the cluster (analogous to care-training's care_config.py).

Copy this file into:
    /home/ouo/care_training/CARE/pretraining/config.py

Or apply via patches/apply_hubert_large_patches.sh.
"""

# ---- MSP-PODCAST audio (unchanged from care-training baseline) ----
podcast_audio_dir = "/home/ouo/dataset/MSP_Podcast/Audios"

# ---- Acoustic supervision target: PASE+ 256-d @ 100Hz (unchanged) ----
podcast_pase_feats      = "/home/ouo/care_training/data/pase_features"
podcast_opensmile_feats = "/home/ouo/care_training/data/pase_features"  # alias, unused

# ---- Semantic supervision target: Llama-3.1-8B mean-pool (NEW, 4096-d) ----
# Produced by scripts/extract_msppodcast_llama_mean.py in this repo.
# For care-training-large only — the baseline care-training uses RoBERTa
# features at a different path.
podcast_llama_feats     = "/home/ouo/care_training_large/data/llama_features"

# ---- Aliases so the modified train script can read either target ----
podcast_roberta_feats                = podcast_llama_feats  # semantic target
podcast_roberta_feats_whisper        = podcast_llama_feats
podcast_roberta_feats_whisper_sup    = podcast_llama_feats
podcast_roberta_logits               = "/home/ouo/care_training_large/data/roberta_logits_PLACEHOLDER"
podcast_roberta_feats_large          = podcast_llama_feats
podcast_roberta_feats_paraphrasings  = podcast_llama_feats
podcast_roberta_feats_all            = podcast_llama_feats
podcast_roberta_feats_supervised     = podcast_llama_feats

# ---- Whisper transcripts (unchanged; not directly used at train time,
#      only during Llama mean-pool extraction) ----
podcast_transcripts     = "/home/ouo/care_training/data/whisper_transcripts"
