"""Extract Llama-3.1-8B mean-pool features from MSP-PODCAST Whisper transcripts.

For each of the 149K MSP-PODCAST utterances:
  1. Read Whisper transcript (already prepared in care-training).
  2. Tokenize and run Llama-3.1-8B (frozen base weights, HuggingFace hub).
  3. Mean-pool `last_hidden_state` over non-pad tokens → (4096,) per utt.
  4. Save as .npy so CARE's dataset_pase.py can read it (drop-in replacement
     for the RoBERTa mean-pool it originally reads).

Runtime estimate: ~4 hours on Blackwell 6000 (batch 16, fp16) for 149K utts.

Usage:
    huggingface-cli login
    python -m scripts.extract_msppodcast_llama_mean \
        --transcripts /home/ouo/care_training/data/whisper_transcripts \
        --out-dir     /home/ouo/care_training_large/data/llama_features \
        --model       meta-llama/Meta-Llama-3.1-8B \
        --batch-size  16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@torch.no_grad()
def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """hidden: (B, T, H), attention_mask: (B, T). Returns (B, H) fp32."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return (summed / denom).float()


def _load_transcripts(transcripts_path: Path) -> Dict[str, str]:
    """Load {utt_id: transcript} from either:
      - a single CSV file (columns: utt_id, text)  — matches merits-l-text layout
      - a directory of JSON files (dict {utt_id: text or {"text": ...}})
      - a single JSON file with {utt_id: text or {"text": ...}}
    """
    out: Dict[str, str] = {}

    # CSV single file (merits-l-text layout)
    if transcripts_path.is_file() and transcripts_path.suffix.lower() == ".csv":
        import csv
        with transcripts_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("utt_id")
                text = row.get("text") or row.get("transcript")
                if uid and text is not None:
                    out[uid] = str(text)
        print(f"Loaded {len(out)} transcripts from CSV: {transcripts_path.name}")
        return out

    # JSON single file
    if transcripts_path.is_file() and transcripts_path.suffix.lower() == ".json":
        data = json.loads(transcripts_path.read_text(encoding="utf-8"))
        for uid, val in data.items():
            if isinstance(val, str):
                out[uid] = val
            elif isinstance(val, dict) and "text" in val:
                out[uid] = str(val["text"])
        print(f"Loaded {len(out)} transcripts from JSON: {transcripts_path.name}")
        return out

    # Directory: try JSONs, then CSVs
    if transcripts_path.is_dir():
        json_files = sorted(transcripts_path.glob("*.json"))
        csv_files = sorted(transcripts_path.glob("*.csv"))
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                for uid, val in data.items():
                    if isinstance(val, str):
                        out[uid] = val
                    elif isinstance(val, dict) and "text" in val:
                        out[uid] = str(val["text"])
        for cf in csv_files:
            import csv
            with cf.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    uid = row.get("utt_id")
                    text = row.get("text") or row.get("transcript")
                    if uid and text is not None and uid not in out:
                        out[uid] = str(text)
        if not out:
            raise FileNotFoundError(
                f"No JSON/CSV transcripts found under {transcripts_path}"
            )
        print(f"Loaded {len(out)} transcripts from {transcripts_path}")
        return out

    raise FileNotFoundError(
        f"{transcripts_path} does not exist or is not a supported file/dir."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts", required=True, type=str,
                   help="Directory holding Whisper transcript JSONs.")
    p.add_argument("--out-dir", required=True, type=str,
                   help="Directory to write per-utt .npy files (4096-d).")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B", type=str)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--max-length", default=256, type=int)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = _load_transcripts(Path(args.transcripts))
    utt_ids = sorted(transcripts.keys())

    # Skip already-extracted (resumable)
    done = set(p.stem for p in out_dir.glob("*.npy"))
    remaining = [uid for uid in utt_ids if uid not in done]
    print(f"Total: {len(utt_ids)}   already-done: {len(done)}   to-do: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    print(f"Loading tokenizer + model ({args.model}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"  hidden_size = {model.config.hidden_size}")

    B = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(remaining), B), desc="extract"):
            batch_uids: List[str] = remaining[i : i + B]
            batch_texts = [transcripts[uid] for uid in batch_uids]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            try:
                out = model(**enc)
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
                arr = pooled.cpu().numpy().astype(np.float32)
            except Exception as e:  # noqa: BLE001
                print(f"\n[warn] batch @ {batch_uids[0]}: {e}")
                continue
            for uid, vec in zip(batch_uids, arr):
                np.save(out_dir / f"{uid}.npy", vec)

    total_done = len(list(out_dir.glob("*.npy")))
    sample = np.load(next(out_dir.glob("*.npy")))
    print(f"\nDone. {total_done} .npy files in {out_dir}   sample shape: {sample.shape}")


if __name__ == "__main__":
    main()
