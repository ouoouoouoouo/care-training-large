#!/bin/bash
# apply_hubert_large_patches.sh
# Applies care-training-large patches into the CARE codebase.

set -euo pipefail

CARE_ROOT="${1:-/home/ouo/care_training/CARE}"
PRETRAIN_DIR="$CARE_ROOT/pretraining"

if [ ! -d "$PRETRAIN_DIR" ]; then
    echo "ERROR: CARE pretraining directory not found at $PRETRAIN_DIR"
    echo "Usage: bash apply_hubert_large_patches.sh [CARE_ROOT]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==== Applying care-training-large patches ===="
echo "  CARE root: $CARE_ROOT"
echo "  patches:   $SCRIPT_DIR"
echo ""

# 1. Config: overwrite config.py with our version
if [ -f "$PRETRAIN_DIR/config.py" ] && [ ! -f "$PRETRAIN_DIR/config.py.bak_baseline" ]; then
    cp "$PRETRAIN_DIR/config.py" "$PRETRAIN_DIR/config.py.bak_baseline"
    echo "  ✓ Backed up existing config.py to config.py.bak_baseline"
fi
cp "$SCRIPT_DIR/care_config.py" "$PRETRAIN_DIR/config.py"
echo "  ✓ Installed care_config.py -> config.py"

# 2. Model: add HuBERT-large variant (baseline model_pase.py untouched)
cp "$SCRIPT_DIR/model_pase_hubert_large.py" "$PRETRAIN_DIR/model_pase_hubert_large.py"
echo "  ✓ Installed model_pase_hubert_large.py"

# 3. Train: add HuBERT-large trainer (baseline train_pase.py untouched)
cp "$SCRIPT_DIR/train_pase_hubert_large.py" "$PRETRAIN_DIR/train_pase_hubert_large.py"
echo "  ✓ Installed train_pase_hubert_large.py"

# 4. Smoke test: shape/gradient check before committing to a 200K-step run
cp "$SCRIPT_DIR/smoke_test_hubert_large.py" "$PRETRAIN_DIR/smoke_test_hubert_large.py"
echo "  ✓ Installed smoke_test_hubert_large.py"

echo ""
echo "==== Verification ===="
ls -lh "$PRETRAIN_DIR"/{config.py,model_pase.py,model_pase_hubert_large.py,train_pase.py,train_pase_hubert_large.py} 2>/dev/null || true

echo ""
echo "Baseline files (untouched):"
echo "  $PRETRAIN_DIR/model_pase.py    (WavLM-base + RoBERTa semantic)"
echo "  $PRETRAIN_DIR/train_pase.py    (baseline CARE trainer)"
echo ""
echo "New files (this project):"
echo "  $PRETRAIN_DIR/model_pase_hubert_large.py    (HuBERT-large + Llama semantic)"
echo "  $PRETRAIN_DIR/train_pase_hubert_large.py    (matching trainer)"
echo ""
echo "Next: run"
echo "  cd $PRETRAIN_DIR"
echo "  CUDA_VISIBLE_DEVICES=4 python train_pase_hubert_large.py \\"
echo "      /home/ouo/care_training_large/ckpts_hubert_large \\"
echo "      --batch_size 8 --num_layers 6 \\"
echo "      --use_conv True --use_pretrained True --pool_fn avg"
