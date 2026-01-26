#!/usr/bin/env bash
set -euo pipefail

CKPT="../../trained_models/hinge_mg_logit_ce_task_heads.ckpt"
CFG="../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
SAVE_DIR="./saved_logits"
RESULTS="hinge_mg_logit_ce_task_heads.txt"

python extract_logits.py \
  --input "../../Validation_Dataset/RoadAnomaly21/images/*.png" \
  --config "$CFG" \
  --save_dir "$SAVE_DIR" \
  --ckpt_path "$CKPT"

python extract_logits.py \
  --input "../../Validation_Dataset/RoadObsticle21/images/*.webp" \
  --config "$CFG" \
  --save_dir "$SAVE_DIR" \
  --ckpt_path "$CKPT"

python extract_logits.py \
  --input "../../Validation_Dataset/FS_LostFound_full/images/*.png" \
  --config "$CFG" \
  --save_dir "$SAVE_DIR" \
  --ckpt_path "$CKPT"

python extract_logits.py \
  --input "../../Validation_Dataset/fs_static/images/*.jpg" \
  --config "$CFG" \
  --save_dir "$SAVE_DIR" \
  --ckpt_path "$CKPT"

python extract_logits.py \
  --input "../../Validation_Dataset/RoadAnomaly/images/*.jpg" \
  --config "$CFG" \
  --save_dir "$SAVE_DIR" \
  --ckpt_path "$CKPT"

python evaluate_logits.py --logits_dir "$SAVE_DIR/RoadAnomaly21"   --results_file "$RESULTS"
python evaluate_logits.py --logits_dir "$SAVE_DIR/RoadObsticle21"  --results_file "$RESULTS"
python evaluate_logits.py --logits_dir "$SAVE_DIR/FS_LostFound_full" --results_file "$RESULTS"
python evaluate_logits.py --logits_dir "$SAVE_DIR/fs_static"       --results_file "$RESULTS"
python evaluate_logits.py --logits_dir "$SAVE_DIR/RoadAnomaly"     --results_file "$RESULTS"
