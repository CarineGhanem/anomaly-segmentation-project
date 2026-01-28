# EoMT Evaluation Usage Guide

This guide explains how to use the evaluation scripts in `eomt/eval` for anomaly segmentation.

## Scripts

The evaluation pipeline consists of four main scripts:

1. **`extract_logits.py`**: Extracts per-pixel logit vectors from images through a trained EoMT model
2. **`evaluate_logits.py`**: Evaluates saved logits using various anomaly scoring methods
3. **`eval_iou_eomt.py`**: Evaluates semantic segmentation performance (mIoU) on Cityscapes dataset
4. **`find_optimal_temperature.py`**: Finds optimal temperature scaling for MSP scoring 

## How to Load Checkpoints

The script supports two checkpoint loading methods:

1. **HuggingFace Hub** (default): If `--ckpt_path` is not provided, the script attempts to download the checkpoint from HuggingFace using the model name specified in the config file.

2. **Local checkpoint**: Provide `--ckpt_path` with the path to your local checkpoint file:
   ```bash
   python eval/extract_logits.py \
     --input "datasets/RoadAnomaly21/images/*.png" \
     --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
     --ckpt_path ./checkpoints/eomt_cityscapes.bin \
     --save_dir ./saved_logits
   ```

The script supports both PyTorch Lightning checkpoints (containing `state_dict` key) and raw PyTorch state dictionaries.

## Evaluation Datasets

Evaluation datasets can be downloaded from:

- [SegmentMeIfYouCan](https://segmentmeifyoucan.com/) - RoadAnomaly21 and RoadObstacle21
- [Fishyscapes](https://fishyscapes.com/) - Lost & Found and Static benchmarks
- [Road Anomaly Dataset](https://github.com/foolwood/RoadAnomaly) - Original Road Anomaly dataset

**Note**: For convenience, checkpoints, datasets, and results are also available in our [shared drive folder](https://drive.google.com/drive/folders/YOUR_DRIVE_FOLDER_ID).

Supported datasets:
- **RoadAnomaly21**: Real street scenes with diverse anomalies
- **RoadObstacle21**: Road-obstacle scenes where anomalies lie on the roadway
- **FS_LostFound_full** (Fishyscapes Lost & Found): Real-world road hazards
- **fs_static** (Fishyscapes Static): Synthetic anomalies pasted into Cityscapes scenes
- **RoadAnomaly**: Collection of real unusual road hazards

## How to Store Logits

Run `extract_logits.py` to process images and save per-pixel logit vectors as `.npz` files:

```bash
python eval/extract_logits.py \
  --input "datasets/RoadAnomaly21/images/*.png" \
  --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --ckpt_path ./checkpoints/eomt_cityscapes.bin \
  --save_dir ./saved_logits
```

## How to Evaluate Logits

Run `evaluate_logits.py` to compute anomaly detection metrics from saved logit files:

```bash
python eval/evaluate_logits.py --logits_dir ./saved_logits/RoadAnomaly21
```

The script evaluates multiple post-hoc anomaly scores:
- **MSP** (Maximum Softmax Probability): `1 - max(softmax(logits))`
- **Entropy**: Measures uncertainty in the predictive distribution
- **Max Logit**: Uses raw logit magnitude as confidence proxy
- **RbA** (Rejected-by-All): Region-level score for mask-based models

Metrics reported:
- **AUPRC** (Area Under Precision-Recall Curve): Higher is better
- **FPR@95** (False Positive Rate at 95% True Positive Rate): Lower is better

## How to Evaluate mIoU

Run `eval_iou_eomt.py` to evaluate semantic segmentation performance (mean Intersection-over-Union) on Cityscapes dataset:

```bash
python eval/eval_iou_eomt.py \
  --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --ckpt_path ./checkpoints/eomt_cityscapes.bin \
  --datadir /path/to/cityscapes \
  --subset val \
  --results-file results.txt
```

The script reports:
- **Per-class IoU**: Intersection-over-Union for each of the 19 Cityscapes classes
- **Mean IoU (mIoU)**: Average IoU across all classes


## Temperature

The `find_optimal_temperature.py` script searches for the optimal temperature scaling parameter for MSP scoring. Temperature scaling improves calibration by adjusting the sharpness of the softmax distribution.

### Usage

```bash
python eval/find_optimal_temperature.py \
  --logits_dir ./saved_logits/RoadAnomaly21 \
  --temp_range "0.5,0.75,1.0,1.1,1.5,2.0,2.5,3.0,5.0,10.0"
```

### Temperature Scaling

Temperature scaling applies: `logits_scaled = logits / temperature`

- **T < 1**: Sharper distribution (more confident predictions)
- **T = 1**: No scaling (default)
- **T > 1**: Smoother distribution (less confident predictions)

The script evaluates each temperature value and reports the one with the best AUPRC and FPR@95.

## Complete Workflow Example

```bash
# 1. Extract logits for all datasets
python eval/extract_logits.py --input "datasets/RoadAnomaly21/images/*.png" --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --save_dir ./saved_logits
python eval/extract_logits.py --input "datasets/RoadObsticle21/images/*.webp" --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --save_dir ./saved_logits
python eval/extract_logits.py --input "datasets/FS_LostFound_full/images/*.png" --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --save_dir ./saved_logits
python eval/extract_logits.py --input "datasets/fs_static/images/*.jpg" --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --save_dir ./saved_logits
python eval/extract_logits.py --input "datasets/RoadAnomaly/images/*.jpg" --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --save_dir ./saved_logits

# 2. Evaluate logits
python eval/evaluate_logits.py --logits_dir ./saved_logits/RoadAnomaly21
python eval/evaluate_logits.py --logits_dir ./saved_logits/RoadObsticle21
python eval/evaluate_logits.py --logits_dir ./saved_logits/FS_LostFound_full
python eval/evaluate_logits.py --logits_dir ./saved_logits/fs_static
python eval/evaluate_logits.py --logits_dir ./saved_logits/RoadAnomaly

# 3. Evaluate mIoU
python eval/eval_iou_eomt.py --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" --ckpt_path ./checkpoints/eomt_cityscapes.bin --datadir /path/to/cityscapes --subset val

# 4. Find optimal temperature (optional)
python eval/find_optimal_temperature.py --logits_dir ./saved_logits/RoadAnomaly21 --temp_range "0.5,0.75,1.0,1.1,1.5,2.0"
```
