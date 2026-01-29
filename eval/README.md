# ERFNet Evaluation Usage Guide

This guide explains how to use the evaluation scripts in `eval/` for anomaly segmentation with ERFNet.

## Scripts

The evaluation pipeline consists of:

1. **`evalAnomaly.py`**: Evaluates anomaly detection using post-hoc scoring methods (MSP, Max Logit, Max Entropy)
2. **`eval_iou.py`**: Evaluates semantic segmentation performance (mIoU) on Cityscapes dataset
3. **`temperature/save_logits.py`**: Saves model logits for fast temperature scaling evaluation
4. **`temperature/test_temperatures_fast.py`**: Tests different temperature values on saved logits

## How to Load Checkpoints

The script loads ERFNet checkpoints from the `trained_models/` folder by default:

```bash
  python evalAnomaly.py --input "datasets/RoadAnomaly21/images/*.png" --loadDir ../trained_models/ --loadWeights erfnet_pretrained.pth
```

## Evaluation Datasets

Supported datasets:
- **RoadAnomaly21**: Real street scenes with diverse anomalies
- **RoadObstacle21**: Road-obstacle scenes where anomalies lie on the roadway
- **FS_LostFound_full** (Fishyscapes Lost & Found): Real-world road hazards
- **fs_static** (Fishyscapes Static): Synthetic anomalies pasted into Cityscapes scenes
- **RoadAnomaly**: Collection of real unusual road hazards

## How to Evaluate Anomaly Detection

Run `evalAnomaly.py` to compute anomaly detection metrics:

```bash
# MSP
python evalAnomaly.py --input "datasets/RoadAnomaly21/images/*.png" --anomaly_score msp

# Max Logit 
python evalAnomaly.py --input "datasets/RoadAnomaly21/images/*.png" --anomaly_score max_logit

# Max Entropy
python evalAnomaly.py --input "datasets/RoadAnomaly21/images/*.png" --anomaly_score entropy
```

The script evaluates post-hoc anomaly scores:
- **MSP** (Maximum Softmax Probability): `1 - max(softmax(logits))`
- **Max Logit**: `-max(logits)`
- **Max Entropy**: `-sum(softmax(logits) * log(softmax(logits)))`

Metrics reported:
- **AuPRC** (Area Under Precision-Recall Curve): Higher is better
- **FPR@95** (False Positive Rate at 95% True Positive Rate): Lower is better

Results are saved to `results.txt` with format:
```
Dataset: RoadAnomaly21 | Method: msp | AuPRC: XX.XX | FPR95: XX.XX
```

## How to Evaluate mIoU

Run `eval_iou.py` to evaluate semantic segmentation performance on Cityscapes:

```bash
python eval_iou.py --datadir /path/to/cityscapes --subset val --loadDir ../trained_models/ --loadWeights erfnet_pretrained.pth --results-file results.txt
```

The script reports:
- **Per-class IoU**: Intersection-over-Union for each of the 19 Cityscapes classes
- **Mean IoU (mIoU)**: Average IoU across all classes

## Temperature

Temperature scaling improves calibration by adjusting softmax sharpness: `logits_scaled = logits / temperature`

**PRO TIP**: Save logits once and test multiple temperatures offline to avoid re-running inference.

### Step 1: Save Logits

```bash
python temperature/save_logits.py --input "datasets/RoadAnomaly21/images/*.png" --output_dir ./saved_logits
```

### Step 2: Test Temperatures

```bash
python temperature/test_temperatures_fast.py --logits_file ./saved_logits/RoadAnomaly21_logits.pkl --method msp --temperatures 0.5 0.75 1.0 1.1 1.5 2.0 2.5 3.0 5.0 10.0
```

### Temperature Scaling

- **T < 1**: Sharper distribution (more confident predictions)
- **T = 1**: No scaling (default)
- **T > 1**: Smoother distribution (less confident predictions)

The script evaluates each temperature value and reports the best AuPRC and FPR@95.
