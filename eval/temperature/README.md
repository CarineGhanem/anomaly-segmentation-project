## Temperature Scaling (ERFNet)

Temperature scaling is used to calibrate the confidence of ERFNet predictions
for anomaly detection. To enable fast experimentation, logits are saved once
and then reused to evaluate multiple temperature values without re-running inference.

### Step 1: Save logits
Run ERFNet once on the dataset and store the logits and ground-truth masks.

```bash
python eval/temperature/save_logits.py \
  --input "eomt/datasets/Validation_Dataset/RoadAnomaly21/images/*.png" \
  --loadDir "trained_models/" \
  --loadWeights "erfnet_pretrained.pth" \
  --output_dir "saved_logits" \
  --cpu
```
### Step 2: Evaluate multiple temperatures

Load the saved logits and evaluate anomaly detection performance for multiple
temperature values efficiently.
```bash

python eval/temperature/test_temperatures_fast.py \
  --logits_file "saved_logits/RoadAnomaly21_logits.pkl" \
  --method msp \
  --temperatures 0.5 0.75 1.0 1.1 1.25 1.5 2.0 \
  --output_file "eval/temperature/temperature_results.txt"
