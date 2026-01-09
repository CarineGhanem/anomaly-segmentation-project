# ============================================================
# find_optimal_temperature.py
# Finds optimal temperature for MSP method ONLY
# Uses saved logits from extract_logits.py
# ============================================================

import os
import glob
import numpy as np
import cv2
from argparse import ArgumentParser

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr


# ============================================================
# Stable softmax with temperature scaling
# ============================================================
def stable_softmax(pixel_logits, temperature=1.0):
    """
    Apply temperature scaling before softmax
    T < 1: sharper (more confident)
    T > 1: smoother (less confident)
    """
    scaled_logits = pixel_logits / temperature
    logits_max = np.max(scaled_logits, axis=0, keepdims=True)
    exp_shifted = np.exp(scaled_logits - logits_max)
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)


# ============================================================
# Compute MSP anomaly score with temperature
# ============================================================
def msp_anomaly_score(pixel_logits, temperature=1.0):
    """MSP: 1 - max softmax probability"""
    sm = stable_softmax(pixel_logits, temperature)
    anomaly = 1.0 - np.max(sm, axis=0)
    return np.nan_to_num(anomaly, nan=0.0)


# ============================================================
# Parse CLI arguments
# ============================================================
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory containing saved .npz logits")

   
    parser.add_argument("--temp_range", type=str,
                        default="0.5,0.75,1.0,1.1,1.5,2.0,2.5,3.0,5.0,10.0",
                        help="Comma-separated temperature values to test")
    return parser.parse_args()


# ============================================================
# Compute AUPRC + FPR@95
# ============================================================
def evaluate_method(all_scores, all_gts):
    anomaly_scores = np.array(all_scores)
    ood_gts = np.array(all_gts)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_vals = anomaly_scores[ood_mask]
    ind_vals = anomaly_scores[ind_mask]

    # Build binary labels
    val_out = np.concatenate([ind_vals, ood_vals])
    val_label = np.concatenate([
        np.zeros(len(ind_vals)),
        np.ones(len(ood_vals))
    ])

    prc_auc = average_precision_score(val_label, val_out)
    fpr95 = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr95


# ============================================================
# MAIN TEMPERATURE SEARCH - MSP ONLY
# ============================================================
def main():
    args = parse_args()

    FIXED_MIOU_PERCENT = 80.50

    # Parse temperature range
    temperatures = [float(t) for t in args.temp_range.split(",") if t.strip() != ""]
    print(f"Testing temperatures: {temperatures}\n")

    # Load logits paths
    files = sorted(glob.glob(os.path.join(args.logits_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError("No .npz files found in logits_dir")

    # Determine dataset name
    dataset_name = os.path.basename(args.logits_dir.rstrip("/"))

    # Output files
  
    table_file = os.path.join(args.logits_dir, "Temperaturetest.txt")

    print("=" * 60)
    print("MSP Temperature Search")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Files: {len(files)}")
    print(f"Fixed mIoU (semantic): {FIXED_MIOU_PERCENT:.2f}")
    print("=" * 60)
    print()

    best_temp = 1.0
    best_auprc = 0.0
    best_fpr95 = 100.0

    # Store results per temperature for table writing
    # results[temp] = (auprc, fpr95)
    results = {}

    # Test each temperature
    for temp in temperatures:
        all_scores, all_gts = [], []

        # Process all files with current temperature
        for f in files:
            data = np.load(f)
            pixel_logits = data["pixel_logits"]  # [C,H,W]
            gt = data["mask_gt"]                 # [H,W]

            # Resize logits if needed
            C, Hm, Wm = pixel_logits.shape
            H_gt, W_gt = gt.shape
            if (Hm, Wm) != (H_gt, W_gt):
                resized = np.zeros((C, H_gt, W_gt), dtype=pixel_logits.dtype)
                for c in range(C):
                    resized[c] = cv2.resize(pixel_logits[c], (W_gt, H_gt),
                                            interpolation=cv2.INTER_LINEAR)
                pixel_logits = resized

            # Compute MSP anomaly score with temperature
            anomaly = msp_anomaly_score(pixel_logits, temp)

            all_scores.append(anomaly)
            all_gts.append(gt)

        # Evaluate
        prc_auc, fpr95 = evaluate_method(all_scores, all_gts)
        results[temp] = (prc_auc, fpr95)

        print(f"T={temp:>5.2f} → AUPRC: {prc_auc*100:>6.2f} | FPR@95: {fpr95*100:>6.2f}")

        # Track best based on AUPRC (primary metric)
        if prc_auc > best_auprc:
            best_auprc = prc_auc
            best_fpr95 = fpr95
            best_temp = temp

    print()
    print("=" * 60)
    print(f"✓ Best Temperature: {best_temp}")
    print(f"  mIoU (fixed): {FIXED_MIOU_PERCENT:.2f}")
    print(f"  AUPRC: {best_auprc*100:.2f}")
    print(f"  FPR@95: {best_fpr95*100:.2f}")
    print("=" * 60)


    # Save Temperaturetest.txt in YOUR requested table row format
    def row(temp):
        if temp in results:
            au, fp = results[temp]
            return f"{FIXED_MIOU_PERCENT:.2f}\t{au*100:.2f}\t{fp*100:.2f}"
        return f"{FIXED_MIOU_PERCENT:.2f}\tNA\tNA"

    with open(table_file, "w") as f:
        f.write(f"{dataset_name}\n")
        f.write("Method\tmIoU\tAuPRC\tFPR95\n")
        f.write(f"MSP\t{row(1.0)}\n")
        f.write(f"MSP(t = 0.5)\t{row(0.5)}\n")
        f.write(f"MSP(t = 0.75)\t{row(0.75)}\n")
        f.write(f"MSP(t = 1.1)\t{row(1.1)}\n")
        f.write(f"MSP (best t = {best_temp})\t{FIXED_MIOU_PERCENT:.2f}\t{best_auprc*100:.2f}\t{best_fpr95*100:.2f}\n")

   
    print(f"✓ Temperature table saved to: {table_file}")


if __name__ == "__main__":
    main()
