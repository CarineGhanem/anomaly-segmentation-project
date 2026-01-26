# ============================================================
# evaluate_logits.py
# Evaluates saved EoMT logits using MSP / entropy / max-logit.
# Produces AUPRC + FPR@95 and updates a global results.txt file.
# ============================================================

# python evaluate_logits.py --logits_dir ./saved_logits/RoadAnomaly21 --results_file results.txt
# python evaluate_logits.py --logits_dir ./saved_logits/RoadObsticle21 --results_file results.txt
# python evaluate_logits.py --logits_dir ./saved_logits/FS_LostFound_full --results_file results.txt
# python evaluate_logits.py --logits_dir ./saved_logits/fs_static --results_file sresults.txt
# python evaluate_logits.py --logits_dir ./saved_logits/RoadAnomaly --results_file results.txt

import os
import glob
import numpy as np
import cv2
from argparse import ArgumentParser

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr
from scipy.ndimage import label

# ============================================================
# Stable softmax for [C,H,W] logits (avoids overflow)
# ============================================================
def stable_softmax(pixel_logits):
    logits_max = np.max(pixel_logits, axis=0, keepdims=True)
    exp_shifted = np.exp(pixel_logits - logits_max)
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)


# ============================================================
# Compute anomaly map from pixel logits
# ============================================================
def anomaly_scores_from_pixel_logits(pixel_logits, method):
    if method == "msp":  # 1 - max softmax prob
        sm = stable_softmax(pixel_logits)
        anomaly = 1.0 - np.max(sm, axis=0)

    elif method == "entropy":  # pixel-wise entropy
        sm = stable_softmax(pixel_logits)
        anomaly = -np.sum(sm * np.log(np.clip(sm, 1e-12, None)), axis=0)

    elif method == "max_logit":  # negative max logit
        anomaly = -np.max(pixel_logits, axis=0)

    elif method == "rba":
        # Official-style RbA on dense logits:
        # RbA(x) = - sum_c tanh(logits_c(x))
        # Higher (closer to 0) => more anomalous
        anomaly = -np.tanh(pixel_logits).sum(axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.nan_to_num(anomaly, nan=0.0)


# ============================================================
# Parse CLI arguments
# ============================================================
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory containing the saved .npz files")
    parser.add_argument("--results_file", type=str, default="results.txt",
                        help="File to save results")
    return parser.parse_args()


# ============================================================
# Compute AUPRC + FPR@95 for a given method 
# ============================================================
def evaluate_method(all_scores, all_gts):
    """
    all_scores: list of 2D arrays
    all_gts:    list of 2D arrays
    """
    anomaly_scores = np.concatenate([score.flatten() for score in all_scores])
    ood_gts = np.concatenate([gt.flatten() for gt in all_gts])

    # Filter out background pixels (255)
    valid_mask = (ood_gts != 255)
    anomaly_scores = anomaly_scores[valid_mask]
    ood_gts = ood_gts[valid_mask]

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    # Check for empty arrays
    if len(ood_out) == 0:
        print("WARNING: No anomaly pixels found!")
        return 0.0, np.nan
    if len(ind_out) == 0:
        print("WARNING: No normal pixels found!")
        return 0.0, np.nan

    val_out = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

    prc_auc = average_precision_score(val_label, val_out)
    fpr95 = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr95


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================
def main():
    args = parse_args()
    # Load logits paths
    files = sorted(glob.glob(os.path.join(args.logits_dir, "*.npz")))
    save_file = args.results_file
    if len(files) == 0:
        print(f"ERROR: No .npz files found in {args.logits_dir}")
        return
    # Extract dataset name from logits_dir
    dataset_name = os.path.basename(os.path.normpath(args.logits_dir))

    results = {}

    # Evaluate 3 anomaly scoring methods
    for method in ["rba","msp", "max_logit", "entropy"]:
        print(f"\n=== Evaluating {method} ===")

        all_scores, all_gts = [], []
        region_scores_total, region_labels_total = [], []

        for f in files:
            data = np.load(f)
            gt = data["mask_gt"].astype(np.uint8, copy=False)
            pixel_logits = data["pixel_logits"].astype(np.float32, copy=False)  # [C,H,W]

            # Validate GT mask values
            invalid_values = np.setdiff1d(np.unique(gt), [0, 1, 255])
            if len(invalid_values) > 0:
                print(f"WARNING: Invalid values in GT mask {os.path.basename(f)}: {invalid_values}")
                gt = np.where(np.isin(gt, [0, 1, 255]), gt, 255).astype(np.uint8)

            # Resize logits to GT shape if needed
            C, Hm, Wm = pixel_logits.shape
            H_gt, W_gt = gt.shape
            if (Hm, Wm) != (H_gt, W_gt):
                resized_logits = np.zeros((C, H_gt, W_gt), dtype=pixel_logits.dtype)
                for c in range(C):
                    resized_logits[c] = cv2.resize(
                        pixel_logits[c],
                        (W_gt, H_gt),
                        interpolation=cv2.INTER_LINEAR
                    )
                pixel_logits = resized_logits

            anomaly = anomaly_scores_from_pixel_logits(pixel_logits, method)  # [H,W]
            all_scores.append(anomaly)
            all_gts.append(gt)

        prc, fpr = evaluate_method(all_scores, all_gts)
        results[method] = (prc, fpr)
        # -------------------------------------------------------
        # Update results.txt (replace previous entries for dataset+method)
        # -------------------------------------------------------

        new_line = (
            f"Dataset: {dataset_name} | "
            f"Method: {method} | "
            f"AUPRC: {prc*100:.2f} | "
            f"FPR@TPR95: {fpr*100:.2f}\n"
        )

        # Load previous results
        existing = []
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                existing = f.readlines()

        updated = []
        for line in existing:
            # Parse dataset + method from the result entry
            parts = [p.strip() for p in line.split("|")]
            parsed = {p.split(":")[0].strip(): p.split(":")[1].strip() for p in parts if ":" in p}

            line_dataset = parsed.get("Dataset", None)
            line_method  = parsed.get("Method", None)
            if line_dataset == dataset_name and line_method == method:
                continue  
            updated.append(line)

        updated.append(new_line)
        with open(save_file, "w") as f:
            f.writelines(updated)

        print(new_line.strip())

    print(f"\n✓ Saved results to: {save_file}")

    print("\nFinal Summary:")
    for method, (prc, fpr) in results.items():
        fpr_str = f"{fpr*100:.2f}" if not np.isnan(fpr) else "nan"
        print(f"{method}: AUPRC {prc*100:.2f} | FPR95 {fpr_str}")


if __name__ == "__main__":
    main()
