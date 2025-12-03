# ============================================================
# evaluate_logits.py
# Evaluates saved EoMT logits using MSP / entropy / max-logit.
# Produces AUPRC + FPR@95 and updates a global results.txt file.
# ============================================================

import os
import glob
import numpy as np
import cv2
from argparse import ArgumentParser

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr


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

    else:
        raise ValueError(f"Unknown method: {method}")

    return np.nan_to_num(anomaly, nan=0.0)


# ============================================================
# Parse CLI arguments
# ============================================================
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory containing saved .npz logits")
    return parser.parse_args()


# ============================================================
# Compute AUPRC + FPR@95 for a given method
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
# MAIN EVALUATION LOOP
# ============================================================
def main():
    args = parse_args()

    # Load logits paths
    files = sorted(glob.glob(os.path.join(args.logits_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError("No .npz files found in logits_dir")

    # Determine dataset name (folder name)
    dataset_name = os.path.basename(args.logits_dir.rstrip("/"))

    # Global results file path (one level above saved_logits/<dataset>/)
    save_file = os.path.abspath(os.path.join(args.logits_dir, "../../results.txt"))

    results = {}

    # Evaluate 3 anomaly scoring methods
    for method in ["msp", "max_logit", "entropy"]:
        print(f"\n=== Evaluating {method} ===")

        all_scores, all_gts = [], []

        for f in files:
            data = np.load(f)
            pixel_logits = data["pixel_logits"]     # [C,H,W]
            gt = data["mask_gt"]                   # [H,W]

            # Resize logits if GT mask resolution differs
            C, Hm, Wm = pixel_logits.shape
            H_gt, W_gt = gt.shape
            if (Hm, Wm) != (H_gt, W_gt):
                resized = np.zeros((C, H_gt, W_gt), dtype=pixel_logits.dtype)
                for c in range(C):
                    resized[c] = cv2.resize(pixel_logits[c], (W_gt, H_gt),
                                            interpolation=cv2.INTER_LINEAR)
                pixel_logits = resized

            # Compute anomaly map
            anomaly = anomaly_scores_from_pixel_logits(pixel_logits, method)

            all_scores.append(anomaly)
            all_gts.append(gt)

        # Compute metrics
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

            # Keep line unless BOTH dataset and method match exactly
            if line_dataset == dataset_name and line_method == method:
                continue  # this is the old entry → delete it

            updated.append(line)
        # Add new line and write file
        updated.append(new_line)
        with open(save_file, "w") as f:
            f.writelines(updated)

        print(new_line.strip())

    print(f"\n✓ Saved results to: {save_file}")

    # Summary printout
    print("\nFinal Summary:")
    for method, (prc, fpr) in results.items():
        print(f"{method}: AUPRC={prc*100:.2f} | FPR95={fpr*100:.2f}")


if __name__ == "__main__":
    main()
