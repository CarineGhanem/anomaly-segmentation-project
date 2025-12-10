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


def rba_pixel_scores(mask_logits, class_logits):
    # mask_logits: [Q, H, W]
    # class_logits: [Q, C]

    Q, H, W = mask_logits.shape
    Q2, C = class_logits.shape
    assert Q == Q2

    L = np.zeros((C, H, W), dtype=np.float32)

    # Loop SOLO sulle classi (C ~ 19 → veloce)
    for k in range(C):
        # [Q,H,W] = broadcast su una sola classe alla volta (RAM-safe)
        vals = class_logits[:, k, None, None] + mask_logits

        max_k = np.max(vals, axis=0)
        L[k] = max_k + np.log(np.sum(np.exp(vals - max_k), axis=0))

    return np.tanh(L).sum(axis=0)



def rba_region_scores(rba_map, gt_mask):
    """
    rba_map: [H, W] with RbA pixel scores
    gt_mask: [H, W] with 1 = OOD, 0 = ID
    Returns:
        region_scores: list of region-level scalar scores
        region_labels: list of 0/1 (1=OOD region, 0=ID region)
    """
    # ---------------------------------------------------------
    # Extract connected components for region-level scoring
    # ---------------------------------------------------------
    print("Rba computation...")
    labeled, num_regions = label(gt_mask == 1)

    region_scores = []
    region_labels = []

    # Add POSITIVE regions
    for rid in range(1, num_regions + 1):
        region = labeled == rid
        if region.sum() == 0:
            continue
        s = rba_map[region].max()
        region_scores.append(s)
        region_labels.append(1)

    # Add a NEGATIVE region (all ID pixels)
    # Sample multiple negative ID regions
    id_labeled, num_id_regions = label(gt_mask == 0)

    for rid in range(1, num_id_regions + 1):
        region = id_labeled == rid
        if region.sum() == 0:
            continue
        s = rba_map[region].max()
        region_scores.append(s)
        region_labels.append(0)

    return region_scores, region_labels



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
    for method in ["rba","msp", "max_logit", "entropy"]:
        print(f"\n=== Evaluating {method} ===")

        all_scores, all_gts = [], []
        region_scores_total, region_labels_total = [], []

        for f in files:
            data = np.load(f)
            pixel_logits = data["pixel_logits"].astype(np.float32, copy=False)    # [C,H,W]
            mask_logits = data["mask_logits"].astype(np.float32, copy=False)
            class_logits = data["class_logits"].astype(np.float32, copy=False)

            gt = data["mask_gt"]                   # [H,W]

            if class_logits.ndim == 3 and class_logits.shape[0] == 1:
                class_logits = class_logits[0]

            # Case 2: (Q, C, 1)
            if class_logits.ndim == 3 and class_logits.shape[2] == 1:
                class_logits = class_logits[:, :, 0]

            # Case 3: wrong shape → print and skip
            if class_logits.ndim != 2:
                print("ERROR: invalid class_logits shape:", class_logits.shape, "file:", f)
                continue
            if mask_logits.ndim == 4 and mask_logits.shape[0] == 1:
                mask_logits = mask_logits[0]

            # Resize logits if GT mask resolution differs
            C, Hm, Wm = pixel_logits.shape
            H_gt, W_gt = gt.shape
            if (Hm, Wm) != (H_gt, W_gt):
                pixel_logits = np.stack([
                    cv2.resize(pixel_logits[c], (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
                    for c in range(C)
                ]).astype(np.float32)

            if mask_logits.shape[1:] != gt.shape:
                mask_logits = np.stack([
                    cv2.resize(mask_logits[q], (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
                    for q in range(mask_logits.shape[0])
                ]).astype(np.float32)

            # Rba
            if method == "rba":
                rba_map = rba_pixel_scores(mask_logits, class_logits)   
                rs, rl = rba_region_scores(rba_map, gt)
                region_scores_total.extend(rs)
                region_labels_total.extend(rl)
                continue
            # Pixel level anomaly map
            anomaly = anomaly_scores_from_pixel_logits(pixel_logits, method)

            all_scores.append(anomaly)
            all_gts.append(gt)

        # Compute metrics
        if method == "rba":
            prc = average_precision_score(region_labels_total, region_scores_total)
            fpr = fpr_at_95_tpr(region_scores_total, region_labels_total)

        else:
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