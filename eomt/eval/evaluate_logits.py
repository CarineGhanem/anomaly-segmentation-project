# evaluate_logits.py

import os
import glob
import numpy as np
import cv2
from argparse import ArgumentParser

from sklearn.metrics import average_precision_score
from ood_metrics import fpr_at_95_tpr


def stable_softmax(pixel_logits):
    """Numerically safe softmax over class dimension [C,H,W]"""
    logits_max = np.max(pixel_logits, axis=0, keepdims=True)     # [1,H,W]
    exp_shifted = np.exp(pixel_logits - logits_max)
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)
    

def anomaly_scores_from_pixel_logits(pixel_logits, method):

    if method == "msp":
        softmax = stable_softmax(pixel_logits)
        MSP = np.max(softmax, axis=0)
        anomaly = 1.0 - MSP

    elif method == "entropy":
        softmax = stable_softmax(pixel_logits)
        anomaly = -np.sum(softmax * np.log(np.clip(softmax, 1e-12, None)), axis=0)

    elif method == "max_logit":
        anomaly = -np.max(pixel_logits, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    # clean up numeric issues (rare but safe)
    anomaly = np.nan_to_num(anomaly, nan=0.0, posinf=0.0, neginf=0.0)
    return anomaly



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--logits_dir", type=str, required=True,
                        help="Directory containing the saved .npz files")
    return parser.parse_args()


def evaluate_method(all_scores, all_gts):
    """
    all_scores: list of 2D arrays
    all_gts: list of 2D arrays
    """
    anomaly_scores = np.array(all_scores)
    ood_gts = np.array(all_gts)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    val_out = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.logits_dir, "*.npz")))

    results = {}

    for method in ["msp", "max_logit", "entropy"]:
        print(f"\n=== Evaluating {method} ===")

        all_scores = []
        all_gts = []

        for f in files:
            data = np.load(f)
            pixel_logits = data["pixel_logits"]  # [C,H,W]
            gt = data["mask_gt"]                # [H,W]

            # resize logits to GT shape if needed
            C, Hm, Wm = pixel_logits.shape
            H_gt, W_gt = gt.shape
            if (Hm, Wm) != (H_gt, W_gt):
                resized_logits = np.zeros((C, H_gt, W_gt))
                for c in range(C):
                    resized_logits[c] = cv2.resize(pixel_logits[c],
                                                   (W_gt, H_gt),
                                                   interpolation=cv2.INTER_LINEAR)
                pixel_logits = resized_logits

            anomaly_map = anomaly_scores_from_pixel_logits(pixel_logits, method)
            all_scores.append(anomaly_map)
            all_gts.append(gt)

        prc, fpr = evaluate_method(all_scores, all_gts)
        results[method] = (prc, fpr)
        print(f"AUPRC: {prc * 100:.2f}")
        print(f"FPR@95: {fpr * 100:.2f}")

    print("\nFinal Summary:")
    for method, (prc, fpr) in results.items():
        print(f"{method}: AUPRC {prc*100:.2f} | FPR95 {fpr*100:.2f}")


if __name__ == "__main__":
    main()
