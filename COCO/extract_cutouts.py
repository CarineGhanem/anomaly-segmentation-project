import json
import random
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# -------------------------------------------------
# Paths (MATCH your screenshot structure)
# -------------------------------------------------
ANN_FILE = Path("./data/coco/annotations_trainval2017/annotations/instances_train2017.json")
IMAGES_DIR = Path("./data/coco/train2017")
OUT_DIR = Path("./data/coco_cutouts")

assert ANN_FILE.exists(), f"Missing annotation file: {ANN_FILE}"
assert IMAGES_DIR.exists(), f"Missing images directory: {IMAGES_DIR}"

# -------------------------------------------------
# Categories  to extract
# -------------------------------------------------
anomaly_classes = [
    'dog','cat','horse','cow','sheep','bear','elephant','zebra','giraffe',
    'chair','couch','dining table','bed','toilet',
    'suitcase','backpack','handbag','umbrella',
    'tv','laptop','keyboard','microwave','oven','refrigerator',
    'book','clock','vase','potted plant',
    'bottle','cup','sports ball','skateboard','surfboard','teddy bear','frisbee','kite'
]

# -------------------------------------------------
# Filters for "clean"
# -------------------------------------------------
MAX_CUTOUTS = 5000
MIN_AREA = 1000                 # skip tiny annotations (COCO ann["area"])
MIN_CROP_SIDE = 20              # skip tiny crops
MIN_FOREGROUND_PIXELS = 200     # skip near-empty masks after crop
MIN_FILL = 0.05                 # mask coverage lower bound
MAX_FILL = 0.95                 # mask coverage upper bound
ALLOW_CROWD = False             # recommended False

random.seed(0)
np.random.seed(0)


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    """
    Decode COCO segmentation to a binary mask (H,W) uint8 {0,1}
    Supports polygons and RLE.
    """
    seg = ann["segmentation"]

    if isinstance(seg, list):
        # polygon(s)
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts", None), list):
        # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        # compressed RLE
        rle = seg

    m = mask_utils.decode(rle)  # (H,W) or (H,W,1)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component to remove mask speckles.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask01.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask01
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def main():
    print("Using annotations:", ANN_FILE)
    print("Using images dir :", IMAGES_DIR)

    coco = COCO(str(ANN_FILE))

    # Output folders
    rgb_dir = OUT_DIR / "rgb"
    msk_dir = OUT_DIR / "mask"
    meta_dir = OUT_DIR / "meta"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Category IDs
    cat_ids = coco.getCatIds(catNms=anomaly_classes)
    if not cat_ids:
        raise RuntimeError("No category IDs found. Check anomaly_classes names exactly match COCO.")

    # Annotation IDs for those categories
    ann_ids = coco.getAnnIds(catIds=cat_ids)
    random.shuffle(ann_ids)

    saved = 0
    skipped_missing_img = 0
    skipped_filters = 0

    pbar = tqdm(total=min(MAX_CUTOUTS, len(ann_ids)), desc="Saving cutouts")

    for ann_id in ann_ids:
        if saved >= MAX_CUTOUTS:
            break

        ann = coco.anns[ann_id]

        # Basic filters
        if (not ALLOW_CROWD) and ann.get("iscrowd", 0) == 1:
            skipped_filters += 1
            continue
        if ann.get("area", 0) < MIN_AREA:
            skipped_filters += 1
            continue

        # Load image
        img_info = coco.loadImgs([ann["image_id"]])[0]
        img_path = IMAGES_DIR / img_info["file_name"]
        if not img_path.exists():
            skipped_missing_img += 1
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            skipped_missing_img += 1
            continue

        H, W = img_bgr.shape[:2]
        full_mask = ann_to_mask(ann, H, W)

        # Crop bbox
        x, y, bw, bh = ann["bbox"]
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(W, int(np.ceil(x + bw)))
        y1 = min(H, int(np.ceil(y + bh)))

        if x1 <= x0 or y1 <= y0:
            skipped_filters += 1
            continue

        crop_bgr = img_bgr[y0:y1, x0:x1]
        crop_mask = full_mask[y0:y1, x0:x1]

        ch, cw = crop_mask.shape[:2]
        if ch < MIN_CROP_SIDE or cw < MIN_CROP_SIDE:
            skipped_filters += 1
            continue

        fg = int(crop_mask.sum())
        if fg < MIN_FOREGROUND_PIXELS:
            skipped_filters += 1
            continue

        fill = fg / float(ch * cw)
        if fill < MIN_FILL or fill > MAX_FILL:
            skipped_filters += 1
            continue

        # Cleanup mask a bit
        crop_mask = keep_largest_component(crop_mask)
        kernel = np.ones((3, 3), np.uint8)
        crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, kernel)

        # Save
        cat_name = coco.loadCats([ann["category_id"]])[0]["name"].replace(" ", "_")
        base = f"ann_{ann_id}_{cat_name}"

        cv2.imwrite(str(rgb_dir / f"{base}.png"), crop_bgr)  # keep BGR for OpenCV
        cv2.imwrite(str(msk_dir / f"{base}.png"), (crop_mask * 255).astype(np.uint8))

        meta = {
            "ann_id": ann_id,
            "image_id": ann["image_id"],
            "category": cat_name,
            "bbox_xyxy": [x0, y0, x1, y1],
            "area": float(ann.get("area", 0)),
            "iscrowd": int(ann.get("iscrowd", 0)),
            "img_file": img_info["file_name"],
        }
        with open(meta_dir / f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        saved += 1
        pbar.update(1)

    pbar.close()
    print("\nDONE.")
    print("Saved:", saved)
    print("Skipped (missing images):", skipped_missing_img)
    print("Skipped (filters):", skipped_filters)
    print("Output folder:", OUT_DIR.resolve())
    print("Example output dirs:")
    print(" ", rgb_dir.resolve())
    print(" ", msk_dir.resolve())
    print(" ", meta_dir.resolve())


if __name__ == "__main__":
    main()
