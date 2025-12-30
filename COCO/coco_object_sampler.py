import json
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


class CocoObjectSampler:
    """
    COCO object sampler that returns (obj_rgb, obj_mask) suitable for OE pasting.

    Returns:
      obj_rgb:  (H,W,3) uint8 RGB
      obj_mask: (H,W)   uint8 {0,1}
    """

    # Cityscapes-overlapping categories to EXCLUDE (name-based)
    CITYSCAPES_OVERLAP = {
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        # optional: "traffic sign" (not a COCO class name, but keep in mind)
    }

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        allowed_class_names: Optional[List[str]] = None,
        filter_cityscapes: bool = True,
        min_area: int = 1000,
        allow_crowd: bool = False,
        seed: Optional[int] = 0,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.filter_cityscapes = bool(filter_cityscapes)
        self.min_area = int(min_area)
        self.allow_crowd = bool(allow_crowd)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")

        # Load COCO json
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Indexes
        self.categories = {c["id"]: c["name"] for c in coco["categories"]}
        self.images = {img["id"]: img for img in coco["images"]}
        self.annotations = coco["annotations"]

        # Decide allowed categories
        if allowed_class_names is not None:
            allowed_set = set(allowed_class_names)
            self.allowed_cat_ids = {cid for cid, name in self.categories.items() if name in allowed_set}
            if not self.allowed_cat_ids:
                raise ValueError("allowed_class_names provided, but none matched COCO category names.")
        else:
            # allow everything except Cityscapes overlap (if filter_cityscapes=True)
            if self.filter_cityscapes:
                self.allowed_cat_ids = {
                    cid for cid, name in self.categories.items()
                    if name not in self.CITYSCAPES_OVERLAP
                }
            else:
                self.allowed_cat_ids = set(self.categories.keys())

        # Pre-filter valid annotation indices (fast sampling later)
        self.valid_ann_indices = []
        for idx, ann in enumerate(self.annotations):
            if ann["category_id"] not in self.allowed_cat_ids:
                continue
            if (not self.allow_crowd) and ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("area", 0) < self.min_area:
                continue
            self.valid_ann_indices.append(idx)

        if not self.valid_ann_indices:
            raise RuntimeError("No valid annotations after filtering. Lower min_area or adjust filters.")

        print(f"[CocoObjectSampler] Images: {len(self.images)} | Anns: {len(self.annotations)}")
        print(f"[CocoObjectSampler] Allowed categories: {len(self.allowed_cat_ids)} | Valid anns: {len(self.valid_ann_indices)}")

    def get_allowed_categories(self) -> Dict[int, str]:
        return {cid: self.categories[cid] for cid in sorted(self.allowed_cat_ids)}

    def _decode_mask(self, ann: Dict, img_h: int, img_w: int) -> np.ndarray:
        seg = ann["segmentation"]

        if isinstance(seg, list):
            # polygons
            rles = mask_utils.frPyObjects(seg, img_h, img_w)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict):
            # rle (compressed or uncompressed)
            if isinstance(seg.get("counts", None), list):
                rle = mask_utils.frPyObjects(seg, img_h, img_w)
            else:
                rle = seg
        else:
            raise ValueError(f"Unknown segmentation format: {type(seg)}")

        m = mask_utils.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.uint8)  # {0,1}

    def _crop_xyxy_from_bbox(self, bbox, img_w, img_h):
        # bbox is [x,y,w,h] floats
        x, y, w, h = bbox
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(img_w, int(np.ceil(x + w)))
        y1 = min(img_h, int(np.ceil(y + h)))
        return x0, y0, x1, y1

    def sample(self, min_foreground_pixels: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (obj_rgb, obj_mask) where:
          obj_rgb:  (H,W,3) uint8 RGB
          obj_mask: (H,W)   uint8 {0,1}
        """
        for _ in range(300):
            ann = self.annotations[random.choice(self.valid_ann_indices)]
            img_info = self.images[ann["image_id"]]
            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            H, W = img.shape[:2]

            full_mask = self._decode_mask(ann, H, W)
            x0, y0, x1, y1 = self._crop_xyxy_from_bbox(ann["bbox"], W, H)
            if x1 <= x0 or y1 <= y0:
                continue

            obj_rgb = img[y0:y1, x0:x1].copy()
            obj_mask = full_mask[y0:y1, x0:x1].copy()

            # Basic cleanliness
            if obj_rgb.size == 0 or obj_mask.size == 0:
                continue
            if int(obj_mask.sum()) < int(min_foreground_pixels):
                continue

            # mask should not be all background or all foreground
            fill = obj_mask.mean()
            if fill < 0.05 or fill > 0.95:
                continue

            return obj_rgb.astype(np.uint8), obj_mask.astype(np.uint8)

        raise RuntimeError("Failed to sample a clean object after many retries.")

    def sample_with_meta(self, min_foreground_pixels: int = 200) -> Tuple[np.ndarray, np.ndarray, Dict]:
        obj_rgb, obj_mask = self.sample(min_foreground_pixels=min_foreground_pixels)

        # meta for debugging (not required by Person 4, but helpful)
        # Note: we re-sample to keep it simple; if you want exact meta, merge with sample() logic.
        # For Person 4, sample() alone is enough.
        return obj_rgb, obj_mask, {}

    # -------------------------
    # Simple visualization test
    # -------------------------
    @staticmethod
    def visualize(obj_rgb: np.ndarray, obj_mask: np.ndarray, title: str = ""):
        import matplotlib.pyplot as plt

        masked = obj_rgb.copy()
        masked[obj_mask == 0] = 0

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(obj_rgb)
        plt.title("obj_rgb")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(obj_mask, cmap="gray")
        plt.title("obj_mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(masked)
        plt.title("masked")
        plt.axis("off")

        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # IMPORTANT: match your screenshot paths
    sampler = CocoObjectSampler(
        images_dir="./data/coco/train2017",
        annotations_file="./data/coco/annotations_trainval2017/annotations/instances_train2017.json",
        # Option A: explicit allowed classes (recommended)
        allowed_class_names=[
            'dog','cat','horse','cow','sheep','bear','elephant','zebra','giraffe',
            'chair','couch','dining table','bed','toilet',
            'suitcase','backpack','handbag','umbrella',
            'tv','laptop','keyboard','microwave','oven','refrigerator',
            'book','clock','vase','potted plant',
            'bottle','cup','sports ball','skateboard','surfboard','teddy bear','frisbee','kite'
        ],
        filter_cityscapes=True,
        min_area=1000,
        allow_crowd=False,
        seed=0,
    )

    # Test: sample 5 and visualize
    for i in range(5):
        rgb, m = sampler.sample()
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
        assert m.ndim == 2 and m.dtype == np.uint8
        assert rgb.shape[:2] == m.shape[:2]
        assert m.max() <= 1 and m.min() >= 0
        print(f"[OK] sample {i+1}: rgb={rgb.shape}, mask_fg={int(m.sum())}")

        CocoObjectSampler.visualize(rgb, m, title=f"Sample {i+1}")
