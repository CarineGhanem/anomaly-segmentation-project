
# extract_logits.py

import os
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2

from transformers import AutoImageProcessor, EomtForUniversalSegmentation

# preprocessing for GT masks
from torchvision.transforms import Compose, Resize

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


def compute_pixel_logits(mask_logits, class_logits):
    Q, H, W = mask_logits.shape
    Q2, C = class_logits.shape
    assert Q == Q2

    # mask: [Q, H, W, 1]
    mask = mask_logits.unsqueeze(-1)

    # cls: [Q, 1, 1, C]
    cls = class_logits.view(Q, 1, 1, C)

    # pixel logits: [H, W, C]
    pixel = (mask * cls).sum(dim=0)

    # [C, H, W]
    return pixel.permute(2, 0, 1)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to images: '/path/*.png'")
    parser.add_argument("--model_id", type=str,
                        default="tue-mps/cityscapes_semantic_eomt_large_1024")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the logits .npz files")
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading EoMT model: {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = EomtForUniversalSegmentation.from_pretrained(args.model_id)
    model.eval()

    # device setup
    if not args.cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model.to(device)
    print("Using device:", device)

    # loop through images
    for path in glob.glob(args.input):

        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        mask_logits = outputs.masks_queries_logits[0]     # [Q, H, W]
        class_logits = outputs.class_queries_logits[0]    # [Q, C]

        pixel_logits = compute_pixel_logits(mask_logits, class_logits)
        pixel_logits_np = pixel_logits.cpu().numpy()

        # load GT mask
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png") 

        filename = pathGT.replace(".png", "")
        print(f"Processing {filename}")

        mask_gt = Image.open(pathGT)
        mask_gt = target_transform(mask_gt)
        ood_mask_np = np.array(mask_gt)

        #fetch the dataset name
        dataset = pathGT.split("/")[-3]
        # Create dataset folder under the save_dir
        dataset_dir = os.path.join(args.save_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        # Save file: saved_logits/<dataset>/<filename>.npz
        save_path = os.path.join(dataset_dir, f"{filename}.npz")
        np.savez_compressed(
            save_path,
            pixel_logits=pixel_logits_np,
            mask_gt=ood_mask_np,
        )

    print("✓ Saved all logits.")


if __name__ == "__main__":
    main()
