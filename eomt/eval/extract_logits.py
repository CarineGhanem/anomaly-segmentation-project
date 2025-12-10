# ============================================================
# EoMT LOGIT EXTRACTION SCRIPT
# Outputs per-pixel logits & ground-truth OOD masks as .npz files
# Used for offline evaluation (evaluate_logits.py)

# RoadAnomaly21
#& "D:\python3.11\pyhon3.11\python.exe" extract_logits.py --input "../../../Validation_Dataset/Validation_Dataset/RoadAnomaly21/images/*.png" --config "../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml" --save_dir ./saved_logits

# RoadObsticle21
#& "D:\python3.11\pyhon3.11\python.exe" extract_logits.py --input "../../../Validation_Dataset/Validation_Dataset/RoadObsticle21/images/*.webp" --config "../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml" --save_dir ./saved_logits

# FS_LostFound_full
#& "D:\python3.11\pyhon3.11\python.exe" extract_logits.py --input "../../../Validation_Dataset/Validation_Dataset/FS_LostFound_full/images/*.png" --config "../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml" --save_dir ./saved_logits

# fs_static
#& "D:\python3.11\pyhon3.11\python.exe" extract_logits.py --input "../../../Validation_Dataset/Validation_Dataset/fs_static/images/*.jpg" --config "../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml" --save_dir ./saved_logits

# RoadAnomaly
#& "D:\python3.11\pyhon3.11\python.exe" extract_logits.py --input "../../../Validation_Dataset/Validation_Dataset/RoadAnomaly/images/*.jpg" --config "../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml" --save_dir ./saved_logits
# ============================================================

import os
import sys
import glob
import torch
import random
import yaml
import importlib
import numpy as np
from argparse import ArgumentParser
from PIL import Image

from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
from torchvision.transforms import Compose, Resize, ToTensor

from huggingface_hub import hf_hub_download

# ============================================================
# Reproducibility
# ============================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Get the root directory of your project (adjust if needed) 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(project_root)


# ============================================================
# Helper: Dataset-specific GT mask remapping
# Converts dataset labels → binary OOD mask {0 in-distribution, 1 anomaly, 255 ignore}
# ============================================================
def remap_gt_mask(mask, pathGT):
    """
    Remap dataset-specific labels to standard OOD mask format:
    - 0: in-distribution (normal)
    - 1: anomaly (out-of-distribution)
    - 255: ignore (background/void)
    """
    if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
        # RoadAnomaly: 0=road, 1=background, 2=anomaly
        # Convert: 0,1 -> 0 (normal), 2 -> 1 (anomaly)
        mask = np.where(mask == 2, 1, 0)  # 2 -> 1, others -> 0
    elif "RoadAnomaly21" in pathGT or "RoadObsticle21" in pathGT:
        # RoadAnomaly21/RoadObsticle21 format: 0=normal, 1=anomaly, 255=ignore
        # Already in standard format, no remapping needed!
        pass  # No remapping needed - already in correct format
    elif "LostAndFound" in pathGT or "FS_LostFound" in pathGT:
        # FS_LostFound_full format: 0=normal, 1=anomaly, 255=ignore
        # Already in standard format, no remapping needed!
        # (Some LostAndFound versions use 0=void, 1=road, >1=anomaly, but FS_LostFound_full uses standard format)
        pass  # No remapping needed - already in correct format
    elif "fs_static" in pathGT:
        # fs_static format: 0=normal, 1=anomaly, 255=ignore
        # Already in standard format, no remapping needed!
        pass  # No remapping needed - already in correct format
    else:
        # If no match, print warning but don't modify mask
        print(f"WARNING: Unknown dataset in path {pathGT}, mask not remapped!")
    
    return mask


# ============================================================
# Build EoMT model (encoder + mask decoder only)
# ============================================================
def build_eomt_model(config, ckpt_path, device):
    print("Loading EoMT checkpoint:", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    # --- Infer #classes from class_head ----
    num_classes = state_dict["network.class_head.weight"].shape[0]

    # --- Infer image size from ViT positional embeddings ----
    T = state_dict["network.encoder.backbone.pos_embed"].shape[1]
    grid = int(T ** 0.5)
    img_size = (grid * 16, grid * 16)  # ViT patch size = 16

    # --- Build encoder ---
    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = enc_cfg["class_path"].rsplit(".", 1)
    Encoder = getattr(importlib.import_module(enc_mod), enc_cls)
    encoder = Encoder(img_size=img_size, **enc_cfg.get("init_args", {}))

    # --- Build EoMT network ---
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    Network = getattr(importlib.import_module(net_mod), net_cls)
    net_kwargs = {k: v for k, v in net_cfg.get("init_args", {}).items() if k != "encoder"}

    network = Network(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    # --- Build Lightning wrapper ---
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    Lit = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    model = Lit(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).to(device).eval()

    # Check if model has correct num_classes, if not, replace class_head
    actual_num_classes = model.network.class_head.weight.shape[0]
    if actual_num_classes != num_classes:
        print(f"WARNING: Model has {actual_num_classes} classes but checkpoint has {num_classes} classes")
        print("Replacing class_head to match checkpoint...")
        
        # Replace class_head to match checkpoint
        import torch.nn as nn
        hidden_dim = model.network.class_head.weight.shape[1]
        model.network.class_head = nn.Linear(hidden_dim, num_classes).to(device)
        
        # Replace criterion.empty_weight if it exists
        if hasattr(model, 'criterion') and hasattr(model.criterion, 'empty_weight'):
            model.criterion.empty_weight = torch.ones(num_classes, device=device)
    
    # Filter state_dict to only include keys that match model structure
    # IMPORTANT: We need to keep class_head weights for correct class_logits generation!
    model_state = model.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping {k}: shape mismatch (model: {model_state[k].shape}, checkpoint: {v.shape})")
        else:
            print(f"Skipping {k}: not in model")
    
    # Load filtered weights (including class_head!)
    model.load_state_dict(filtered_state_dict, strict=False)

    print("EoMT model ready | img_size:", img_size, "| num_classes:", num_classes, "\n")
    return model, img_size


# ============================================================
# Forward pass: returns pixel logits [C, H, W]
# ============================================================
def eomt_forward_logits(model, img_tensor, device):
    H, W = img_tensor.shape[-2:]
    img_uint8 = (img_tensor * 255).clamp(0, 255).byte()

    imgs = [img_uint8.to(device)]
    img_sizes = [(H, W)]

    amp_ctx = autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    with torch.no_grad(), amp_ctx:
        crops, origins = model.window_imgs_semantic(imgs)
        mask_logits_layers, class_logits_layers = model(crops)

        mask_logits = F.interpolate(
            mask_logits_layers[-1], size=(H, W), mode="bilinear", align_corners=False
        )

        per_pixel = model.to_per_pixel_logits_semantic(mask_logits, class_logits_layers[-1])
        stitched = model.revert_window_logits_semantic(per_pixel, origins, img_sizes)

    return stitched[0]  # [C, H, W]


# ============================================================
# MAIN SCRIPT
# ============================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="../../Validation_Dataset/RoadAnomaly21/images/*.png")
    parser.add_argument("--config", type=str,
                        default="../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml")
    parser.add_argument("--save_dir", type=str, default="./saved_logits")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # --- Select device ---
    device = (
        torch.device("cpu") if args.cpu else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print("Device:", device)

    # --- Load YAML config ---
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Resolve HF checkpoint ---
    model_name = config["trainer"]["logger"]["init_args"]["name"]
    ckpt_path = hf_hub_download(f"tue-mps/{model_name}", "pytorch_model.bin")

    # --- Build model ---
    model, img_size = build_eomt_model(config, ckpt_path, device)

    # --- Transforms ---
    input_tf = Compose([Resize(img_size), ToTensor()])
    target_tf = Compose([Resize(img_size, Image.NEAREST)])

    # --- Determine dataset name for saving folder ---
    dataset_name = args.input.split("/")[-3]
    out_dir = os.path.join(args.save_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # LOOP: Extract logits & GT masks
    # ============================================================
    for path in glob.glob(args.input):
        fname = os.path.splitext(os.path.basename(path))[0]
        print("Processing:", fname)

        # --- Load + preprocess image ---
        img = input_tf(Image.open(path).convert("RGB")).to(device)

        # --- Forward pass → pixel logits ---
        logits = eomt_forward_logits(model, img, device)
        logits_np = logits.cpu().numpy()

        # --- Load + remap GT mask ---
        pathGT = path.replace("images", "labels_masks") \
                     .replace(".jpg", ".png") \
                     .replace(".webp", ".png")

        # Check if GT file exists
        if not os.path.exists(pathGT):
            print(f"WARNING: GT file not found: {pathGT}, skipping...")
            continue

        gt = target_tf(Image.open(pathGT))
        mask_np = remap_gt_mask(np.array(gt), pathGT)

        # --- Save .npz ---
        save_path = os.path.join(out_dir, fname + ".npz")
        np.savez_compressed(save_path, pixel_logits=logits_np, mask_gt=mask_np)

    print("\nAll logits saved to:", out_dir)


if __name__ == "__main__":
    main()
