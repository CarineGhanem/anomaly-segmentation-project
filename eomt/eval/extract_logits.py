# ============================================================
# EoMT LOGIT EXTRACTION SCRIPT
# Outputs per-pixel logits & ground-truth OOD masks as .npz files
# Used for offline evaluation (evaluate_logits.py)
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
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Get the root directory of your project (adjust if needed) 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(project_root)


# ============================================================
# Helper: Dataset-specific GT mask remapping
# Converts dataset labels → binary OOD mask {0 in-distribution, 1 anomaly}
# ============================================================
def remap_gt_mask(mask, pathGT):
    if "RoadAnomaly" in pathGT:
        mask = np.where(mask == 2, 1, mask)
    elif "LostAndFound" in pathGT:
        mask = np.where(mask == 0, 255, mask)
        mask = np.where(mask == 1, 0, mask)
        mask = np.where((mask > 1) & (mask < 201), 1, mask)
    elif "Streethazard" in pathGT:
        mask = np.where(mask == 14, 255, mask)
        mask = np.where(mask < 20, 0, mask)
        mask = np.where(mask == 255, 1, mask)
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

    # --- Drop class_head + criterion weights ---
    for k in list(state_dict.keys()):
        if any(x in k for x in ["class_head", "criterion", "empty_weight"]):
            del state_dict[k]

    # --- Load remaining weights ---
    model.load_state_dict(state_dict, strict=False)

    print("✓ EoMT model ready | img_size:", img_size, "\n")
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
        class_logits = class_logits_layers[-1]

        per_pixel = model.to_per_pixel_logits_semantic(mask_logits, class_logits_layers[-1])
        stitched = model.revert_window_logits_semantic(per_pixel, origins, img_sizes)

    return stitched[0], mask_logits, class_logits  # [C, H, W]


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
        pixel_logits, mask_logits, class_logits = eomt_forward_logits(model, img, device)
        pixel_logits = pixel_logits.detach().to(dtype=torch.float16)
        mask_logits  = mask_logits.detach().to(dtype=torch.float16)
        class_logits = class_logits.detach().to(dtype=torch.float16)

        logits_np       = pixel_logits.cpu().numpy()
        mask_np_logits  = mask_logits.cpu().numpy()
        class_np_logits = class_logits.cpu().numpy()
        # --- Load + remap GT mask ---
        pathGT = path.replace("images", "labels_masks") \
                     .replace(".jpg", ".png") \
                     .replace(".webp", ".png")

        gt = target_tf(Image.open(pathGT))
        mask_np = remap_gt_mask(np.array(gt), pathGT)

        # --- Save .npz ---
        save_path = os.path.join(out_dir, fname + ".npz")
        np.savez(save_path, pixel_logits=logits_np, mask_logits=mask_np_logits, class_logits=class_np_logits, mask_gt=mask_np)

    print("\n✓ All logits saved to:", out_dir)


if __name__ == "__main__":
    main()
