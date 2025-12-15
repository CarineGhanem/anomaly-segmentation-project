# ============================================================
# ULTIMATE FIX: Handle internal num_classes adjustment
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# ============================================================
# Helper: Dataset-specific GT mask remapping
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
# Ultimate fix: Patch the state dict to add extra class
# ============================================================
def build_eomt_model(config, ckpt_path, device):
    """
    Add an extra class to the checkpoint to match model.
    Use this if the above approach doesn't work.
    """
    print("="*80)
    print("Loading EoMT checkpoint:", ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    checkpoint_num_classes = state_dict["network.class_head.weight"].shape[0]
    print(f"✓ Checkpoint has {checkpoint_num_classes} classes")

    # Build model normally (will likely create checkpoint_num_classes + 1)
    T = state_dict["network.encoder.backbone.pos_embed"].shape[1]
    grid = int(T ** 0.5)
    img_size = (grid * 16, grid * 16)

    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = enc_cfg["class_path"].rsplit(".", 1)
    Encoder = getattr(importlib.import_module(enc_mod), enc_cls)
    encoder = Encoder(img_size=img_size, **enc_cfg.get("init_args", {}))

    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    Network = getattr(importlib.import_module(net_mod), net_cls)
    net_kwargs = {k: v for k, v in net_cfg.get("init_args", {}).items() if k != "encoder"}

    network = Network(
        masked_attn_enabled=False,
        num_classes=checkpoint_num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    Lit = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    model = Lit(
        img_size=img_size,
        num_classes=checkpoint_num_classes,
        network=network,
        **model_kwargs,
    ).to(device).eval()

    # Check actual model num_classes
    actual_num_classes = model.network.class_head.weight.shape[0]
    
    if actual_num_classes != checkpoint_num_classes:
        print(f"\n⚠ Model has {actual_num_classes} classes (checkpoint has {checkpoint_num_classes})")
        print(f"→ Extending checkpoint to match model...")
        
        # Extend checkpoint class_head
        checkpoint_weight = state_dict["network.class_head.weight"]
        checkpoint_bias = state_dict["network.class_head.bias"]
        hidden_dim = checkpoint_weight.shape[1]
        
        # Create extended tensors
        extended_weight = torch.zeros(actual_num_classes, hidden_dim, dtype=checkpoint_weight.dtype)
        extended_bias = torch.zeros(actual_num_classes, dtype=checkpoint_bias.dtype)
        
        # Copy trained weights
        extended_weight[:checkpoint_num_classes] = checkpoint_weight
        extended_bias[:checkpoint_num_classes] = checkpoint_bias
        
        # Extra classes get small random initialization
        extra_classes = actual_num_classes - checkpoint_num_classes
        extended_weight[checkpoint_num_classes:] = torch.randn(extra_classes, hidden_dim) * 0.01
        
        # Replace in state dict
        state_dict["network.class_head.weight"] = extended_weight
        state_dict["network.class_head.bias"] = extended_bias
        
        print(f"  ✓ Extended class_head to {actual_num_classes} classes")
        print(f"  ✓ First {checkpoint_num_classes} classes: from checkpoint (trained)")
        print(f"  ✓ Last {extra_classes} class(es): random init (unused for inference)")

    # Remove training weights
    for k in list(state_dict.keys()):
        if any(x in k for x in ["criterion", "empty_weight"]):
            del state_dict[k]

    # Load
    model.load_state_dict(state_dict, strict=False)
    
    print("✓ Model ready with extended class_head")
    print("="*80 + "\n")
    
    return model, img_size, checkpoint_num_classes  # Return original count


# ============================================================
# Forward pass
# ============================================================
def eomt_forward_logits(model, img_tensor, device, debug=False):
    """
    Forward pass through EoMT.
    
    Args:
        num_classes_to_use: If specified, only use first N classes from output.
                           Useful when model has extra classes we want to ignore.
    """
    H, W = img_tensor.shape[-2:]
    img_uint8 = (img_tensor * 255).clamp(0, 255).byte()

    imgs = [img_uint8.to(device)]
    img_sizes = [(H, W)]

    amp_ctx = autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    with torch.no_grad(), amp_ctx:
        crops, origins = model.window_imgs_semantic(imgs)
        
        if debug:
            print(f"  Crops: {len(crops)}, shape: {crops[0].shape}")
        
        mask_logits_layers, class_logits_layers = model(crops)
        
        if debug:
            print(f"  Mask logits: {mask_logits_layers[-1].shape}")
            print(f"  Class logits: {class_logits_layers[-1].shape}")

        mask_logits = F.interpolate(
            mask_logits_layers[-1], 
            size=(H, W), 
            mode="bilinear", 
            align_corners=False
        )

        class_logits = class_logits_layers[-1]

        per_pixel = model.to_per_pixel_logits_semantic(
            mask_logits, 
            class_logits
        )
        
        stitched = model.revert_window_logits_semantic(per_pixel, origins, img_sizes)
        pixel_logits = stitched[0]  # [C, H, W]

    return pixel_logits, mask_logits, class_logits


# ============================================================
# Verification
# ============================================================
def verify_logits(logits, num_classes, img_path=None):
    """Sanity checks for extracted logits."""
    C, H, W = logits.shape
    
    print("\n" + "="*80)
    print("LOGIT VERIFICATION")
    if img_path:
        print(f"Image: {os.path.basename(img_path)}")
    print("="*80)
    
    assert C == num_classes, f"Expected {num_classes} classes, got {C}"
    print(f"✓ Class dimension: {C}")
    
    assert not torch.isnan(logits).any(), "❌ Logits contain NaN!"
    assert not torch.isinf(logits).any(), "❌ Logits contain Inf!"
    print(f"✓ No NaN or Inf")
    
    min_val, max_val = logits.min().item(), logits.max().item()
    print(f"✓ Logit range: [{min_val:.2f}, {max_val:.2f}]")
    
    probs = F.softmax(logits, dim=0)
    max_probs, pred_classes = probs.max(dim=0)
    
    print(f"✓ Predicted classes: [{pred_classes.min()}, {pred_classes.max()}]")
    print(f"✓ Confidence range: [{max_probs.min():.4f}, {max_probs.max():.4f}]")
    print(f"✓ Mean confidence: {max_probs.mean():.4f}")
    
    prob_sum = probs.sum(dim=0)
    sum_ok = torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-4)
    
    if sum_ok:
        print(f"✓ Probabilities sum to 1.0")
    else:
        print(f"⚠ Probability sum deviation: {(prob_sum - 1.0).abs().max():.6f}")
    
    unique_preds = pred_classes.unique()
    print(f"✓ Unique classes in image: {len(unique_preds)}/{num_classes}")
    
    print("="*80 + "\n")
    return sum_ok


# ============================================================
# MAIN
# ============================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="../../Validation_Dataset/RoadAnomaly21/images/*.png")
    parser.add_argument("--config", type=str,
                        default="../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml")
    parser.add_argument("--save_dir", type=str, default="./saved_logits")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    device = (
        torch.device("cpu") if args.cpu else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}\n")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["trainer"]["logger"]["init_args"]["name"]
    ckpt_path = hf_hub_download(f"tue-mps/{model_name}", "pytorch_model.bin")

    model, img_size, num_classes = build_eomt_model(config, ckpt_path, device)


    input_tf = Compose([Resize(img_size), ToTensor()])
    target_tf = Compose([Resize(img_size, Image.NEAREST)])

    dataset_name = args.input.split("/")[-3]
    out_dir = os.path.join(args.save_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    image_paths = sorted(glob.glob(args.input))
    print(f"Processing {len(image_paths)} images...\n")
    
    for idx, path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(path))[0]
        print(f"[{idx+1}/{len(image_paths)}] {fname}")

        img = input_tf(Image.open(path).convert("RGB")).to(device)
        
        debug_first = args.debug and idx == 0
        pixel_logits, mask_logits, class_logits = eomt_forward_logits(
            model, img, device, debug=debug_first
        )
        
        if args.verify and idx == 0:
            verify_logits(pixel_logits, num_classes, path)

        pixel_logits = pixel_logits.detach().to(torch.float16)
        mask_logits  = mask_logits.detach().to(torch.float16)
        class_logits = class_logits.detach().to(torch.float16)

        pathGT = path.replace("images", "labels_masks") \
                     .replace(".jpg", ".png") \
                     .replace(".webp", ".png")
        gt = target_tf(Image.open(pathGT))
        mask_np = remap_gt_mask(np.array(gt), pathGT)

        save_path = os.path.join(out_dir, fname + ".npz")
        np.savez_compressed(
            save_path,
            pixel_logits=pixel_logits.cpu().numpy(),   # [C, H, W]
            mask_logits=mask_logits.cpu().numpy(),     # [Q, H, W]
            class_logits=class_logits.cpu().numpy(),   # [Q, C]
            mask_gt=mask_np,
            num_classes=num_classes
        )

    print(f"\n{'='*80}")
    print(f"✓ All logits saved to: {out_dir}")
    print(f"  Format: [{num_classes}, H, W] per-pixel logits")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()