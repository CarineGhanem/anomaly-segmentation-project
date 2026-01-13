import os
import sys
import torch
import time
import yaml
import importlib
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext

from huggingface_hub import hf_hub_download

# Add eval directory to path to import iouEval and dataset
eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval"))
sys.path.insert(0, eval_dir)
from iouEval import iouEval, getColorEntry
from dataset import cityscapes
from transform import Relabel, ToLabel

# Get the root directory of your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

NUM_CLASSES = 20  # EoMT outputs 20 classes (0-19), where 19 is background/void (ignore)


def create_transforms(img_size):
    """Create transforms based on model's image size"""
    input_transform = Compose([
        Resize(img_size, Image.BILINEAR),
        ToTensor(),
    ])
    
    target_transform = Compose([
        Resize(img_size, Image.NEAREST),
        ToLabel(),
        Relabel(255, 19),   # ignore label to 19
    ])
    
    return input_transform, target_transform


def build_eomt_model(config, ckpt_path, device):
    """Build EoMT model from config and checkpoint"""
    print("Loading EoMT checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Handle Lightning checkpoint format (contains "state_dict" key)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"Loaded Lightning checkpoint (epoch: {ckpt.get('epoch', 'N/A')}, step: {ckpt.get('global_step', 'N/A')})")
    else:
        state_dict = ckpt

    # Infer #classes from class_head
    num_classes = state_dict["network.class_head.weight"].shape[0]

    # Infer image size from ViT positional embeddings
    T = state_dict["network.encoder.backbone.pos_embed"].shape[1]
    grid = int(T ** 0.5)
    img_size = (grid * 16, grid * 16)  # ViT patch size = 16

    # Build encoder
    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = enc_cfg["class_path"].rsplit(".", 1)
    Encoder = getattr(importlib.import_module(enc_mod), enc_cls)
    encoder = Encoder(img_size=img_size, **enc_cfg.get("init_args", {}))

    # Build EoMT network
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    Network = getattr(importlib.import_module(net_mod), net_cls)
    net_kwargs = {k: v for k, v in net_cfg.get("init_args", {}).items() if k != "encoder"}
    # Override num_classes from checkpoint (not from config)
    net_kwargs["num_classes"] = num_classes

    network = Network(
        masked_attn_enabled=False,
        encoder=encoder,
        **net_kwargs,
    )

    # Build Lightning wrapper
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    Lit = getattr(importlib.import_module(lit_mod), lit_cls)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    # Override num_classes from checkpoint (not from config)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["img_size"] = img_size
    model_kwargs["network"] = network

    model = Lit(**model_kwargs).to(device).eval()

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
    
    # Load filtered weights
    model.load_state_dict(filtered_state_dict, strict=False)

    print("EoMT model ready | img_size:", img_size, "| num_classes:", num_classes)
    return model, img_size


def eomt_forward_predictions(model, img_tensor, device, target_size):
    """Forward pass: returns semantic segmentation predictions [H, W]"""
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

        # Important: Do NOT use LogitNorm during inference (standard practice from paper)
        per_pixel = model.to_per_pixel_logits_semantic(
            mask_logits, 
            class_logits_layers[-1],
            use_logit_norm=False,  # Never use LogitNorm during inference
        )
        stitched = model.revert_window_logits_semantic(per_pixel, origins, img_sizes)

    # Get predictions: [C, H, W] -> [H, W]
    logits = stitched[0]  # [C, H, W] - shape is [20, H, W] for Cityscapes
    predictions = torch.argmax(logits, dim=0)  # [H, W]
    
    # Resize to target size if needed
    if (H, W) != target_size:
        predictions = predictions.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        predictions = F.interpolate(
            predictions, size=target_size, mode="nearest"
        ).squeeze(0).squeeze(0).long()  # [H, W]

    # Keep predictions in range [0, 19] - iouEval with nClasses=20 can handle this
    # Prediction 19 (background/void) will be treated as ignore by iouEval
    predictions = torch.clamp(predictions, 0, 19)

    return predictions


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="../configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
                        help="Path to YAML config file. Default: eomt_base_640.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Local path to checkpoint file (if not on HuggingFace). If not provided, will try to download from HuggingFace.")
    parser.add_argument("--datadir", type=str,
                        default="E:\\advanced machine learning\\project-kevser")
    parser.add_argument("--subset", type=str, default="val")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--results-file", type=str, default="results.txt",
                        help="File to save mIoU results")
    args = parser.parse_args()

    # Select device
    device = (
        torch.device("cpu") if args.cpu else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print("Using device:", device)

    # Load YAML config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve checkpoint path
    if args.ckpt_path:
        # Use local checkpoint file
        ckpt_path = args.ckpt_path
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint file not found: {ckpt_path}")
            return
        print(f"Using local checkpoint: {ckpt_path}")
    else:
        # Try to download from HuggingFace
        try:
            model_name = config["trainer"]["logger"]["init_args"]["name"]
            ckpt_path = hf_hub_download(f"tue-mps/{model_name}", "pytorch_model.bin")
            print(f"Downloaded checkpoint from HuggingFace: {ckpt_path}")
        except Exception as e:
            print(f"ERROR: Could not download checkpoint from HuggingFace: {e}")
            print(f"Model name: {config['trainer']['logger']['init_args']['name']}")
            print("Please provide --ckpt_path with local checkpoint file path")
            return

    # Build model
    model, img_size = build_eomt_model(config, ckpt_path, device)
    
    # Create transforms based on model's image size
    input_transform_cityscapes, target_transform_cityscapes = create_transforms(img_size)
    print(f"Using image size: {img_size}")

    # Check Cityscapes directory structure
    leftImg8bit_path = os.path.join(args.datadir, "leftImg8bit", args.subset)
    gtFine_path = os.path.join(args.datadir, "gtFine", args.subset)

    if not os.path.exists(leftImg8bit_path) or not os.path.exists(gtFine_path):
        print(f"Error: Cityscapes directory structure not found!")
        print(f"Expected: {leftImg8bit_path} and {gtFine_path}")
        return

    # Create dataset loader
    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=19)  # 19 is ignore class (background/void)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Get predictions from EoMT
        predictions = eomt_forward_predictions(model, images[0], device, labels.shape[-2:])
        predictions = predictions.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Labels are already processed by target_transform (255 -> 19)
        # Add batch to IoU evaluator
        iouEvalVal.addBatch(predictions.data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[-1] if "leftImg8bit/" in filename[0] else filename[0]
        if (step + 1) % 50 == 0:
            print(f"Processed {step + 1}/{len(loader)}: {filenameSave}")

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    class_names = [
        "Road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i] * 100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time() - start, "seconds")
    print("=======================================")
    print("Per-Class IoU:")
    for i, name in enumerate(class_names):
        print(iou_classes_str[i], name)
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")

    # Save mIoU to results file
    results_file = args.results_file
    model_name = "EOMT"

    # Remove old mIoU result for this model if exists
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(results_file, 'w', encoding='utf-8') as f:
            for line in lines:
                # Keep line if it doesn't match current model mIoU entry
                if not (f'Model: {model_name} | mIoU:' in line):
                    f.write(line)

    # Append new mIoU result
    with open(results_file, 'a', encoding='utf-8') as f:
        result_line = f'Model: {model_name} | mIoU: {iouVal * 100:.2f}\n'
        f.write(result_line)
        print(f"\nmIoU saved to {results_file}: {iouVal * 100:.2f}%")


if __name__ == '__main__':
    main()

