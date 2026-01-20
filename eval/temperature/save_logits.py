# Step 1: Save Model Logits (Run Once)
import os
import glob
import torch
import random
import pickle
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from erfnet import ERFNet
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, help="Input images glob pattern")
    parser.add_argument('--loadDir', default="trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--output_dir', default="saved_logits/", help="Directory to save logits")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # Extract dataset name for organizing saved logits
    input_path = str(args.input)
    dataset_name = "Unknown"
    if "RoadAnomaly21" in input_path:
        dataset_name = "RoadAnomaly21"
    elif "RoadObsticle21" in input_path:
        dataset_name = "RoadObsticle21"
    elif "FS_LostFound_full" in input_path or "LostFound" in input_path:
        dataset_name = "FS_LostFound_full"
    elif "fs_static" in input_path:
        dataset_name = "fs_static"
    elif "RoadAnomaly" in input_path and "RoadAnomaly21" not in input_path:
        dataset_name = "RoadAnomaly"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{dataset_name}_logits.pkl")
    
    # Setup device
    if not args.cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights
    
    print(f"Loading model: {modelpath}")
    print(f"Loading weights: {weightspath}")
    
    model = ERFNet(NUM_CLASSES)
    model = model.to(device)
    
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    continue
            else:
                own_state[name].copy_(param)
        return model
    
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()
    
    # Get all image paths
    image_paths = glob.glob(os.path.expanduser(args.input))
    print(f"Found {len(image_paths)} images")
    
    # Storage for logits and ground truth
    data_to_save = {
        'logits': [],
        'ground_truths': [],
        'image_paths': []
    }
    
    print(f"\nProcessing images and saving logits...")
    processed = 0
    
    for path in image_paths:
        # Load and process image
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().to(device)
        
        # Forward pass to get logits
        with torch.no_grad():
            logits = model(images)
        
        # Save logits to CPU
        logits_cpu = logits.cpu().numpy()
        
        # Load ground truth
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        
        # Process ground truth based on dataset
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)
        
        # Only save if there are anomalies
        if 1 in np.unique(ood_gts):
            data_to_save['logits'].append(logits_cpu)
            data_to_save['ground_truths'].append(ood_gts)
            data_to_save['image_paths'].append(path)
            processed += 1
        
        # Clear memory
        del logits, images, mask
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if processed % 10 == 0:
            print(f"Processed {processed} images...")
    
    # Save to pickle file
    print(f"\nSaving {processed} logits to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"✓ Done! Logits saved to {output_file}")
    print(f"  Total images with anomalies: {processed}")
    print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


if __name__ == '__main__':
    main()