import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

# Cityscapes label mapping: labelId -> trainId
# Based on cityscapesScripts
LABEL_MAPPING = {
    0: 255,   # unlabeled -> ignore
    1: 255,   # ego vehicle -> ignore
    2: 255,   # rectification border -> ignore
    3: 255,   # out of roi -> ignore
    4: 255,   # static -> ignore
    5: 255,   # dynamic -> ignore
    6: 255,   # ground -> ignore
    7: 0,     # road
    8: 1,     # sidewalk
    9: 255,   # parking -> ignore
    10: 255,  # rail track -> ignore
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail -> ignore
    15: 255,  # bridge -> ignore
    16: 255,  # tunnel -> ignore
    17: 5,    # pole
    18: 255,  # polegroup -> ignore
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan -> ignore
    30: 255,  # trailer -> ignore
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    255: 255  # license plate -> ignore
}

def convert_labelIds_to_trainIds(labelIds_path, output_path):
    """Convert a single labelIds image to labelTrainIds format"""
    # Load labelIds image
    label_img = np.array(Image.open(labelIds_path))
    
    # Create output array - initialize with 255 for unmapped values
    trainIds_img = np.full_like(label_img, 255, dtype=np.uint8)
    
    # Map each labelId to trainId
    for labelId, trainId in LABEL_MAPPING.items():
        trainIds_img[label_img == labelId] = trainId
    
    # Check for unmapped values 
    unmapped_mask = ~np.isin(label_img, list(LABEL_MAPPING.keys()))
    if unmapped_mask.any():
        unmapped_values = np.unique(label_img[unmapped_mask])
        print(f"Warning: Found unmapped labelId values in {labelIds_path}: {unmapped_values}")
        print(f"  These pixels will be set to 255 (ignore)")
    
    # Save as labelTrainIds
    Image.fromarray(trainIds_img).save(output_path)
    return True

def process_directory(gtFine_dir, subset='val'):
    """Process all labelIds files in a subset directory"""
    subset_path = Path(gtFine_dir) / subset
    
    if not subset_path.exists():
        print(f"Error: {subset_path} does not exist")
        return
    
    # Find all labelIds files
    labelIds_files = list(subset_path.rglob("*_labelIds.png"))
    
    if len(labelIds_files) == 0:
        print(f"No labelIds files found in {subset_path}")
        return
    
    print(f"Found {len(labelIds_files)} labelIds files")
    print("Converting to labelTrainIds...")
    
    converted = 0
    for labelIds_file in labelIds_files:
        # Generate output filename
        trainIds_file = labelIds_file.parent / labelIds_file.name.replace("_labelIds.png", "_labelTrainIds.png")
        
        # Convert
        try:
            convert_labelIds_to_trainIds(labelIds_file, trainIds_file)
            converted += 1
            if converted % 50 == 0:
                print(f"Converted {converted}/{len(labelIds_files)} files...")
        except Exception as e:
            print(f"Error converting {labelIds_file}: {e}")
    
    print(f"\nConversion complete! Converted {converted}/{len(labelIds_files)} files")
    print(f"Output files saved in: {subset_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert Cityscapes labelIds to labelTrainIds')
    parser.add_argument('--gtFine_dir', type=str, required=True,
                        help='Path to gtFine directory (e.g., E:/advanced machine learning/project-kevser/gtFine)')
    parser.add_argument('--subset', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Subset to convert (default: val)')
    
    args = parser.parse_args()
    
    process_directory(args.gtFine_dir, args.subset)

if __name__ == '__main__':
    main()

