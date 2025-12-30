from pycocotools.coco import COCO
import requests
import os
from tqdm import tqdm
import zipfile
from pathlib import Path

def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_coco_annotations():
    """Download and extract COCO annotations if not present."""
    # Check both possible locations
    possible_paths = [
        Path('./data/coco/annotations/instances_train2017.json'),
        Path('./data/coco/annotations_trainval2017/annotations/instances_train2017.json')
    ]
    
    for annotations_file in possible_paths:
        if annotations_file.exists():
            print(f"✓ Annotations found at: {annotations_file}")
            return str(annotations_file)
    
    # If not found, download
    annotations_dir = Path('./data/coco/annotations')
    annotations_file = annotations_dir / 'instances_train2017.json'
    
    print("Annotations not found. Downloading...")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Download annotations zip
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    zip_path = './data/coco/annotations_trainval2017.zip'
    
    print("Downloading COCO annotations (~252 MB)...")
    download_file(annotations_url, zip_path, "Downloading annotations")
    
    # Extract
    print("\nExtracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data/coco/')
    
    # Clean up zip
    os.remove(zip_path)
    print("✓ Annotations extracted successfully")
    
    return str(annotations_file)

# Main script
print("="*60)
print("COCO Anomaly Dataset Downloader")
print("="*60)

# Step 1: Ensure annotations exist
annFile = setup_coco_annotations()

# Step 2: Load COCO annotations
print("\nLoading COCO annotations...")
coco = COCO(annFile)

# Anomaly categories prioritized for FS Static failures
anomaly_classes = [
    # HIGH PRIORITY - Animals 
    'dog', 'cat', 'horse', 'cow', 'sheep', 'bear', 'elephant', 'zebra', 'giraffe',
    
    # HIGH PRIORITY - Furniture 
    'chair', 'couch', 'dining table', 'bed', 'toilet',
    
    # HIGH PRIORITY - Cargo/Personal items
    'suitcase', 'backpack', 'handbag', 'umbrella',
    
    # MEDIUM PRIORITY - Indoor objects
    'tv', 'laptop', 'keyboard', 'microwave', 'oven', 'refrigerator',
    'book', 'clock', 'vase', 'potted plant',
    
    # MEDIUM PRIORITY - Sports/misc
    'bottle', 'cup', 'sports ball', 'skateboard', 'surfboard',
    'teddy bear', 'frisbee', 'kite'
]

print(f"\n{'='*60}")
print(f"Selected {len(anomaly_classes)} anomaly categories")
print(f"{'='*60}")
for i, cat in enumerate(anomaly_classes, 1):
    print(f"{i:2d}. {cat}")

# Get category IDs
catIds = coco.getCatIds(catNms=anomaly_classes)
print(f"\n✓ Found {len(catIds)} valid categories in COCO")

# Get all annotations for these categories first
annIds = coco.getAnnIds(catIds=catIds)
print(f"✓ Total objects available: {len(annIds)}")

# Get unique image IDs from annotations
imgIds = list(set([coco.anns[annId]['image_id'] for annId in annIds]))
print(f"✓ Found {len(imgIds)} images with anomaly objects")

# Limit to 2000 images (enough for 1500+ objects)
imgIds = imgIds[:2000]
print(f"✓ Will download {len(imgIds)} images")

# Create directory
images_dir = Path('./data/coco/train2017')
images_dir.mkdir(parents=True, exist_ok=True)

# Check how many already exist
existing = sum(1 for imgId in imgIds 
               if (images_dir / coco.loadImgs(imgId)[0]['file_name']).exists())
print(f"✓ {existing} images already exist, will download {len(imgIds) - existing} new images")

# Download images
print(f"\n{'='*60}")
print("Downloading images...")
print(f"{'='*60}")
downloaded = 0
failed = 0
skipped = 0

for imgId in tqdm(imgIds, desc="Progress"):
    img_info = coco.loadImgs(imgId)[0]
    img_url = img_info['coco_url']
    img_path = images_dir / img_info['file_name']
    
    # Skip if already exists
    if img_path.exists():
        skipped += 1
        continue
    
    try:
        response = requests.get(img_url, timeout=15)
        response.raise_for_status()
        with open(img_path, 'wb') as f:
            f.write(response.content)
        downloaded += 1
    except Exception as e:
        failed += 1
        # Optionally log the error
        # print(f"\nFailed to download {img_info['file_name']}: {e}")
        continue

print(f"\n{'='*60}")
print("Download Complete!")
print(f"{'='*60}")
print(f"  Downloaded:     {downloaded:4d} new images")
print(f"  Already exist:  {skipped:4d} images")
print(f"  Failed:         {failed:4d} images")
print(f"  Total:          {downloaded + skipped:4d} images")
print(f"\n  Images saved to: {images_dir.absolute()}")
print(f"  Annotations at:  {Path(annFile).absolute()}")
print(f"{'='*60}")

# Summary stats
print(f"\nDataset Summary:")
print(f"  Categories:     {len(catIds)}")
print(f"  Images:         {len(imgIds)}")
print(f"  Objects:        {len(annIds)}")
if len(imgIds) > 0:
    print(f"  Avg objects/img: {len(annIds)/len(imgIds):.1f}")
print(f"\n✓ Ready to use with CocoObjectSampler!")