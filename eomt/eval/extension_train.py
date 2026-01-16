'''
import getpass
token = getpass.getpass("GitHub Token: ")
!git clone https://{token}@github.com/CarineGhanem/anomaly-segmentation-project.git
%cd anomaly-segmentation-project
!ls -la
!git branch
!git status
%cd /content/anomaly-segmentation-project/eomt
'''


'''
!pip install -q gitignore_parser==0.1.12 jsonargparse[signatures]==4.38 matplotlib==3.10.1 timm==1.0.15 wandb==0.19.10 lightning==2.5.1.post0 transformers==4.56.1 scipy==1.15.2 torch==2.7.0 torchvision==0.22.0 ipykernel==6.29.5 fvcore==0.1.5.post20221221 torchmetrics==1.7.1 pycocotools==2.0.8
print("requirements installed")
'''
'''
import os
import torch
import glob
import shutil
import time
import threading
from google.colab import drive

drive.mount('/content/drive')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if torch.cuda.is_available():
    torch.cuda.empty_cache()

%cd /content/anomaly-segmentation-project/eomt

pretrained_checkpoint = "/content/drive/MyDrive/eomt_cityscapes.bin"  # pretrained checkpoint
data_path = "/content/drive/MyDrive"
drive_checkpoint_path = "/content/drive/MyDrive/stage_a_checkpointfixtwo.ckpt"


def save_checkpoint_periodically():
    checkpoint_base = "/content/anomaly-segmentation-project/eomt"
    while True:
        time.sleep(600)  
        try:
           
            all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)
            if all_ckpts:
                latest_ckpt = max(all_ckpts, key=os.path.getmtime)
               
                if not os.path.exists(drive_checkpoint_path) or \
                   os.path.getmtime(latest_ckpt) > os.path.getmtime(drive_checkpoint_path):
                    shutil.copy(latest_ckpt, drive_checkpoint_path)
                    file_size = os.path.getsize(drive_checkpoint_path) / (1024 * 1024)
                    print(f"\n[saved] Checkpoint updated to Drive ({file_size:.2f} MB)")
        except Exception as e:
            print(f"[error] {e}")

monitor_thread = threading.Thread(target=save_checkpoint_periodically, daemon=True)
monitor_thread.start()

# ============================================================
# Training Stage A
# ============================================================
!python main.py fit \
  --config configs/dinov2/cityscapes/semantic/stageA.yaml \
  --model.init_args.ckpt_path "{pretrained_checkpoint}" \
  --data.init_args.path "{data_path}" \
  --model.init_args.load_ckpt_class_head=false \
  --model.init_args.logit_norm_temp=0.3 \
  --trainer.max_epochs=20 \
  --trainer.accumulate_grad_batches=16 \
  --trainer.num_sanity_val_steps=1 \
  --trainer.check_val_every_n_epoch=1 \
  --data.init_args.batch_size=8 \
  --data.init_args.num_workers=8 \
  --compile_disabled

# ============================================================
# After training, save the final checkpoint
# ============================================================
print("\n" + "="*60)
print("Training completed, saving final checkpoint...")
print("="*60)

# Find the latest checkpoint (prioritize outputs/stage_a/best-*.ckpt)
checkpoint_base = "/content/anomaly-segmentation-project/eomt"
best_ckpt_pattern = f"{checkpoint_base}/outputs/stage_a/best-*.ckpt"

# Find the best checkpoint
best_ckpts = glob.glob(best_ckpt_pattern)
if best_ckpts:
    latest_ckpt = max(best_ckpts, key=os.path.getmtime)
else:
    # If not found, find all checkpoints
    all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)
    if all_ckpts:
        # Exclude pretrained model
        stage_a_ckpts = [ckpt for ckpt in all_ckpts if "eomt_cityscapes" not in ckpt.lower()]
        if stage_a_ckpts:
            latest_ckpt = max(stage_a_ckpts, key=os.path.getmtime)
        else:
            latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    else:
        latest_ckpt = None

if latest_ckpt and os.path.exists(latest_ckpt):
    shutil.copy(latest_ckpt, drive_checkpoint_path)
    file_size = os.path.getsize(drive_checkpoint_path) / (1024 * 1024)
    print(f"[OK] Final checkpoint saved to Drive: {drive_checkpoint_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Source file: {latest_ckpt}")
    print(f"\nFor Stage B training:")
    print(f"  --model.init_args.ckpt_path \"{drive_checkpoint_path}\"")
else:
    print(f"[ERROR] Checkpoint file not found")
    # List all possible checkpoints
    all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)
    if all_ckpts:
        print("\nFound the following checkpoints:")
        for ckpt in sorted(all_ckpts, key=os.path.getmtime, reverse=True)[:5]:
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            mtime = time.ctime(os.path.getmtime(ckpt))
            print(f"  - {ckpt} ({size_mb:.2f} MB, {mtime})")
'''

'''
import os
import torch
import glob
import shutil
import time
import threading

from google.colab import drive
drive.mount('/content/drive')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
if torch.cuda.is_available():
    torch.cuda.empty_cache()

%cd /content/anomaly-segmentation-project/eomt

# ============================================================
# Configuration path
# ============================================================
stage_a_checkpoint = "/content/drive/MyDrive/stage_a_checkpointfix.ckpt"  # Stage A checkpoint
data_path = "/content/drive/MyDrive"
drive_checkpoint_path = "/content/drive/MyDrive/stage_b_checkpointfixtempone.ckpt"

print("="*60)
print("Stage B: Query Embeddings Fine-tuning")
print("="*60)


# ============================================================
# Background monitoring thread: save checkpoint every 10 minutes
# ============================================================
def save_checkpoint_periodically():
    checkpoint_base = "/content/anomaly-segmentation-project/eomt"
    while True:
        time.sleep(600)  
        try:
           
            all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)
            if all_ckpts:
                latest_ckpt = max(all_ckpts, key=os.path.getmtime)
               
                if not os.path.exists(drive_checkpoint_path) or \
                   os.path.getmtime(latest_ckpt) > os.path.getmtime(drive_checkpoint_path):
                    shutil.copy(latest_ckpt, drive_checkpoint_path)
                    file_size = os.path.getsize(drive_checkpoint_path) / (1024 * 1024)
                    print(f"\n[saved] Checkpoint updated to Drive ({file_size:.2f} MB)")
        except Exception as e:
            print(f"[error] {e}")

# Start background monitoring
monitor_thread = threading.Thread(target=save_checkpoint_periodically, daemon=True)
monitor_thread.start()

# ============================================================
# Training Stage B
# ============================================================
!python main.py fit \
  --config configs/dinov2/cityscapes/semantic/stageB.yaml \
  --model.init_args.ckpt_path "{stage_a_checkpoint}" \
  --data.init_args.path "{data_path}" \
  --model.init_args.load_ckpt_class_head=true \
  --trainer.max_epochs=20 \
  --trainer.accumulate_grad_batches=16 \
  --trainer.num_sanity_val_steps=1 \
  --trainer.check_val_every_n_epoch=1 \
  --data.init_args.batch_size=8 \
  --data.init_args.num_workers=8 \
  --compile_disabled

# ============================================================
# After training, save the final checkpoint
# ============================================================
print("\n" + "="*60)
print("Training completed, saving final checkpoint...")
print("="*60)

# Find the latest checkpoint
checkpoint_base = "/content/anomaly-segmentation-project/eomt"
all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)

if all_ckpts:
    # Find the latest checkpoint (exclude Stage A)
    stage_b_ckpts = [ckpt for ckpt in all_ckpts if "stage_a" not in ckpt.lower()]
    if stage_b_ckpts:
        latest_ckpt = max(stage_b_ckpts, key=os.path.getmtime)
    else:
        latest_ckpt = max(all_ckpts, key=os.path.getmtime)

    if os.path.exists(latest_ckpt):
        shutil.copy(latest_ckpt, drive_checkpoint_path)
        file_size = os.path.getsize(drive_checkpoint_path) / (1024 * 1024)
        print(f"[OK] Final checkpoint saved to Drive: {drive_checkpoint_path}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Source file: {latest_ckpt}")
        print(f"\nFor Stage C training:")
        print(f"  --model.init_args.ckpt_path \"{drive_checkpoint_path}\"")
    else:
        print(f"[ERROR] Checkpoint file not found: {latest_ckpt}")
else:
    print("[ERROR] No checkpoint file found")
    # List all possible checkpoints
    all_ckpts = glob.glob(f"{checkpoint_base}/**/*.ckpt", recursive=True)
    if all_ckpts:
        print("\nFound the following checkpoints:")
        for ckpt in sorted(all_ckpts, key=os.path.getmtime, reverse=True)[:5]:
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            mtime = time.ctime(os.path.getmtime(ckpt))
            print(f"  - {ckpt} ({size_mb:.2f} MB, {mtime})")
'''