"""import os
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
# Configuration
# ============================================================
pretrained_checkpoint = "/content/drive/MyDrive/eomt_cityscapes.bin"
data_path = "/content/drive/MyDrive"
drive_checkpoint_path = "/content/drive/MyDrive/stage_a_checkpoint_temp03.ckpt"

# ============================================================
# Enable Live Plotting (IMPORTANT!)
# ============================================================
# Make sure matplotlib displays inline
%matplotlib inline

print("="*60)
print("🎨 Live plotting enabled!")
print("   Plots will update after each validation epoch")
print("="*60)

# ============================================================
# Background Checkpoint Saver
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
                    print(f"\n[Auto-save] Checkpoint → Drive ({file_size:.2f} MB)")
        except Exception as e:
            print(f"[Error] {e}")

monitor_thread = threading.Thread(target=save_checkpoint_periodically, daemon=True)
monitor_thread.start()

# ============================================================
# Training Stage A with Live Plots
# ============================================================
print("\n" + "="*60)
print("Starting Stage A Training")
print("="*60)

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
# After Training: Save Checkpoint
# ============================================================
print("\n" + "="*60)
print("Training Completed!")
print("="*60)

checkpoint_base = "/content/anomaly-segmentation-project/eomt"
best_ckpts = glob.glob(f"{checkpoint_base}/**/best-*.ckpt", recursive=True)

if best_ckpts:
    latest_ckpt = max(best_ckpts, key=os.path.getmtime)
    shutil.copy(latest_ckpt, drive_checkpoint_path)
    file_size = os.path.getsize(drive_checkpoint_path) / (1024 * 1024)
    print(f"✅ Checkpoint: {drive_checkpoint_path} ({file_size:.2f} MB)")

print("\n📊 Final training curves saved to Drive")
print("="*60)
"""