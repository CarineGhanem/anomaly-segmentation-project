# EoMT

This is almost the original repository of the authors of EoMT if something is not clear refer to the [original repo](https://github.com/tue-mps/eomt). You will have to use the code in this folder and adapt it with the eval folder to be able to evaluate and train a EoMT model if needed. You can find a EoMT model trained on Cityscapes dataset with the [config file](eomt/configs/dinov2/cityscapes/semantic) at this [link](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Requirements Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation for training

You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch (don't do it, it will be impossible to do it in Colab due to resource contraints):

```bash
python3 main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on Citiscapes segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.

✅ Make sure the total batch size is `devices × batch_size = 16`
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

## Fine-tuning EoMT (optional)

Fine-tuning EoMT is configuration-driven and implemented using **PyTorch Lightning**.
The core training logic is in `eomt/training/lightning_module.py`, while experiment settings are controlled by YAML configuration files.

> **Note:** Fine-tuning is **not used in the core project experiments** and is included as an optional extension.

---

### Key Features

- **Stage-based fine-tuning**: Control which parameters to train via `finetune_stage`
- **LogitNorm calibration**: Temperature-scaled normalization during training (not applied during inference)
- **Magnitude-aware training**: Loss variants: `"mse"`, `"margin"`, `"hinge"`, `"soft_margin"`
- **LoRA support**: Parameter-efficient fine-tuning

---

### 1) Choose a training configuration

Configs are located in: `eomt/configs/dinov2/cityscapes/semantic/`

**Examples:**
- `eomt_base_640.yaml` – Base setup
- `logit_norm_ablation/stageA_class_head_only.yaml` – Train classification head only
- `magnitude_aware_training/*.yaml` – Various magnitude loss configurations

---

### 2) Prepare required paths

You need:
- **Cityscapes** dataset directory
- **Pretrained EoMT checkpoint** (e.g., `eomt_cityscapes.bin`)

---

### 3) Run fine-tuning

From the repository root:
```bash
cd eomt

python main.py fit \
  -c "configs/dinov2/cityscapes/semantic/magnitude_aware_training/margin_magnitude.yaml" \
  --data.init_args.path "/path/to/cityscapes" \
  --model.init_args.ckpt_path "/path/to/eomt_cityscapes.bin"
```