# EoMT: Anomaly Segmentation with Logit Normalization and Magnitude-Aware Training

This repository contains our implementation of anomaly segmentation using the Encoder-only Mask Transformer (EoMT) with logit normalization and magnitude-aware regularization.


**Checkpoints and Datasets**: Pre-trained models, evaluation datasets, and results are available in our [shared drive folder](https://drive.google.com/drive/folders/1Lc8KCF1ZsfYe7m7tjXIOicdymhETFsS1?usp=sharing).

## `eomt/eval`

The `eomt/eval` folder contains scripts for evaluating anomaly segmentation performance. The evaluation pipeline consists of:

1. **Logit Extraction** (`extract_logits.py`): Processes images through a trained model and saves per-pixel logit vectors as `.npz` files. This enables offline evaluation without re-running inference.

2. **Logit Evaluation** (`evaluate_logits.py`): Computes anomaly detection metrics (AUPRC, FPR@95) from saved logits using multiple scoring methods:
   - **MSP** (Maximum Softmax Probability)
   - **Entropy**
   - **Max Logit**
   - **RbA** (Rejected-by-All) - region-level score for mask-based models

3. **mIoU Evaluation** (`eval_iou_eomt.py`): Evaluates semantic segmentation performance (mean Intersection-over-Union) on Cityscapes dataset. Reports per-class IoU and mean IoU across all classes.

4. **Temperature Search** (`find_optimal_temperature.py`): Finds optimal temperature scaling for MSP scoring to improve calibration.

For detailed usage instructions, see [`eval/usage.md`](eval/usage.md).

## `eomt/configs`

All experiments are configured via YAML files in `eomt/configs/dinov2/cityscapes/semantic/`. The configuration system allows easy experimentation with different training strategies.

### Configuration Structure

- **Base configs**: `eomt_base_640.yaml` - Standard EoMT training configuration
- **LogitNorm ablation**: `logit_norm_ablation/` - Stage A/B/C fine-tuning configurations
  - `stageA.yaml` - Task heads only
  - `stageA_class_head_only.yaml` - Classification head only
  - `stageB.yaml` - Task heads + query embeddings
  - `stageC.yaml` - Task heads + queries + LoRA adapters
- **Magnitude-aware training**: `magnitude_aware_training/` - Different magnitude loss variants
  - `mse_magnitude.yaml` - Fixed-target MSE loss
  - `hinge_magnitude.yaml` - Threshold-based hinge loss
  - `margin_magnitude.yaml` - Relative margin loss
  - `soft_margin_magnitude.yaml` - Smooth margin loss

### Key Configuration Parameters

- **LogitNorm**: `use_logit_norm`, `logit_norm_temp`, `logit_norm_mode`
- **Magnitude loss**: `use_magnitude_loss`, `magnitude_loss_type`, `magnitude_coefficient`, `magnitude_margin`, `magnitude_threshold`
- **Fine-tuning stage**: `finetune_stage` (A_head, A_class_only, B_queries, C_lora, full)
- **LoRA**: `lora.enabled`, `lora.rank`, `lora.alpha`

## Experiment Setup

Our experiments follow a progressive ablation protocol:

### Adaptation Stages

1. **Stage A**: Fine-tune task heads (classification + mask heads) only
   - Fastest and most computationally efficient
   - Best compromise between performance and efficiency

2. **Stage B**: Fine-tune task heads + query embeddings
   - Refines object-centric partitioning of the scene
   - Improves ability to maintain high uncertainty on OoD regions

3. **Stage C**: Fine-tune task heads + queries + LoRA adapters
   - Adds representation-level adaptation via Low-Rank Adapters
   - Mixed effects: improves some datasets but degrades others

### Training Protocol

- **Base model**: Cityscapes-pretrained EoMT (DINOv2 backbone)
- **Fine-tuning**: Progressive adaptation with increasing model flexibility
- **Loss components**: Cross-entropy (with LogitNorm) + magnitude regularization + mask losses
- **Evaluation**: Post-hoc anomaly scoring on raw (unnormalized) logits

## How to Run Fine-tuning

Fine-tuning is configuration-driven and implemented using **PyTorch Lightning**. The core training logic is in `eomt/training/lightning_module.py`, while experiment settings are controlled by YAML configuration files.

### Steps

1. **Choose a configuration** from `eomt/configs/dinov2/cityscapes/semantic/`

2. **Prepare required paths**:
   - Cityscapes dataset directory
   - Pretrained EoMT checkpoint

3. **Run fine-tuning**:

```bash
cd eomt

python main.py fit \
  -c "configs/dinov2/cityscapes/semantic/logit_norm_ablation/stageA.yaml" \
  --data.init_args.path "/path/to/cityscapes" \
  --model.init_args.ckpt_path "/path/to/eomt_cityscapes.bin"
```

## PyTorch Lightning Changes for Fine-tuning Steps

We modified the Lightning module (`training/lightning_module.py`) to support stage-based fine-tuning:

1. **Stage Control**: Added `finetune_stage` parameter to control which parameters are trainable:
   - `"A_head"`: Task heads only (classification + mask heads)
   - `"A_class_only"`: Classification head only (variant of Stage A)
   - `"B_queries"`: Query embeddings + optionally task heads (controlled by `train_heads_in_stage_b`)
   - `"C_lora"`: LoRA adapters + optionally task heads (controlled by `train_heads_in_stage_c`)

2. **Parameter Freezing**: The `apply_finetune_stage()` method selectively freezes parameters by setting `requires_grad=False` based on `finetune_stage`:
   - Stage A: Freezes encoder and query embeddings, unfreezes task heads
   - Stage B: Freezes encoder only, unfreezes query embeddings and optionally task heads
   - Stage C: Freezes encoder backbone, unfreezes LoRA adapters and optionally task heads
   - The `configure_optimizers()` method then creates optimizer groups based on `requires_grad` status

3. **LoRA Integration**: Added LoRA (Low-Rank Adaptation) support for parameter-efficient fine-tuning:
   - Configurable rank, alpha, and target modules
   - Applied to attention projection layers in the last Transformer blocks

4. **Learning Rate Scheduling**: Maintains layer-wise learning rate decay (LLRD) for fine-tuning with different learning rates for different components.

## Mask Classification Loss: LogitNorm & Magnitude Additions

The mask classification loss (`training/mask_classification_loss.py`) implements our key contributions:

### Logit Normalization (LogitNorm)

LogitNorm normalizes logits by their L2 norm and applies temperature scaling:

```
ẑ_i = (z_i / ||z_i||_2) / τ
```

**Key implementation details**:
- Applied **only during training** for loss computation
- **NOT applied during inference** to preserve magnitude information
- Hungarian matching behavior depends on `logit_norm_mode`:
  - `"ce"` mode: Matching uses raw logits, LogitNorm applied only in cross-entropy loss (recommended)
  - `"matching"` mode: LogitNorm applied only in matching, not in loss
  - `"both"` mode: LogitNorm applied in both matching and loss
- Temperature `τ` controls softmax sharpness (typically 0.5 for our experiments)

**Modes**:
- `"ce"`: Apply LogitNorm only in cross-entropy loss, matching uses raw logits (recommended)
- `"matching"`: Apply LogitNorm only in Hungarian matching
- `"both"`: Apply LogitNorm in both matching and loss computation

### Magnitude-Aware Regularization

To complement LogitNorm, we add magnitude regularization losses that structure the L2 norm of raw logits:

1. **MSE Loss** (`magnitude_loss_type: "mse"`): Forces unmatched queries toward a fixed target magnitude
2. **Hinge Loss** (`magnitude_loss_type: "hinge"`): Only penalizes unmatched queries exceeding threshold
3. **Margin Loss** (`magnitude_loss_type: "margin"`): Enforces relative separation between matched and unmatched query magnitudes
4. **Soft Margin Loss** (`magnitude_loss_type: "soft_margin"`): Smooth relaxation of margin loss

**Key findings**: Relative (margin) and threshold-based (hinge) constraints are more robust for OoD separation than fixed-target norm forcing (MSE).

### Overall Training Objective

```python
L = L_CE(ẑ, y) + λ_mag * L_mag(z) + L_mask
```

Where:
- `L_CE`: Cross-entropy on LogitNorm-normalized logits
- `L_mag`: Magnitude regularization on raw logits
- `L_mask`: Standard segmentation losses (dice, mask)

### Inference Behavior

**Critical**: At inference time:
- **No LogitNorm** is applied - raw logits are used
- Both semantic predictions and anomaly scores are computed from raw logits
- This preserves magnitude information needed for confidence-based OoD detection
