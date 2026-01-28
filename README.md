# Mask Architecture Anomaly Segmentation for Road Scenes [[Course Project](https://drive.google.com/file/d/19gQ2uhI8jPxWIdGewkaA2DqihUnQLVfB/view?usp=drive_link)]
This repository provides a starter-code setup for the Real-Time Anomaly Segmentation project of the Machine Learning Course. It consists of a code base for training and testing ERFNet on the Cityscapes dataset and performing anomaly segmentation.It also contains code referring to EoMT.
## Table of Contents
- [Motivation](#motivation)
- [Models](#models)
- [Post-hoc Anomaly Scoring](#post-hoc-anomaly-scoring-used-during-inference)
- [Calibration](#calibration)
- [Datasets](#datasets)
- [Metrics Reported](#metrics-reported)
- [Packages](#packages)
  - [ERFNet Evaluation](#eval-erfnet-evaluation-scripts)
  - [Temperature Scaling](#evaltemperature-temperature-scaling-utilities)
  - [EoMT Pipeline](#eomt-mask-based-model-pipeline)
  - [EoMT Evaluation](#eomteval-evaluation-and-logit-based-analysis)
  - [EoMT Fine-Tuning](#eomttraining-eomt-fine-tuning)

- [Fine-Tuning](#fine-tuning)


## Motivation
Semantic segmentation networks often fail in open-world driving scenarios because they are trained with a closed set of classes. For autonomous driving, it is important not only to segment known classes, but also to flag unknown regions as anomalies.


### Models
- **ERFNet** pixel-based semantic segmentation backbone
- **EoMT** mask-based architecture with transformer-style outputs

### Post-hoc anomaly scoring (used during inference)
- **MSP** (Maximum Softmax Probability)
- **Max Logit**
- **Entropy**
- **(EoMT only)** RbA-style rejection methods (mask-based)

### Calibration
Temperature scaling : fast evaluation supported by caching logits

## Datasets

Evaluated on road anomaly benchmarks such as:
- RoadAnomaly / RoadAnomaly21
- RoadObstacle21
- Fishyscapes (Lost & Found, Static)
- Cityscapes (for in-distribution segmentation quality via mIoU)

## Metrics Reported
- **mIoU** semantic segmentation performance on Cityscapes
- **AUPRC** anomaly detection precision-recall area
- **FPR@95TPR** false positive rate at 95% true positive rate


## Packages
For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing the an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.
### Root-level / main folders
- `eval/`  
  Scripts to evaluate ERFNet on Cityscapes and anomaly benchmarks and compute anomaly metrics.
- `eval/temperature/`  
  Utilities for temperature scaling experiments (including saving logits once, then testing many temperatures fast).
- `eomt/`  
  Full codebase for EoMT (mask-based model). Includes configs, dataset logic, training, and evaluation.
- `trained_models/`  
  Stores pretrained weights (e.g., `erfnet_pretrained.pth`).\

For instructions, please refer to the README in each folder:
### `eval/` (ERFNet evaluation scripts)

- `eval/evalAnomaly.py`  
  Main anomaly evaluation script for **pixel-based (ERFNet)** inference on anomaly datasets.  
  Computes anomaly maps using **MSP / Max Logit / Entropy**, then reports **AUPRC** and **FPR@95TPR** and appends results to `eval/results.txt`.

- `eval/eval_iou.py`  
  Computes **mIoU** of ERFNet on Cityscapes (val/train) and saves the final mIoU to `eval/results.txt`.

- `eval/eval_cityscapes_color.py`  
  Produces **colored semantic segmentation visualizations** for Cityscapes (saves into `eval/save_color/`).

- `eval/eval_cityscapes_server.py`  
  Exports Cityscapes predictions converted back to **labelIds format** for evaluation scripts / server submission (saves into `eval/save_results/`).

- `eval/eval_forwardTime.py`  
  Measures **forward pass time** (speed benchmark) for ERFNet at a given resolution.

- `eval/dataset.py`  
  Dataset loaders (Cityscapes + VOC-style helpers) used by evaluation scripts.

- `eval/transform.py`  
  Label transforms, relabeling, and colorization utilities used by Cityscapes evaluation / visualization.

- `eval/erfnet.py` and `eval/erfnet_nobn.py`  
  ERFNet architecture definitions (standard + variant without BN/dropout depending on the experiment).

- `eval/iouEval.py`  
  IoU accumulator and utilities used by `eval_iou.py`.

- `eval/convert_labelIds_to_trainIds.py`  
  Utility to convert Cityscapes **labelIds → trainIds** for correct training/evaluation formatting.

- `eval/results.txt`  
  Text log of evaluation outputs (AUPRC/FPR95 for anomaly benchmarks + mIoU).
### `eval/temperature/` (temperature scaling utilities)

- `eval/temperature/save_logits.py`  
  Runs ERFNet once on a dataset and **saves logits + GT masks** into a pickle file in `saved_logits/`.  
  This enables fast temperature sweeps without repeated forward passes.

- `eval/temperature/test_temperatures_fast.py`  
  Loads saved logits and evaluates multiple temperature values quickly for a selected anomaly score method.

- `eval/temperature/temperature_*.txt`  
  Result logs for temperature sweeps per dataset.

### `eomt/` (mask-based model pipeline)

Inside `eomt/` :
- `configs/`  
  YAML configs for training and ablation studies (logit normalization, magnitude-aware losses, etc.).
- `training/`  
  Training pipeline code for EoMT fine-tuning/experiments.
- `eval/`  
  EoMT evaluation scripts (logit extraction, dataset evaluation, temperature finding, etc.).

eomt/eval/ (evaluation and logit-based analysis)

### `eomt/eval/` (evaluation and logit-based analysis):
- `saved_logits/`  
  Directory for cached EoMT logits and intermediate outputs used for fast re-evaluation.

- `eval_iou_eomt.py`  
  Script for computing segmentation metrics (IoU / mIoU) for EoMT models.

- `evaluate_all_datasets.sh`  
  Shell script to run EoMT evaluation across all supported anomaly datasets.

- `evaluate_logits.py`  
  Evaluates anomaly detection performance (e.g., AuPRC, FPR@95) using saved logits.

- `extract_logits.py`  
  Runs EoMT inference and saves logits for downstream evaluation and temperature scaling.

- `find_optimal_temperature*.py`  
  Searches for the optimal temperature for temperature scaling using cached logits.

### `eomt/training/` (EoMT fine-tuning)

Contains the training-time fine-tuning pipeline for EoMT, including model adaptation, loss definitions, and optimization utilities.

- `lightning_module.py`\
PyTorch Lightning module defining the EoMT training and validation loops.

- `mask_classification_semantic.py`\
Semantic mask-classification logic used for Cityscapes fine-tuning.

- `mask_classification_panoptic.py`\
Panoptic-style mask-classification components for EoMT.

- `mask_classification_instance.py`\
Instance-level mask-classification utilities for object-centric training variants.

- `mask_classification_loss.py`\
Loss functions for mask-based classification and segmentation.

- `lora_utils.py`\
Utilities for LoRA-based parameter-efficient fine-tuning.

- `two_stage_warmup_poly_schedule.py`\
Two-stage warmup and polynomial learning-rate schedule.

- `visualization_callback.py`\
Training callback for real-time visualization of training/validation loss and mIoU.
## Fine-Tuning

In addition to post-hoc anomaly scoring, this repository includes **training-time fine-tuning of the End-to-End Open-set Mask Transformer (EoMT)** to improve semantic segmentation quality and anomaly detection in open-world road scenes.

EoMT is fine-tuned on the Cityscapes dataset using a **mask-based semantic segmentation objective**. The training pipeline supports **calibration-aware strategies**, including logit normalization and magnitude-aware training, which aim to reduce overconfident predictions and improve the reliability of uncertainty-based anomaly scores.

Fine-tuning is performed using **parameter-efficient adaptation (LoRA)** and a **staged training schedule**, enabling stable optimization while limiting the number of updated parameters. The fine-tuned models are compatible with the provided EoMT evaluation and logit-based anomaly analysis tools.
