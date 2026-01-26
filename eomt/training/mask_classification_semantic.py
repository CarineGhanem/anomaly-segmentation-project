# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
import logging

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # LogitNorm parameters
        use_logit_norm: bool = True,
        logit_norm_temp: float = 0.1,
        logit_norm_mode: str = "ce",
        # NEW: Magnitude-aware calibration parameters
        use_magnitude_loss: bool = False,
        magnitude_coefficient: float = 0.5,
        magnitude_threshold: float = 1.0,
        use_adaptive_temperature: bool = False,
        adaptive_temp_range: tuple = (0.1, 1.0),
        magnitude_loss_type: str = "margin",  # NEW
        magnitude_margin: float = 2.0,  # NEW
        # Fine-tuning stage parameters
        finetune_stage: str = "full",
        lora: Optional[dict] = None,
        train_heads_in_stage_b: bool = True,
        train_heads_in_stage_c: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            finetune_stage=finetune_stage,
            lora=lora,
            train_heads_in_stage_b=train_heads_in_stage_b,
            train_heads_in_stage_c=train_heads_in_stage_c,
            use_logit_norm=use_logit_norm,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        # Initialize criterion with LogitNorm AND magnitude-aware calibration support
        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            use_logit_norm=use_logit_norm,
            logit_norm_temp=logit_norm_temp,
            logit_norm_mode=logit_norm_mode,
            # NEW: Magnitude-aware calibration
            use_magnitude_loss=use_magnitude_loss,
            magnitude_coefficient=magnitude_coefficient,
            magnitude_loss_type=magnitude_loss_type,
            magnitude_margin=magnitude_margin,
            magnitude_threshold=magnitude_threshold,
            use_adaptive_temperature=use_adaptive_temperature,
            adaptive_temp_range=adaptive_temp_range,
            
        )
        
        # Store settings for logging
        self.use_logit_norm = use_logit_norm
        self.logit_norm_temp = logit_norm_temp
        self.logit_norm_mode = logit_norm_mode
        self.use_magnitude_loss = use_magnitude_loss
        self.magnitude_coefficient = magnitude_coefficient
        self.use_adaptive_temperature = use_adaptive_temperature
        
        # Log configuration
        logging.info(f"\n{'='*60}")
        logging.info(f"Calibration Configuration:")
        logging.info(f"  LogitNorm: {use_logit_norm}")
        if use_logit_norm:
            logging.info(f"    Temperature: {logit_norm_temp}")
            logging.info(f"    Mode: {logit_norm_mode}")
            logging.info(f"    Adaptive: {use_adaptive_temperature}")
            if use_adaptive_temperature:
                logging.info(f"    Temp Range: {adaptive_temp_range}")
        logging.info(f"  Magnitude Loss: {use_magnitude_loss}")
        if use_magnitude_loss:
            logging.info(f"    Coefficient: {magnitude_coefficient}")
            logging.info(f"    Threshold: {magnitude_threshold}")
        logging.info(f"{'='*60}\n")

        # Initialize metrics
        num_metric_blocks = self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        self.init_metrics_semantic(ignore_idx, num_metric_blocks)
        
    def training_step(self, batch, batch_idx):
        """Training step with safe magnitude logging."""
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses
            
            # Log magnitude stats ONLY for final block
            if i == len(mask_logits_per_block) - 1:
                # Check if magnitude loss is enabled and it's margin type
                if (self.use_magnitude_loss and 
                    self.criterion.magnitude_loss_type == "margin"):
                    
                    # Safely access magnitude stats
                    try:
                        matched_mag = self.criterion._last_matched_mag.item()
                        unmatched_mag = self.criterion._last_unmatched_mag.item()
                        mag_diff = matched_mag - unmatched_mag
                        
                        # Log to progress bar
                        self.log("matched_mag", matched_mag, 
                                on_step=True, prog_bar=True, logger=True)
                        self.log("unmatched_mag", unmatched_mag,
                                on_step=True, prog_bar=True, logger=True)
                        self.log("mag_diff", mag_diff, 
                                on_step=True, prog_bar=True, logger=True)
                    except Exception as e:
                        # Silently skip if anything goes wrong
                        pass

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        """
        Evaluation step for semantic segmentation.
        
        CRITICAL: LogitNorm is NOT applied during inference.
        LogitNorm is only used during training for loss computation.
        This preserves magnitude information for OOD detection at test time.
        """
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

            # CRITICAL: Do NOT apply LogitNorm during inference
            # This preserves magnitude information needed for OOD detection
            crop_logits = self.to_per_pixel_logits_semantic(
                mask_logits,
                class_logits,
                use_logit_norm=False,  # Never use LogitNorm during inference
            )
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        """Called at the end of validation."""
        self._on_eval_end_semantic("val")