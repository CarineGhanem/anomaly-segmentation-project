# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# IMPROVED MAGNITUDE-AWARE CALIBRATION
# - Margin-based magnitude loss (dataset-agnostic)
# - Hinge loss option (prevents over-aggressive regularization)
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional


class MaskClassificationLoss(nn.Module):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        use_logit_norm: bool = False,
        logit_norm_temp: float = 0.1,
        logit_norm_mode: str = "ce",
        # IMPROVED: Magnitude loss with multiple types
        use_magnitude_loss: bool = False,
        magnitude_coefficient: float = 0.5,
        magnitude_loss_type: str = "margin",  # "margin", "hinge", or "mse"
        magnitude_threshold: float = 2.0,  # For hinge loss (higher than before)
        magnitude_margin: float = 2.0,  # For margin loss
        # Adaptive temperature (kept for compatibility)
        use_adaptive_temperature: bool = False,
        adaptive_temp_range: tuple = (0.1, 1.0),
    ):
        super().__init__()
        
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.no_object_coefficient = no_object_coefficient
        
        # LogitNorm parameters
        self.use_logit_norm = use_logit_norm
        self.logit_norm_temp = logit_norm_temp
        
        # Validate logit_norm_mode
        allowed = {"none", "ce", "matching", "both"}
        if logit_norm_mode not in allowed:
            raise ValueError(f"logit_norm_mode must be one of {allowed}, got {logit_norm_mode}")
        self.logit_norm_mode = logit_norm_mode
        
        # IMPROVED: Magnitude loss with multiple types
        self.use_magnitude_loss = use_magnitude_loss
        self.magnitude_coefficient = magnitude_coefficient
        self.magnitude_loss_type = magnitude_loss_type
        self.magnitude_threshold = magnitude_threshold
        self.magnitude_margin = magnitude_margin
        
        # Validate magnitude_loss_type
        allowed_types = {"margin", "hinge", "mse"}
        if magnitude_loss_type not in allowed_types:
            raise ValueError(f"magnitude_loss_type must be one of {allowed_types}, got {magnitude_loss_type}")
        
        # Adaptive temperature
        self.use_adaptive_temperature = use_adaptive_temperature
        self.adaptive_temp_range = adaptive_temp_range
        
        # Class weights with no-object class
        empty_weight = torch.ones(num_labels + 1)
        empty_weight[-1] = no_object_coefficient
        self.register_buffer("empty_weight", empty_weight)
        
        # For logging/debugging
        self.register_buffer("_last_adaptive_temp", torch.tensor(logit_norm_temp))
        self.register_buffer("_last_matched_mag", torch.tensor(0.0))
        self.register_buffer("_last_unmatched_mag", torch.tensor(0.0))

    def apply_logit_norm(
        self,
        class_logits: torch.Tensor,
        *,
        context: str,
        temperature: Optional[float] = None,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Apply LogitNorm depending on mode + context."""
        if not self.use_logit_norm:
            return class_logits
        
        mode = self.logit_norm_mode
        if mode == "none":
            return class_logits
        elif mode == "both":
            apply = True
        elif mode == "ce":
            apply = (context == "ce")
        elif mode == "matching":
            apply = (context == "matching")
        else:
            apply = False
        
        if not apply:
            return class_logits
        
        # Use provided temperature or default
        temp = temperature if temperature is not None else self.logit_norm_temp
        
        # L2 normalize and scale by temperature
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        return class_logits / (norm + eps) / temp

    def compute_adaptive_temperature(
        self,
        class_logits: torch.Tensor,
        indices: List[tuple]
    ) -> float:
        """Compute adaptive temperature based on prediction confidence."""
        if not self.use_adaptive_temperature:
            return self.logit_norm_temp
        
        # Collect logits for matched predictions
        matched_logits = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                max_logits = class_logits[i, src_idx, :-1].max(dim=-1)[0]
                matched_logits.append(max_logits)
        
        if len(matched_logits) == 0:
            return self.adaptive_temp_range[1]
        
        # Compute mean confidence
        all_matched = torch.cat(matched_logits)
        mean_max_logit = all_matched.mean()
        
        # Map logit magnitude to temperature
        confidence_score = torch.sigmoid(mean_max_logit / 10.0)
        
        # Linear interpolation
        min_temp, max_temp = self.adaptive_temp_range
        adaptive_temp = max_temp + (min_temp - max_temp) * confidence_score
        
        # Store for logging
        self._last_adaptive_temp.copy_(adaptive_temp.detach())
        
        return adaptive_temp.item()
    


    def compute_magnitude_regularizer(self, class_logits, indices):
        """Dispatch to appropriate magnitude loss type."""
        if self.magnitude_loss_type == "margin":
            return self.compute_margin_magnitude_loss(class_logits, indices)
        elif self.magnitude_loss_type == "soft_margin":  # NEW
            return self.compute_soft_margin_magnitude_loss(class_logits, indices)
        elif self.magnitude_loss_type == "hinge":
            return self.compute_hinge_magnitude_loss(class_logits, indices)
        else:  # "mse"
            return self.compute_mse_magnitude_loss(class_logits, indices)

    def compute_soft_margin_magnitude_loss(self, class_logits, indices):
        """
        Soft margin: Always provides gradient via exponential penalty.
        """
        B, Q = class_logits.shape[:2]
        device = class_logits.device
        
        magnitudes = torch.norm(class_logits, p=2, dim=-1)
        
        matched_mags = []
        unmatched_mags = []
        
        for i in range(B):
            src_idx = indices[i][0]
            if len(src_idx) > 0:
                matched_mags.append(magnitudes[i, src_idx])
            
            unmatched_mask = torch.ones(Q, dtype=torch.bool, device=device)
            if len(src_idx) > 0:
                unmatched_mask[src_idx] = False
            unmatched_mags.append(magnitudes[i, unmatched_mask])
        
        if len(matched_mags) == 0:
            self._last_matched_mag.fill_(0.0)
            self._last_unmatched_mag.fill_(0.0)
            return torch.tensor(0.0, device=device)
        
        matched_mean = torch.cat(matched_mags).mean()
        unmatched_mean = torch.cat(unmatched_mags).mean()
        
        self._last_matched_mag.copy_(matched_mean.detach())
        self._last_unmatched_mag.copy_(unmatched_mean.detach())
        
        # Soft exponential penalty
        separation = matched_mean - unmatched_mean
        target_separation = self.magnitude_margin
        
        # exp(-(sep - target)/target) 
        # If sep = target: loss = exp(0) = 1.0
        # If sep > target: loss < 1.0 (decreasing)
        # If sep < target: loss > 1.0 (increasing)
        loss = torch.exp(-(separation - target_separation) / target_separation)
        
        return loss

    def compute_margin_magnitude_loss(
            self,
            class_logits: torch.Tensor,
            indices: List[tuple],
        ) -> torch.Tensor:
            """
            NEW: Margin-based magnitude loss (ArcFace-style).
            
            Theory:
            - Matched queries should have HIGH magnitude (confident predictions)
            - Unmatched queries should have LOW magnitude (uncertain/no-object)
            - Ensure: mean(matched) > mean(unmatched) + margin
            
            Advantages:
            - No absolute threshold (dataset-agnostic)
            - Relative comparison adapts to dataset's natural magnitude range
            - More robust than fixed thresholds
            
            Paper: "ArcFace: Additive Angular Margin Loss" (Deng et al., 2019)
            """
            B, Q = class_logits.shape[:2]
            device = class_logits.device
            
            # Compute L2 norm (magnitude) per query
            magnitudes = torch.norm(class_logits, p=2, dim=-1)  # [B, Q]
            
            matched_mags = []
            unmatched_mags = []
            
            for i in range(B):
                src_idx = indices[i][0]
                
                # Collect matched magnitudes
                if len(src_idx) > 0:
                    matched_mags.append(magnitudes[i, src_idx])
                
                # Collect unmatched magnitudes
                unmatched_mask = torch.ones(Q, dtype=torch.bool, device=device)
                if len(src_idx) > 0:
                    unmatched_mask[src_idx] = False
                unmatched_mags.append(magnitudes[i, unmatched_mask])
            
            # Need at least some matched predictions
            if len(matched_mags) == 0:
                return torch.tensor(0.0, device=device)
            
            # Compute mean magnitudes
            matched_mean = torch.cat(matched_mags).mean()
            unmatched_mean = torch.cat(unmatched_mags).mean()
            
            # Store for logging
            self._last_matched_mag.copy_(matched_mean.detach())
            self._last_unmatched_mag.copy_(unmatched_mean.detach())
            
            # Margin loss: ensure matched > unmatched + margin
            # If difference is already > margin, loss = 0
            # If difference is < margin, penalize
            loss = F.relu(self.magnitude_margin - (matched_mean - unmatched_mean))
            
            return loss

    def compute_hinge_magnitude_loss(
        self,
        class_logits: torch.Tensor,
        indices: List[tuple],
    ) -> torch.Tensor:
        """
        NEW: Hinge-based magnitude loss.
        
        Theory:
        - Only penalize unmatched queries if magnitude > threshold
        - Allows magnitudes to be anywhere below threshold
        - More flexible than MSE (doesn't force exact value)
        
        Advantages:
        - Prevents over-aggressive pushing down of magnitudes
        - More forgiving for borderline cases
        - Reduces risk of over-flagging (RoadObstacle21 issue)
        
        Paper: "Large Margin Deep Networks" (Elsayed et al., 2018)
        """
        B, Q = class_logits.shape[:2]
        device = class_logits.device
        
        # Compute magnitudes
        magnitudes = torch.norm(class_logits, p=2, dim=-1)  # [B, Q]
        
        # Create mask for unmatched queries
        unmatched_mask = torch.ones(B, Q, dtype=torch.bool, device=device)
        for i, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                unmatched_mask[i, src_idx] = False
        
        # Get magnitudes for unmatched queries
        unmatched_mags = magnitudes[unmatched_mask]
        
        if len(unmatched_mags) == 0:
            return torch.tensor(0.0, device=device)
        
        # Store for logging
        self._last_unmatched_mag.copy_(unmatched_mags.mean().detach())
        
        # Hinge loss: max(0, magnitude - threshold)
        # Only penalize if magnitude exceeds threshold
        loss = F.relu(unmatched_mags - self.magnitude_threshold).mean()
        
        return loss

    def compute_mse_magnitude_loss(
        self,
        class_logits: torch.Tensor,
        indices: List[tuple],
    ) -> torch.Tensor:
        """
        ORIGINAL: MSE-based magnitude loss.
        Kept for comparison with new methods.
        """
        B, Q = class_logits.shape[:2]
        device = class_logits.device
        
        magnitudes = torch.norm(class_logits, p=2, dim=-1)
        
        unmatched_mask = torch.ones(B, Q, dtype=torch.bool, device=device)
        for i, (src_idx, _) in enumerate(indices):
            if len(src_idx) > 0:
                unmatched_mask[i, src_idx] = False
        
        unmatched_magnitudes = magnitudes[unmatched_mask]
        
        if len(unmatched_magnitudes) == 0:
            return torch.tensor(0.0, device=device)
        
        # Store for logging
        self._last_unmatched_mag.copy_(unmatched_magnitudes.mean().detach())
        
        # MSE: force magnitudes toward threshold
        target_magnitude = self.magnitude_threshold
        loss = F.mse_loss(
            unmatched_magnitudes,
            torch.full_like(unmatched_magnitudes, target_magnitude)
        )
        
        return loss

    def point_sample(self, input, point_coords, **kwargs):
        """Sample features at point coordinates."""
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2)
        
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
        
        if add_dim:
            output = output.squeeze(3)
        
        return output

    def get_uncertain_point_coords(self, mask_logits, num_points):
        """Sample points with highest uncertainty."""
        num_boxes = mask_logits.shape[0]
        num_points = min(mask_logits.shape[2] * mask_logits.shape[3], num_points)
        
        uncertainty = -(mask_logits.abs() - 0.5).abs()
        _, idx = uncertainty.view(num_boxes, -1).topk(num_points, dim=1)
        
        h, w = mask_logits.shape[2:]
        point_coords = torch.zeros(
            num_boxes, num_points, 2, 
            dtype=torch.float, 
            device=mask_logits.device
        )
        point_coords[:, :, 0] = idx % w
        point_coords[:, :, 1] = idx // w
        point_coords[:, :, 0] = point_coords[:, :, 0] / w
        point_coords[:, :, 1] = point_coords[:, :, 1] / h
        
        return point_coords

    def sample_points(self, mask_logits, gt_masks):
        """Sample points for mask loss computation."""
        with torch.no_grad():
            num_oversample = int(self.num_points * self.oversample_ratio)
            num_uncertain = int(self.importance_sample_ratio * num_oversample)
            point_coords = self.get_uncertain_point_coords(mask_logits, num_uncertain)
            
            num_random = num_oversample - num_uncertain
            if num_random > 0:
                B, _, H, W = mask_logits.shape
                random_coords = torch.rand(B, num_random, 2, device=mask_logits.device)
                point_coords = torch.cat([point_coords, random_coords], dim=1)
            
            idx = torch.randperm(point_coords.shape[1], device=point_coords.device)
            idx = idx[:self.num_points]
            point_coords = point_coords[:, idx]
        
        mask_logits_sampled = self.point_sample(
            mask_logits, point_coords, align_corners=False
        )
        gt_masks_sampled = self.point_sample(
            gt_masks.float(), point_coords, align_corners=False
        )
        
        return mask_logits_sampled, gt_masks_sampled

    def dice_loss(self, inputs, targets):
        """Compute dice loss."""
        inputs = inputs.sigmoid().flatten(1)
        targets = targets.flatten(1)
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        
        return loss

    def sigmoid_ce_loss(self, inputs, targets):
        """Compute sigmoid cross entropy loss."""
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        return loss.mean(1)
    
    def pairwise_sigmoid_ce_loss(self, inputs, targets):
        """Pairwise sigmoid CE for Hungarian matching."""
        Q, P = inputs.shape
        T = targets.shape[0]

        out = inputs[:, None, :].expand(Q, T, P)
        tgt = targets[None, :, :].float().expand(Q, T, P)

        loss = F.binary_cross_entropy_with_logits(out, tgt, reduction="none")
        return loss.mean(-1)

    def pairwise_dice_loss(self, inputs, targets):
        """Pairwise dice loss for Hungarian matching."""
        Q, P = inputs.shape
        T = targets.shape[0]

        out = inputs[:, None, :].sigmoid().expand(Q, T, P)
        tgt = targets[None, :, :].float().expand(Q, T, P)

        numerator = 2 * (out * tgt).sum(-1)
        denominator = out.sum(-1) + tgt.sum(-1)
        return 1 - (numerator + 1.0) / (denominator + 1.0)

    def hungarian_matching(self, mask_logits, class_logits, gt_masks, gt_labels):
        """Perform Hungarian matching between predictions and ground truth."""
        B, Q = class_logits.shape[:2]
        
        # Apply LogitNorm for matching if mode requires it
        class_logits_for_matching = self.apply_logit_norm(
            class_logits,
            context="matching"
        )
        
        # Flatten batch dimension
        class_logits_flat = class_logits_for_matching.flatten(0, 1)
        mask_logits_flat = mask_logits.detach().flatten(0, 1)
        
        indices = []
        
        for i in range(B):
            tgt_ids = gt_labels[i]
            tgt_masks = gt_masks[i]
            
            if len(tgt_ids) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=mask_logits.device),
                    torch.tensor([], dtype=torch.long, device=mask_logits.device)
                ))
                continue
            
            out_prob = class_logits_flat[i*Q:(i+1)*Q].softmax(-1)
            out_mask = mask_logits_flat[i*Q:(i+1)*Q]
            
            # Classification cost
            cost_class = -out_prob[:, tgt_ids]
            
            # Sample points for cost computation
            with torch.no_grad():
                point_coords = torch.rand(
                    1, min(self.num_points, tgt_masks.shape[1] * tgt_masks.shape[2]), 2,
                    device=mask_logits.device
                )
                
                tgt_masks_sample = self.point_sample(
                    tgt_masks.unsqueeze(1).float(),
                    point_coords.repeat(len(tgt_masks), 1, 1),
                    align_corners=False
                ).squeeze(1)
                
                out_mask_sample = self.point_sample(
                    out_mask.unsqueeze(1),
                    point_coords.repeat(Q, 1, 1),
                    align_corners=False
                ).squeeze(1)
            
            cost_mask = self.pairwise_sigmoid_ce_loss(out_mask_sample, tgt_masks_sample)
            cost_dice = self.pairwise_dice_loss(out_mask_sample, tgt_masks_sample)
            
            # Total cost
            cost = (
                self.class_coefficient * cost_class +
                self.mask_coefficient * cost_mask +
                self.dice_coefficient * cost_dice
            )
            
            # Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.long, device=mask_logits.device),
                torch.as_tensor(tgt_idx, dtype=torch.long, device=mask_logits.device)
            ))
        
        return indices

    def forward(self, masks_queries_logits, class_queries_logits, targets):
        """Compute losses with improved magnitude-aware calibration."""
        B, Q = class_queries_logits.shape[:2]
        device = class_queries_logits.device
        
        # Prepare ground truth
        gt_masks = [t["masks"].float() for t in targets]
        gt_labels = [t["labels"] for t in targets]
        
        # Resize mask logits if needed
        if masks_queries_logits.shape[-2:] != gt_masks[0].shape[-2:]:
            masks_queries_logits = F.interpolate(
                masks_queries_logits,
                size=gt_masks[0].shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        
        if not masks_queries_logits.requires_grad:
            masks_queries_logits = masks_queries_logits.detach()
        
        # Hungarian matching
        indices = self.hungarian_matching(
            masks_queries_logits,
            class_queries_logits,
            gt_masks,
            gt_labels
        )
        
        # Compute adaptive temperature if enabled
        if self.use_adaptive_temperature:
            adaptive_temp = self.compute_adaptive_temperature(
                class_queries_logits,
                indices
            )
        else:
            adaptive_temp = None
        
        # Initialize losses
        losses = {}
        
        # Classification loss with LogitNorm
        class_logits_norm = self.apply_logit_norm(
            class_queries_logits,
            context="ce",
            temperature=adaptive_temp
        )
        
        # Prepare target classes
        target_classes = torch.full(
            (B, Q), class_queries_logits.shape[-1] - 1,
            dtype=torch.long, device=device
        )
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_classes[i, src_idx] = gt_labels[i][tgt_idx]
        
        # Compute cross entropy
        loss_ce = F.cross_entropy(
            class_logits_norm.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean"
        )
        losses["loss_ce"] = loss_ce
        
        # Mask losses
        num_masks = sum(len(tgt_idx) for _, tgt_idx in indices)
        
        if num_masks > 0:
            src_masks = []
            tgt_masks = []
            
            for i, (src_idx, tgt_idx) in enumerate(indices):
                if len(tgt_idx) > 0:
                    src_masks.append(masks_queries_logits[i, src_idx])
                    tgt_masks.append(gt_masks[i][tgt_idx])
            
            src_masks = torch.cat(src_masks)
            tgt_masks = torch.cat(tgt_masks)
            
            src_masks_sample, tgt_masks_sample = self.sample_points(
                src_masks.unsqueeze(1), tgt_masks.unsqueeze(1)
            )
            src_masks_sample = src_masks_sample.squeeze(1)
            tgt_masks_sample = tgt_masks_sample.squeeze(1)
            
            losses["loss_mask"] = self.sigmoid_ce_loss(
                src_masks_sample, tgt_masks_sample
            ).mean()
            
            losses["loss_dice"] = self.dice_loss(
                src_masks_sample, tgt_masks_sample
            ).mean()
        else:
            losses["loss_mask"] = masks_queries_logits.sum() * 0.0
            losses["loss_dice"] = masks_queries_logits.sum() * 0.0
        
        # IMPROVED: Magnitude regularization
        if self.use_magnitude_loss:
            loss_magnitude = self.compute_magnitude_regularizer(
                class_queries_logits,  # Use RAW logits
                indices
            )
            losses["loss_magnitude"] = loss_magnitude
        
        return losses

    # In mask_classification_loss.py, update loss_total():

def loss_total(self, losses_dict: Dict[str, torch.Tensor], log_fn) -> torch.Tensor:
    """Compute total weighted loss."""
    total_loss = 0.0
    
    for key, value in losses_dict.items():
        if "loss_ce" in key:
            weighted_loss = value * self.class_coefficient
        elif "loss_mask" in key:
            weighted_loss = value * self.mask_coefficient
        elif "loss_dice" in key:
            weighted_loss = value * self.dice_coefficient
        elif "loss_magnitude" in key:
            weighted_loss = value * self.magnitude_coefficient
        else:
            weighted_loss = value
        
        total_loss += weighted_loss
        log_fn(key, value, on_step=True, prog_bar=True)
    
    # ADD THIS BLOCK - Log magnitude stats
    if self.use_magnitude_loss and self.magnitude_loss_type == "margin":
        log_fn("matched_mag", self._last_matched_mag, 
               on_step=True, prog_bar=True)
        log_fn("unmatched_mag", self._last_unmatched_mag, 
               on_step=True, prog_bar=True)
        mag_diff = self._last_matched_mag - self._last_unmatched_mag
        log_fn("mag_diff", mag_diff, on_step=True, prog_bar=True)
    
    log_fn("loss_total", total_loss, on_step=True, prog_bar=True)
    
    return total_loss