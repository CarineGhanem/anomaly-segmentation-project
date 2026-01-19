# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict


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
        
        # Class weights with no-object class
        empty_weight = torch.ones(num_labels + 1)
        empty_weight[-1] = no_object_coefficient
        self.register_buffer("empty_weight", empty_weight)

    def apply_logit_norm(self, class_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Apply LogitNorm to class logits.
        
        Args:
            class_logits: [B, Q, C+1] raw class logits
            eps: small constant for numerical stability
            
        Returns:
            normalized_logits: [B, Q, C+1] normalized logits scaled by temperature
        """
        if not self.use_logit_norm:
            return class_logits
        
        # Compute L2 norm across class dimension (including no-object class)
        norm = torch.norm(class_logits, p=2, dim=-1, keepdim=True)
        
        # Normalize and scale by temperature
        normalized_logits = class_logits / (norm + eps) / self.logit_norm_temp
        
        return normalized_logits

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
        """Sample points with highest uncertainty (close to 0.5 after sigmoid)."""
        num_boxes = mask_logits.shape[0]
        num_points = min(mask_logits.shape[2] * mask_logits.shape[3], num_points)
        
        # Calculate uncertainty as distance from 0.5
        uncertainty = -(mask_logits.abs() - 0.5).abs()
        
        # Get top-k uncertain points
        _, idx = uncertainty.view(num_boxes, -1).topk(num_points, dim=1)
        
        # Convert to point coordinates
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
            # Oversample points
            num_oversample = int(self.num_points * self.oversample_ratio)
            
            # Sample uncertain points
            num_uncertain = int(self.importance_sample_ratio * num_oversample)
            point_coords = self.get_uncertain_point_coords(mask_logits, num_uncertain)
            
            # Sample remaining points randomly
            num_random = num_oversample - num_uncertain
            if num_random > 0:
                B, _, H, W = mask_logits.shape
                random_coords = torch.rand(B, num_random, 2, device=mask_logits.device)
                point_coords = torch.cat([point_coords, random_coords], dim=1)
            
            # Get final subset
            idx = torch.randperm(point_coords.shape[1], device=point_coords.device)
            idx = idx[:self.num_points]
            point_coords = point_coords[:, idx]
        
        # Sample mask logits and ground truth
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
        """
        inputs:  [Q, P] logits
        targets: [T, P] {0,1}
        returns: [Q, T]
        """
        Q, P = inputs.shape
        T = targets.shape[0]

        out = inputs[:, None, :].expand(Q, T, P)                 # [Q,T,P]
        tgt = targets[None, :, :].float().expand(Q, T, P)        # [Q,T,P]

        loss = F.binary_cross_entropy_with_logits(out, tgt, reduction="none")  # [Q,T,P]
        return loss.mean(-1)  # [Q,T]



    def pairwise_dice_loss(self, inputs, targets):
        """
        inputs:  [Q, P] logits
        targets: [T, P] {0,1}
        returns: [Q, T]
        """
        Q, P = inputs.shape
        T = targets.shape[0]

        out = inputs[:, None, :].sigmoid().expand(Q, T, P)       # [Q,T,P]
        tgt = targets[None, :, :].float().expand(Q, T, P)        # [Q,T,P]

        numerator = 2 * (out * tgt).sum(-1)                      # [Q,T]
        denominator = out.sum(-1) + tgt.sum(-1)                  # [Q,T]
        return 1 - (numerator + 1.0) / (denominator + 1.0)


    def hungarian_matching(self, mask_logits, class_logits, gt_masks, gt_labels):
        """Perform Hungarian matching between predictions and ground truth."""
        B, Q = class_logits.shape[:2]
        
        # Apply LogitNorm before matching
        class_logits_norm = self.apply_logit_norm(class_logits)
        
        # Flatten batch dimension for matching
        class_logits_flat = class_logits_norm.flatten(0, 1)  # [B*Q, C+1]
       
        mask_logits_flat = mask_logits.detach().flatten(0, 1)  # [B*Q, H, W]
        
        indices = []
        
        for i in range(B):
            # Get ground truth for this sample
            tgt_ids = gt_labels[i]
            tgt_masks = gt_masks[i]
            
            if len(tgt_ids) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=mask_logits.device),
                    torch.tensor([], dtype=torch.long, device=mask_logits.device)
                ))
                continue
            
            # Get predictions for this sample
            out_prob = class_logits_flat[i*Q:(i+1)*Q].softmax(-1)  # [Q, C+1]
            out_mask = mask_logits_flat[i*Q:(i+1)*Q]  # [Q, H, W]
            
            # Classification cost
            cost_class = -out_prob[:, tgt_ids]  # [Q, num_gt]
            
            # Mask costs (using sampled points for efficiency)
            
            with torch.no_grad():
                # Sample points for cost computation
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
            
            cost_mask = self.pairwise_sigmoid_ce_loss(out_mask_sample, tgt_masks_sample)  # [Q,T]
            cost_dice = self.pairwise_dice_loss(out_mask_sample, tgt_masks_sample)        # [Q,T]
            
            # Total cost
            cost = (
                self.class_coefficient * cost_class +
                self.mask_coefficient * cost_mask +
                self.dice_coefficient * cost_dice
            )
            
            # Hungarian algorithm (detach before converting to numpy)
            src_idx, tgt_idx = linear_sum_assignment(cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.long, device=mask_logits.device),
                torch.as_tensor(tgt_idx, dtype=torch.long, device=mask_logits.device)
            ))
        
        return indices

    def forward(self, masks_queries_logits, class_queries_logits, targets):
        """
        Compute losses.
        
        Args:
            masks_queries_logits: [B, Q, H, W]
            class_queries_logits: [B, Q, C+1]
            targets: list of dicts with 'masks' and 'labels'
        """
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
            masks_queries_logits, class_queries_logits, gt_masks, gt_labels
        )
        
        # Initialize losses
        losses = {}
        
        # Classification loss (with LogitNorm)
        class_logits_norm = self.apply_logit_norm(class_queries_logits)
        
        # Prepare target classes
        target_classes = torch.full(
            (B, Q), class_queries_logits.shape[-1] - 1,
            dtype=torch.long, device=device
        )
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                target_classes[i, src_idx] = gt_labels[i][tgt_idx]
        
        # Compute cross entropy with LogitNorm logits
        loss_ce = F.cross_entropy(
            class_logits_norm.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            reduction="mean"
        )
        losses["loss_ce"] = loss_ce
        
        # Mask losses (only for matched queries)
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
            
            # Sample points
            src_masks_sample, tgt_masks_sample = self.sample_points(
                src_masks.unsqueeze(1), tgt_masks.unsqueeze(1)
            )
            src_masks_sample = src_masks_sample.squeeze(1)
            tgt_masks_sample = tgt_masks_sample.squeeze(1)
            
            # Compute mask losses
            losses["loss_mask"] = self.sigmoid_ce_loss(
                src_masks_sample, tgt_masks_sample
            ).mean()
            
            losses["loss_dice"] = self.dice_loss(
                src_masks_sample, tgt_masks_sample
            ).mean()
        else:
            losses["loss_mask"] = masks_queries_logits.sum() * 0.0
            losses["loss_dice"] = masks_queries_logits.sum() * 0.0
        
        return losses

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
            else:
                weighted_loss = value
            
            total_loss += weighted_loss
            log_fn(key, value, on_step=True, prog_bar=True)
        
        log_fn("loss_total", total_loss, on_step=True, prog_bar=True)
        
        return total_loss