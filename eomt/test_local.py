#!/usr/bin/env python3
"""
Test script for validating the 3-stage fine-tuning setup locally.
Tests parameter freezing, LogitNorm, and loss computation without heavy training.
"""

import torch
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_stage_freezing(stage_name: str):
    """Test that the correct parameters are frozen/unfrozen for a given stage."""
    from omegaconf import OmegaConf
    from training.mask_classification_semantic import MaskClassificationSemantic
    from models.eomt import EoMT
    from models.vit import ViT
    
    logger.info(f"\n{'='*70}")
    logger.info(f"TESTING STAGE: {stage_name}")
    logger.info('='*70)
    
    # Create minimal network manually to avoid config complexity
    logger.info("Creating model...")
    try:
        # Create network directly
        encoder = ViT(
            backbone_name="vit_base_patch14_reg4_dinov2",
            img_size=[256, 512],
            ckpt_path=None,
        )
        
        network = EoMT(
            encoder=encoder,
            num_classes=19,
            num_q=50,
            num_blocks=2,
            masked_attn_enabled=True,  
        )

        
        # Create model with the network
        model = MaskClassificationSemantic(
            network=network,
            img_size=[256, 512],
            num_classes=19,
            attn_mask_annealing_enabled=False,
            attn_mask_annealing_start_steps=None,
            attn_mask_annealing_end_steps=None,
            use_logit_norm=True,
            logit_norm_temp=0.1,
            finetune_stage=stage_name,
            train_heads_in_stage_b=True,
            train_heads_in_stage_c=True,
            lora={"enabled": False},
            ignore_idx=255,
            lr=0.001,
            llrd=0.8,
            llrd_l2_enabled=True,
            lr_mult=1.0,
            weight_decay=0.01,
            poly_power=0.9,
            warmup_steps=[5, 10],
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            no_object_coefficient=0.1,
            mask_thresh=0.8,
            overlap_thresh=0.8,
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Manually trigger stage application (normally happens in on_fit_start)
    model.apply_finetune_stage()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100
    
    logger.info(f"\nPARAMETER STATISTICS:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"  Frozen: {frozen_params:,} ({100-trainable_pct:.2f}%)")
    
    # List trainable parameters
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    logger.info(f"\nTRAINABLE PARAMETERS ({len(trainable_names)} tensors):")
    for name in trainable_names[:10]:  # Show first 10
        logger.info(f"  ✓ {name}")
    if len(trainable_names) > 10:
        logger.info(f"  ... and {len(trainable_names) - 10} more")
    
    # Verify expected behavior
    success = True
    
    if stage_name == "A_head":
        # Should only have class_head and mask_head trainable
        head_params = [n for n in trainable_names if 'class_head' in n or 'mask_head' in n or 'class_predictor' in n or 'mask_predictor' in n]
        non_head_params = [n for n in trainable_names if n not in head_params]
        
        if non_head_params:
            logger.warning(f"⚠ Stage A should only train heads, but found: {non_head_params[:5]}")
            success = False
        else:
            logger.info("✓ Stage A correctly freezes everything except heads")
        
        if trainable_pct > 10:
            logger.warning(f"⚠ Stage A has {trainable_pct:.1f}% trainable (expected < 10%)")
            success = False
    
    elif stage_name == "B_queries":
        # Should have queries + optionally heads
        query_params = [n for n in trainable_names if 'q.weight' in n or 'query' in n]
        if not query_params:
            logger.warning("⚠ Stage B should train queries, but none found")
            success = False
        else:
            logger.info(f"✓ Stage B correctly unfroze {len(query_params)} query parameters")
        
        if trainable_pct > 15:
            logger.warning(f"⚠ Stage B has {trainable_pct:.1f}% trainable (expected < 15%)")
            success = False
    
    elif stage_name == "C_lora":
        # Should have LoRA params
        lora_params = [n for n in trainable_names if 'lora_' in n]
        if not lora_params:
            logger.warning("⚠ Stage C should have LoRA params, but none found")
            logger.info("   Make sure LoRA is enabled in config and target_modules match your network")
            success = False
        else:
            logger.info(f"✓ Stage C correctly injected {len(lora_params)} LoRA parameters")
    
    elif stage_name == "full":
        if trainable_pct < 90:
            logger.warning(f"⚠ Full training should have >90% trainable, got {trainable_pct:.1f}%")
            success = False
        else:
            logger.info("✓ Full training correctly unfroze all parameters")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Stage {stage_name}: {'✓ PASSED' if success else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return success


def test_logit_norm():
    """Test LogitNorm computation."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING LOGITNORM")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Create loss with LogitNorm enabled
    criterion = MaskClassificationLoss(
        num_points=1024,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=19,
        no_object_coefficient=0.1,
        use_logit_norm=True,
        logit_norm_temp=0.1,
        logit_norm_mode="both",
    )
    
    # Create dummy logits
    B, Q, C = 2, 50, 19
    class_logits = torch.randn(B, Q, C + 1)  # +1 for no-object class
    
    logger.info(f"\nInput class logits shape: {class_logits.shape}")
    logger.info(f"Input class logits norm (per query): {class_logits.norm(dim=-1).mean().item():.4f}")
    
    # Apply LogitNorm
    normalized = criterion.apply_logit_norm(class_logits, context="both")
    
    logger.info(f"\nNormalized logits shape: {normalized.shape}")
    logger.info(f"Normalized logits norm (per query): {normalized.norm(dim=-1).mean().item():.4f}")
    logger.info(f"Expected norm: ~{1.0 / criterion.logit_norm_temp:.4f}")
    
    # Check that norm is approximately 1/temperature
    expected_norm = 1.0 / criterion.logit_norm_temp
    actual_norm = normalized.norm(dim=-1).mean().item()
    norm_diff = abs(actual_norm - expected_norm)
    
    success = norm_diff < 0.5  # Allow some tolerance
    
    if success:
        logger.info(f"✓ LogitNorm working correctly (norm diff: {norm_diff:.4f})")
    else:
        logger.warning(f"⚠ LogitNorm norm mismatch (diff: {norm_diff:.4f})")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"LogitNorm: {'✓ PASSED' if success else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return success


def test_loss_computation():
    """Test that loss computation works with LogitNorm."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING LOSS COMPUTATION")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Create loss
    criterion = MaskClassificationLoss(
        num_points=512,  # Small for speed
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=19,
        no_object_coefficient=0.1,
        use_logit_norm=True,
        logit_norm_temp=0.1,
    )
    
    # Create dummy data - FIXED: Match number of queries
    B, Q, H, W, C = 1, 20, 64, 64, 19
    num_objects = 5
    
    mask_logits = torch.randn(B, Q, H, W)
    class_logits = torch.randn(B, Q, C + 1)
    
    # Create dummy targets
    targets = [{
        'masks': torch.randint(0, 2, (num_objects, H, W)).bool(),  # num_objects masks
        'labels': torch.randint(0, C, (num_objects,)),
    }]
    
    logger.info(f"\nInput shapes:")
    logger.info(f"  Mask logits: {mask_logits.shape}")
    logger.info(f"  Class logits: {class_logits.shape}")
    logger.info(f"  Targets: {len(targets[0]['masks'])} objects")
    
    try:
        # Compute losses
        losses = criterion(mask_logits, class_logits, targets)
        
        logger.info(f"\nComputed losses:")
        for key, value in losses.items():
            logger.info(f"  {key}: {value.item():.4f}")
        
        # Check that losses are finite
        all_finite = all(torch.isfinite(v) for v in losses.values())
        
        if all_finite:
            logger.info("✓ All losses are finite")
            success = True
        else:
            logger.warning("⚠ Some losses are NaN or Inf")
            success = False
        
    except Exception as e:
        logger.error(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Loss computation: {'✓ PASSED' if success else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return success


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("STARTING LOCAL VALIDATION TESTS")
    logger.info("="*70 + "\n")
    
    results = {}
    
    # Test each stage
    for stage in ["A_head", "B_queries", "full"]:
        results[f"stage_{stage}"] = test_stage_freezing(stage)
    
    # Test Stage C separately (requires LoRA config)
    logger.info("NOTE: Stage C (LoRA) test skipped by default.")
    logger.info("      To test Stage C, set lora.enabled=True in trial_local.yaml")
    logger.info("      and uncomment the test below.\n")
    # results["stage_C_lora"] = test_stage_freezing("C_lora")
    
    # Test LogitNorm
    results["logit_norm"] = test_logit_norm()
    
    # Test loss computation
    results["loss_computation"] = test_loss_computation()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:25s}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("ALL TESTS PASSED ✓")
        logger.info("\nYou can now run training with:")
        logger.info("  python main.py fit --config configs/trial_local.yaml")
    else:
        logger.info("SOME TESTS FAILED ✗")
        logger.info("\nPlease fix the issues before proceeding to training.")
    logger.info("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__": 
    sys.exit(main())