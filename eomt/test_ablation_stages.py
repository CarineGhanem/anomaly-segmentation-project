#!/usr/bin/env python3
"""
Test script for validating magnitude-aware calibration setup locally.
Tests parameter freezing, LogitNorm modes, and magnitude loss.
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
            logit_norm_temp=0.5,
            logit_norm_mode="ce",  # Updated default
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
            # NEW: Magnitude-aware parameters
            use_magnitude_loss=False,
            magnitude_coefficient=0.5,
            magnitude_threshold=1.0,
        )
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Manually trigger stage application (normally happens in setup)
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


def test_logit_norm_modes():
    """Test LogitNorm with different modes (none, ce, matching, both)."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING LOGITNORM MODES")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Test data
    B, Q, C = 2, 50, 19
    class_logits = torch.randn(B, Q, C + 1)
    
    modes_to_test = ["none", "ce", "matching", "both"]
    results = {}
    
    for mode in modes_to_test:
        logger.info(f"\n--- Testing mode: {mode} ---")
        
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
            logit_norm_temp=0.5,
            logit_norm_mode=mode,
        )
        
        # Test CE context
        ce_normalized = criterion.apply_logit_norm(class_logits, context="ce")
        ce_is_normalized = not torch.allclose(ce_normalized, class_logits, rtol=0.01)
        
        # Test matching context
        matching_normalized = criterion.apply_logit_norm(class_logits, context="matching")
        matching_is_normalized = not torch.allclose(matching_normalized, class_logits, rtol=0.01)
        
        logger.info(f"  CE context normalized: {ce_is_normalized}")
        logger.info(f"  Matching context normalized: {matching_is_normalized}")
        
        # Verify behavior
        if mode == "none":
            expected = (False, False)
        elif mode == "ce":
            expected = (True, False)
        elif mode == "matching":
            expected = (False, True)
        elif mode == "both":
            expected = (True, True)
        
        actual = (ce_is_normalized, matching_is_normalized)
        success = actual == expected
        
        if success:
            logger.info(f"  ✓ Mode '{mode}' behaves correctly")
        else:
            logger.warning(f"  ✗ Mode '{mode}' behavior incorrect: expected {expected}, got {actual}")
        
        results[mode] = success
    
    all_passed = all(results.values())
    
    logger.info(f"\n{'='*70}")
    logger.info(f"LogitNorm Modes: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return all_passed


def test_magnitude_loss():
    """Test magnitude regularization computation."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING MAGNITUDE LOSS")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Create loss with magnitude loss enabled
    criterion = MaskClassificationLoss(
        num_points=512,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_coefficient=5.0,
        dice_coefficient=5.0,
        class_coefficient=2.0,
        num_labels=19,
        no_object_coefficient=0.1,
        use_logit_norm=True,
        logit_norm_temp=0.5,
        logit_norm_mode="ce",
        # Enable magnitude loss
        use_magnitude_loss=True,
        magnitude_coefficient=0.5,
        magnitude_threshold=1.0,
    )
    
    # Create dummy data
    B, Q, C = 2, 20, 19
    class_logits = torch.randn(B, Q, C + 1)
    
    # Create dummy indices (some matched, some unmatched)
    indices = [
        (torch.tensor([0, 2, 5], dtype=torch.long), torch.tensor([0, 1, 2], dtype=torch.long)),
        (torch.tensor([1, 3], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)),
    ]
    
    logger.info(f"\nTest setup:")
    logger.info(f"  Batch size: {B}, Queries: {Q}, Classes: {C}")
    logger.info(f"  Matched queries batch 0: {indices[0][0].tolist()}")
    logger.info(f"  Matched queries batch 1: {indices[1][0].tolist()}")
    
    try:
        # Compute magnitude loss
        loss_magnitude = criterion.compute_magnitude_regularizer(class_logits, indices)
        
        logger.info(f"\nMagnitude loss computation:")
        logger.info(f"  Loss value: {loss_magnitude.item():.4f}")
        logger.info(f"  Loss is finite: {torch.isfinite(loss_magnitude).item()}")
        logger.info(f"  Loss is non-negative: {(loss_magnitude >= 0).item()}")
        
        # Verify properties
        success = (
            torch.isfinite(loss_magnitude).item() and
            loss_magnitude.item() >= 0
        )
        
        if success:
            logger.info("✓ Magnitude loss computed correctly")
        else:
            logger.warning("✗ Magnitude loss has invalid properties")
        
    except Exception as e:
        logger.error(f"✗ Magnitude loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Magnitude Loss: {'✓ PASSED' if success else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return success


def test_full_loss_computation():
    """Test full loss computation with all magnitude-aware features."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING FULL LOSS COMPUTATION")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Test configurations
    configs = [
        {
            "name": "LogitNorm CE-only + Magnitude (Ablation 2)",
            "use_logit_norm": True,
            "logit_norm_mode": "ce",
            "use_magnitude_loss": True,
        },
    ]
    
    results = {}
    
    for config in configs:
        logger.info(f"\n--- Testing: {config['name']} ---")
        
        # Create loss
        criterion = MaskClassificationLoss(
            num_points=512,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=19,
            no_object_coefficient=0.1,
            use_logit_norm=config["use_logit_norm"],
            logit_norm_temp=0.5,
            logit_norm_mode=config["logit_norm_mode"],
            use_magnitude_loss=config["use_magnitude_loss"],
            magnitude_coefficient=0.5,
            magnitude_threshold=1.0,
        )
        
        # Create dummy data
        B, Q, H, W, C = 1, 20, 64, 64, 19
        num_objects = 5
        
        mask_logits = torch.randn(B, Q, H, W)
        class_logits = torch.randn(B, Q, C + 1)
        
        targets = [{
            'masks': torch.randint(0, 2, (num_objects, H, W)).bool(),
            'labels': torch.randint(0, C, (num_objects,)),
        }]
        
        try:
            # Compute losses
            losses = criterion(mask_logits, class_logits, targets)
            
            logger.info(f"  Computed losses:")
            for key, value in losses.items():
                logger.info(f"    {key}: {value.item():.4f}")
            
            # Check expected losses
            expected_losses = ["loss_ce", "loss_mask", "loss_dice"]
            if config["use_magnitude_loss"]:
                expected_losses.append("loss_magnitude")
            
            # Verify all expected losses present
            missing = [k for k in expected_losses if k not in losses]
            unexpected = [k for k in losses if k not in expected_losses]
            
            if missing:
                logger.warning(f"  ⚠ Missing losses: {missing}")
            if unexpected:
                logger.warning(f"  ⚠ Unexpected losses: {unexpected}")
            
            # Check all losses are finite
            all_finite = all(torch.isfinite(v) for v in losses.values())
            
            success = all_finite and not missing
            
            if success:
                logger.info(f"  ✓ All losses computed correctly")
            else:
                logger.warning(f"  ✗ Loss computation has issues")
            
            results[config["name"]] = success
            
        except Exception as e:
            logger.error(f"  ✗ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            results[config["name"]] = False
    
    all_passed = all(results.values())
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Full Loss Computation: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return all_passed


def test_backward_compatibility():
    """Test that the new implementation is backward compatible."""
    logger.info(f"\n{'='*70}")
    logger.info("TESTING BACKWARD COMPATIBILITY")
    logger.info('='*70)
    
    from training.mask_classification_loss import MaskClassificationLoss
    
    # Test case 1: Old style (no magnitude-aware features)
    logger.info("\n--- Test 1: Old configuration (baseline) ---")
    try:
        criterion_old = MaskClassificationLoss(
            num_points=512,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=19,
            no_object_coefficient=0.1,
            use_logit_norm=False,
            logit_norm_temp=0.1,
            logit_norm_mode="both",
            # No magnitude-aware parameters
        )
        logger.info("  ✓ Old configuration works")
        old_works = True
    except Exception as e:
        logger.error(f"  ✗ Old configuration failed: {e}")
        old_works = False
    
    # Test case 2: Partial new features
    logger.info("\n--- Test 2: Partial new features ---")
    try:
        criterion_partial = MaskClassificationLoss(
            num_points=512,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=19,
            no_object_coefficient=0.1,
            use_logit_norm=True,
            logit_norm_temp=0.5,
            logit_norm_mode="ce",
            use_magnitude_loss=True,  # Only magnitude loss
            magnitude_coefficient=0.5,
        )
        logger.info("  ✓ Partial new features work")
        partial_works = True
    except Exception as e:
        logger.error(f"  ✗ Partial new features failed: {e}")
        partial_works = False
    
    success = old_works and partial_works
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Backward Compatibility: {'✓ PASSED' if success else '✗ FAILED'}")
    logger.info('='*70 + '\n')
    
    return success


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("MAGNITUDE-AWARE CALIBRATION - LOCAL VALIDATION")
    logger.info("="*70 + "\n")
    
    results = {}
    
    # Test 1: Stage freezing (basic functionality)
    logger.info("="*70)
    logger.info("PART 1: STAGE FREEZING TESTS")
    logger.info("="*70)
    for stage in ["A_head", "B_queries", "full"]:
        results[f"stage_{stage}"] = test_stage_freezing(stage)
    
    # Test 2: LogitNorm modes
    logger.info("\n" + "="*70)
    logger.info("PART 2: LOGITNORM MODE TESTS")
    logger.info("="*70)
    results["logit_norm_modes"] = test_logit_norm_modes()
    
    # Test 3: Magnitude loss
    logger.info("\n" + "="*70)
    logger.info("PART 3: MAGNITUDE LOSS TESTS")
    logger.info("="*70)
    results["magnitude_loss"] = test_magnitude_loss()
    
    # Test 4: Full loss computation
    logger.info("\n" + "="*70)
    logger.info("PART 4: FULL LOSS COMPUTATION TESTS")
    logger.info("="*70)
    results["full_loss_computation"] = test_full_loss_computation()
    
    # Test 5: Backward compatibility
    logger.info("\n" + "="*70)
    logger.info("PART 5: BACKWARD COMPATIBILITY TESTS")
    logger.info("="*70)
    results["backward_compatibility"] = test_backward_compatibility()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:30s}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("ALL TESTS PASSED ✓")
        logger.info("\nYour magnitude-aware calibration is ready!")
        logger.info("\nNext steps:")
        logger.info("  1. Update checkpoint paths in config files")
        logger.info("  2. Run Ablation 2 (primary contribution):")
        logger.info("     python main.py fit --config configs/dinov2/cityscapes/semantic/ablation2_logitnorm_ce_magnitude.yaml")
        logger.info("  3. Evaluate on anomaly datasets")
    else:
        logger.info("SOME TESTS FAILED ✗")
        logger.info("\nPlease review the failures above and:")
        logger.info("  1. Check that you've updated both code files")
        logger.info("  2. Verify imports are working correctly")
        logger.info("  3. Review error messages for specific issues")
    logger.info("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__": 
    sys.exit(main())
