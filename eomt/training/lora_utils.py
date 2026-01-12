import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging


class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation (LoRA).
    
    Implements: h = Wx + (BA)x, where B and A are low-rank matrices.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        
        # LoRA parameters (trainable)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Flag to track if LoRA is enabled
        self.lora_enabled = True
        
    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
            base_output: output from the original frozen linear layer
        """
        if not self.lora_enabled:
            return base_output
        
        # Apply LoRA: output = base_output + scaling * (x @ A @ B)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return base_output + lora_out * self.scaling
    
    def enable_lora(self):
        self.lora_enabled = True
    
    def disable_lora(self):
        self.lora_enabled = False


class LoRAInjector:
    """Utility class to inject LoRA into existing models."""
    
    @staticmethod
    def inject_lora_into_linear(
        module: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> nn.Module:
        """
        Wraps a linear layer to include LoRA.
        Returns a module that applies: frozen_linear + lora_adaptation
        """
        # Freeze original weights
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        
        # Create LoRA adapter
        lora = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # Return wrapped module
        return LoRAWrappedLinear(module, lora)


class LoRAWrappedLinear(nn.Module):
    """Wrapper that combines frozen linear + LoRA."""
    
    def __init__(self, base_linear: nn.Linear, lora: LoRALinear):
        super().__init__()
        self.base_linear = base_linear
        self.lora = lora
        
        # Expose attributes for compatibility
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight = base_linear.weight
        self.bias = base_linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base output (frozen)
        base_out = self.base_linear(x)
        # Add LoRA adaptation
        return self.lora(x, base_out)


def inject_lora_into_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_only_last_n_blocks: Optional[int] = None,
) -> int:
    """
    Inject LoRA into specified modules of a model.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'v_proj', 'k_proj'])
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability for LoRA
        lora_only_last_n_blocks: If specified, only apply LoRA to last N blocks
        
    Returns:
        Number of LoRA adapters injected
    """
    injected_count = 0
    
    # Get all block indices if limiting to last N blocks
    if lora_only_last_n_blocks is not None:
        block_indices = []
        for name, _ in model.named_modules():
            if 'blocks.' in name:
                parts = name.split('blocks.')
                if len(parts) > 1:
                    idx = int(parts[1].split('.')[0])
                    if idx not in block_indices:
                        block_indices.append(idx)
        block_indices.sort()
        target_block_indices = set(block_indices[-lora_only_last_n_blocks:])
        logging.info(f"Applying LoRA only to blocks: {target_block_indices}")
    else:
        target_block_indices = None
    
    # Iterate through all modules
    for name, module in list(model.named_modules()):
        # Skip if we're limiting to certain blocks
        if target_block_indices is not None:
            if 'blocks.' in name:
                parts = name.split('blocks.')
                if len(parts) > 1:
                    idx = int(parts[1].split('.')[0])
                    if idx not in target_block_indices:
                        continue
            else:
                continue
        
        # Check if this module matches our target patterns
        if not isinstance(module, nn.Linear):
            continue
        
        should_inject = False
        for target in target_modules:
            if target in name:
                should_inject = True
                break
        
        if not should_inject:
            continue
        
        # Inject LoRA
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)
        
        # Create LoRA-wrapped version
        lora_module = LoRAInjector.inject_lora_into_linear(
            module, rank=rank, alpha=alpha, dropout=dropout
        )
        
        # Replace module
        setattr(parent, child_name, lora_module)
        injected_count += 1
        
        logging.info(f"Injected LoRA into: {name}")
    
    logging.info(f"Total LoRA adapters injected: {injected_count}")
    return injected_count


def count_lora_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count LoRA parameters and total trainable parameters.
    
    Returns:
        (lora_params, total_trainable_params)
    """
    lora_params = 0
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
    
    return lora_params, total_trainable


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters from model state dict."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.detach().clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict, strict: bool = True):
    """Load LoRA parameters into model."""
    missing_keys = []
    unexpected_keys = []
    
    model_state = {k: v for k, v in model.named_parameters() if 'lora_' in k}
    
    for key in lora_state:
        if key in model_state:
            model_state[key].data.copy_(lora_state[key])
        else:
            unexpected_keys.append(key)
    
    for key in model_state:
        if key not in lora_state:
            missing_keys.append(key)
    
    if strict and (missing_keys or unexpected_keys):
        error_msg = []
        if missing_keys:
            error_msg.append(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            error_msg.append(f"Unexpected keys: {unexpected_keys}")
        raise RuntimeError('\n'.join(error_msg))
    
    return missing_keys, unexpected_keys