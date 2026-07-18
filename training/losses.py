import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, Tuple

class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy with class weights
    ρ = (# negatives) / (# positives)
    """
    
    def __init__(self, pos_weight: float = 1.0, max_weight: float = 100.0):
        super().__init__()
        self.pos_weight = min(pos_weight, max_weight)  # Cap to avoid instability
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) or (N, 1) unnormalized predictions
            targets: (N,) or (N, 1) binary labels
        """
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=pos_weight_tensor
        )
        return loss


class AdaptiveWeightedBCELoss(nn.Module):
    """
    Adaptive weighted BCE using scheduled weight decay
    Starts with high weight for positives, gradually decreases
    """
    
    def __init__(self, initial_pos_weight: float, 
                 decay_rate: float = 0.95, min_weight: float = 1.0):
        super().__init__()
        self.initial_pos_weight = initial_pos_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.epoch = 0
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with epoch-dependent weighting"""
        
        # Decay weight over time
        current_weight = max(
            self.initial_pos_weight * (self.decay_rate ** self.epoch),
            self.min_weight
        )
        
        pos_weight_tensor = torch.tensor([current_weight], device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=pos_weight_tensor
        )
        return loss
        
    def step_epoch(self):
        """Advance epoch count to decay weight"""
        self.epoch += 1


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    L = -α(1-p_t)^γ log(p_t)
    
    Where:
    - α: balance weight for positive class
    - γ: focusing parameter (higher = more focus on hard examples)
    - p_t: predicted probability
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N,) unnormalized predictions
            targets: (N,) binary labels (0 or 1)
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Compute sigmoid to get probabilities
        p = torch.sigmoid(logits)
        
        # Compute cross-entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weight for class imbalance
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combine Weighted BCE + Focal Loss
    Total loss = (1 - λ) * WeightedBCE + λ * FocalLoss
    """
    
    def __init__(self, pos_weight: float = 10.0, alpha: float = 0.25, 
                 gamma: float = 2.0, lambda_weight: float = 0.5):
        super().__init__()
        self.wbce = WeightedBCELoss(pos_weight)
        self.focal = FocalLoss(alpha, gamma)
        self.lambda_weight = lambda_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        wbce_loss = self.wbce(logits, targets)
        focal_loss = self.focal(logits, targets)
        
        return (1 - self.lambda_weight) * wbce_loss + \
               self.lambda_weight * focal_loss
