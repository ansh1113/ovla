"""
STRATEGY MAPPER - Cross-Class Strategy Correction

Maps strategies from source robot class to target robot class.

Example:
- Source (Arm): "Reach straight to point"
- Target (Humanoid): "Lean torso + shift weight + extend arm"

This is the KEY innovation for cross-class transfer.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class StrategyMapper(nn.Module):
    """
    Maps task strategies across robot morphologies.
    
    Key difference from Universal Mapper:
    - Universal Mapper: Low-level motion mapping
    - Strategy Mapper: High-level strategy correction
    
    Example:
        Arm strategy: "Extend arm to reach cup"
        → Humanoid strategy: "Shift weight left, lean forward, extend right arm"
    """
    
    def __init__(
        self,
        strategy_dim: int = 64,
        morphology_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Encode source strategy
        self.source_strategy_encoder = nn.Sequential(
            nn.Linear(strategy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encode morphology differences
        self.morphology_diff_encoder = nn.Sequential(
            nn.Linear(morphology_dim * 2, hidden_dim),  # Concat source + target
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention: How should target robot modify strategy?
        self.strategy_correction = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Generate corrected strategy
        self.strategy_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, strategy_dim)
        )
        
    def forward(
        self,
        source_strategy: torch.Tensor,  # [B, strategy_dim]
        source_morphology: torch.Tensor,  # [B, morphology_dim]
        target_morphology: torch.Tensor  # [B, morphology_dim]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Map strategy from source to target robot.
        
        Returns:
            corrected_strategy: Adjusted strategy for target robot
            corrections: Dict of what was corrected
        """
        
        # Encode source strategy
        strategy_features = self.source_strategy_encoder(source_strategy)  # [B, H]
        
        # Encode morphology difference
        morph_diff = torch.cat([source_morphology, target_morphology], dim=-1)
        morph_features = self.morphology_diff_encoder(morph_diff)  # [B, H]
        
        # Cross-attention: Strategy queries morphology difference
        strategy_features = strategy_features.unsqueeze(1)  # [B, 1, H]
        morph_features = morph_features.unsqueeze(1)  # [B, 1, H]
        
        corrected_features, attention_weights = self.strategy_correction(
            query=strategy_features,
            key=morph_features,
            value=morph_features
        )
        
        corrected_features = corrected_features.squeeze(1)  # [B, H]
        
        # Generate corrected strategy
        corrected_strategy = self.strategy_generator(corrected_features)
        
        # Compute what changed
        corrections = {
            'attention_weights': attention_weights.squeeze(1),
            'strategy_delta': corrected_strategy - source_strategy
        }
        
        return corrected_strategy, corrections
