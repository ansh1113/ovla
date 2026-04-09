"""
Action Mapper - The Missing Link in O-VLA

Maps VLA actions from source robot to target robot using:
- Layer 1's morphology-conditioned visual embeddings
- Semantic joint correspondence learning
- Cross-attention fusion

This is what makes "ANY VLA → ANY Robot" actually work.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActionMapper(nn.Module):
    """
    Maps actions from source robot (VLA was trained on) to target robot
    
    Architecture:
    - Encode VLA action, visual context (Layer 1), and target morphology separately
    - Fuse them via multi-head attention
    - Decode to target robot's joint space
    """
    
    def __init__(
        self, 
        max_source_dof=32,  # Max DOF for any source robot
        max_target_dof=32,  # Max DOF for any target robot
        visual_dim=128,     # Layer 1 embedding dimension
        morphology_dim=256, # Morphology feature dimension
        hidden_dim=256,
        num_attention_heads=8
    ):
        super().__init__()
        
        self.max_source_dof = max_source_dof
        self.max_target_dof = max_target_dof
        
        # Encoders for each input modality
        self.action_encoder = nn.Sequential(
            nn.Linear(max_source_dof, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.target_morphology_encoder = nn.Sequential(
            nn.Linear(morphology_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.source_morphology_encoder = nn.Sequential(
            nn.Linear(morphology_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-head cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_attention_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Self-attention for refining
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Decoder to target action space
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, max_target_dof)
        )
        
        # Learnable positional encoding for joint sequences
        self.joint_pos_encoding = nn.Parameter(torch.randn(1, max_target_dof, hidden_dim) * 0.02)
        
    def forward(
        self, 
        source_action,          # [B, source_dof]
        visual_embedding,       # [B, 128] from Layer 1
        target_morphology,      # [B, 256]
        source_morphology,      # [B, 256]
        source_dof,             # int
        target_dof              # int
    ):
        """
        Map action from source robot to target robot
        
        Args:
            source_action: VLA output for source robot
            visual_embedding: Layer 1's morphology-conditioned visual features
            target_morphology: Target robot's morphology features
            source_morphology: Source robot's morphology features
            source_dof: Actual DOF of source robot
            target_dof: Actual DOF of target robot
            
        Returns:
            target_action: Mapped action for target robot [B, target_dof]
        """
        batch_size = source_action.shape[0]
        
        # Pad source action to max_source_dof
        if source_action.shape[1] < self.max_source_dof:
            padding = torch.zeros(
                batch_size, 
                self.max_source_dof - source_action.shape[1],
                device=source_action.device
            )
            source_action_padded = torch.cat([source_action, padding], dim=1)
        else:
            source_action_padded = source_action
        
        # Encode all inputs
        action_encoded = self.action_encoder(source_action_padded)  # [B, hidden_dim]
        visual_encoded = self.visual_encoder(visual_embedding)      # [B, hidden_dim]
        target_morph_encoded = self.target_morphology_encoder(target_morphology)  # [B, hidden_dim]
        source_morph_encoded = self.source_morphology_encoder(source_morphology)  # [B, hidden_dim]
        
        # Create query: target morphology with positional encoding for each joint
        # [B, target_dof, hidden_dim]
        target_queries = target_morph_encoded.unsqueeze(1).repeat(1, target_dof, 1)
        target_queries = target_queries + self.joint_pos_encoding[:, :target_dof, :]
        
        # Create key-value: concatenate action, visual, and source morphology
        # [B, 3, hidden_dim]
        keys_values = torch.stack([
            action_encoded,
            visual_encoded, 
            source_morph_encoded
        ], dim=1)
        
        # Cross-attention: target queries attend to source information
        fused, attention_weights = self.cross_attention(
            target_queries,  # queries: what we want (target joints)
            keys_values,     # keys: what we have (source action + context)
            keys_values      # values: what we have
        )
        
        # Self-attention: refine joint representations
        refined, _ = self.self_attention(fused, fused, fused)
        
        # Add residual connection
        refined = refined + fused
        
        # Decode to target action space
        # Take mean across joint dimension for global context
        global_context = refined.mean(dim=1)  # [B, hidden_dim]
        
        # Decode
        target_action_full = self.decoder(global_context)  # [B, max_target_dof]
        
        # Extract only the target_dof dimensions
        target_action = target_action_full[:, :target_dof]
        
        return target_action
    
    def get_attention_weights(
        self,
        source_action,
        visual_embedding,
        target_morphology,
        source_morphology,
        source_dof,
        target_dof
    ):
        """Return attention weights for interpretability"""
        batch_size = source_action.shape[0]
        
        # Same encoding as forward
        if source_action.shape[1] < self.max_source_dof:
            padding = torch.zeros(
                batch_size, 
                self.max_source_dof - source_action.shape[1],
                device=source_action.device
            )
            source_action_padded = torch.cat([source_action, padding], dim=1)
        else:
            source_action_padded = source_action
        
        action_encoded = self.action_encoder(source_action_padded)
        visual_encoded = self.visual_encoder(visual_embedding)
        target_morph_encoded = self.target_morphology_encoder(target_morphology)
        source_morph_encoded = self.source_morphology_encoder(source_morphology)
        
        target_queries = target_morph_encoded.unsqueeze(1).repeat(1, target_dof, 1)
        target_queries = target_queries + self.joint_pos_encoding[:, :target_dof, :]
        
        keys_values = torch.stack([
            action_encoded,
            visual_encoded,
            source_morph_encoded
        ], dim=1)
        
        _, attention_weights = self.cross_attention(
            target_queries,
            keys_values,
            keys_values
        )
        
        return attention_weights


def test_action_mapper():
    """Test that ActionMapper works"""
    print("Testing ActionMapper...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mapper = ActionMapper().to(device)
    
    # Test case: Franka (7-DOF) → Humanoid (23-DOF)
    batch_size = 4
    
    source_action = torch.randn(batch_size, 7).to(device)
    visual_embedding = torch.randn(batch_size, 128).to(device)
    target_morphology = torch.randn(batch_size, 256).to(device)
    source_morphology = torch.randn(batch_size, 256).to(device)
    
    output = mapper(
        source_action,
        visual_embedding,
        target_morphology,
        source_morphology,
        source_dof=7,
        target_dof=23
    )
    
    assert output.shape == (batch_size, 23), f"Expected (4, 23), got {output.shape}"
    print(f"✓ Franka (7-DOF) → Humanoid (23-DOF): {output.shape}")
    
    # Test case: Humanoid (23-DOF) → Quadruped (12-DOF)
    source_action = torch.randn(batch_size, 23).to(device)
    output = mapper(
        source_action,
        visual_embedding,
        target_morphology,
        source_morphology,
        source_dof=23,
        target_dof=12
    )
    
    assert output.shape == (batch_size, 12), f"Expected (4, 12), got {output.shape}"
    print(f"✓ Humanoid (23-DOF) → Quadruped (12-DOF): {output.shape}")
    
    # Test attention weights
    attn = mapper.get_attention_weights(
        torch.randn(1, 7).to(device),
        torch.randn(1, 128).to(device),
        torch.randn(1, 256).to(device),
        torch.randn(1, 256).to(device),
        source_dof=7,
        target_dof=23
    )
    
    print(f"✓ Attention weights shape: {attn.shape}")
    print("\n✅ ActionMapper tests passed!")


if __name__ == '__main__':
    test_action_mapper()
