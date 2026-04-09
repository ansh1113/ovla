"""
UNIVERSAL SEMANTIC MAPPER - The Core of "ANY VLA → ANY Robot"

This is THE innovation that makes O-VLA truly universal.

Purpose:
  Take a semantic action description (robot-agnostic) and map it to 
  joint commands for ANY target robot.

Key Difference from Old Action Mapper:
  OLD: Learned "Franka → G1" specific mapping (memorized pairs)
  NEW: Learns "semantic motion → robot execution" (universal primitives)

Training:
  NOT on robot pairs (Franka→G1, Franka→Laikago, etc.)
  BUT on semantic primitives across diverse robots
  
  Example training sample:
    {
      'semantic': 'wrist_flexion_15deg',
      'target_robot': UR5,
      'execution': [0, 0, 0, 0, 0.26, 0]  # UR5's wrist joint moves
    }

This allows zero-shot to new robots if semantic is understood.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ovla.semantic_extractor import SemanticAction

@dataclass
class URDFGraphNode:
    """Node in the kinematic graph"""
    joint_index: int
    joint_name: str
    joint_type: str  # revolute, prismatic
    parent_index: Optional[int]
    children_indices: List[int]
    
    # Kinematic properties
    axis: np.ndarray
    limits: Tuple[float, float]
    
    # Semantic properties
    role_label: str  # wrist, elbow, shoulder, etc.
    component_label: str  # left_arm, right_arm, etc.
    workspace_contribution: float

class URDFGraphEncoder(nn.Module):
    """
    Graph Neural Network for encoding robot kinematic structure.
    
    This learns to represent robot morphology in a way that captures
    semantic relationships between joints.
    """
    
    def __init__(self, node_feature_dim: int = 32, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(16, node_feature_dim),  # 16 raw features per joint
            nn.LayerNorm(node_feature_dim),
            nn.ReLU(),
        )
        
        # Graph convolution layers (message passing)
        self.gconv1 = nn.Linear(node_feature_dim * 2, hidden_dim)  # node + neighbor
        self.gconv2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gconv3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Global graph pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        
        # Per-joint encoding (for joint matching)
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [batch, num_nodes, 16] - Raw joint features
            adjacency: [batch, num_nodes, num_nodes] - Kinematic tree adjacency
        
        Returns:
            global_graph_embedding: [batch, output_dim] - Whole robot representation
            joint_embeddings: [batch, num_nodes, output_dim] - Per-joint representations
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Encode node features
        x = self.node_encoder(node_features)  # [batch, num_nodes, node_feature_dim]
        
        # Graph convolution layer 1
        neighbors = torch.bmm(adjacency, x)  # [batch, num_nodes, node_feature_dim]
        x_with_neighbors = torch.cat([x, neighbors], dim=-1)  # [batch, num_nodes, 2*node_feature_dim]
        x = F.relu(self.gconv1(x_with_neighbors))  # [batch, num_nodes, hidden_dim]
        
        # Graph convolution layer 2
        neighbors = torch.bmm(adjacency, x)
        x_with_neighbors = torch.cat([x, neighbors], dim=-1)
        x = F.relu(self.gconv2(x_with_neighbors))
        
        # Graph convolution layer 3
        neighbors = torch.bmm(adjacency, x)
        x_with_neighbors = torch.cat([x, neighbors], dim=-1)
        x = F.relu(self.gconv3(x_with_neighbors))  # [batch, num_nodes, hidden_dim]
        
        # Global graph embedding (mean pooling)
        global_embedding = self.graph_pool(x.mean(dim=1))  # [batch, output_dim]
        
        # Per-joint embeddings
        joint_embeddings = self.joint_encoder(x)  # [batch, num_nodes, output_dim]
        
        return global_embedding, joint_embeddings


class SemanticActionEncoder(nn.Module):
    """
    Encode semantic action into a representation suitable for mapping.
    
    Takes the 128-dim semantic fingerprint and enriches it with
    task-relevant information.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Motion type embedding encoder
        self.motion_type_encoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Fusion
        self.fusion = nn.Linear(output_dim + 64, output_dim)
    
    def forward(self, semantic_fingerprint: torch.Tensor, motion_type_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_fingerprint: [batch, 128] - Full semantic fingerprint
            motion_type_embedding: [batch, 16] - Motion type encoding
        
        Returns:
            encoded_semantic: [batch, 256] - Enriched semantic representation
        """
        # Encode semantic fingerprint
        semantic_encoded = self.encoder(semantic_fingerprint)  # [batch, 256]
        
        # Encode motion type
        motion_encoded = self.motion_type_encoder(motion_type_embedding)  # [batch, 64]
        
        # Fuse
        fused = torch.cat([semantic_encoded, motion_encoded], dim=-1)  # [batch, 320]
        output = self.fusion(fused)  # [batch, 256]
        
        return output


class JointSemanticMatcher(nn.Module):
    """
    Match semantic action to target robot joints using attention.
    
    This is the KEY module that learns:
    "Which joints in the target robot should execute this semantic action?"
    
    Uses cross-attention where:
    - Queries: Target robot joints
    - Keys/Values: Semantic action representation
    """
    
    def __init__(self, semantic_dim: int = 256, joint_dim: int = 256, num_heads: int = 8):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(joint_dim)
        self.norm2 = nn.LayerNorm(joint_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim * 4, joint_dim),
            nn.Dropout(0.1),
        )
        self.norm3 = nn.LayerNorm(joint_dim)
    
    def forward(self, joint_embeddings: torch.Tensor, semantic_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_embeddings: [batch, num_joints, 256] - Target robot joint embeddings
            semantic_embedding: [batch, 256] - Semantic action embedding
        
        Returns:
            matched_joint_features: [batch, num_joints, 256] - Joints informed by semantics
        """
        # Expand semantic to match joints dimension
        semantic_expanded = semantic_embedding.unsqueeze(1)  # [batch, 1, 256]
        
        # Cross-attention: Joints attend to semantic action
        attn_output, _ = self.cross_attention(
            query=joint_embeddings,
            key=semantic_expanded,
            value=semantic_expanded
        )
        joint_embeddings = self.norm1(joint_embeddings + attn_output)
        
        # Self-attention: Joints attend to each other
        attn_output, _ = self.self_attention(
            query=joint_embeddings,
            key=joint_embeddings,
            value=joint_embeddings
        )
        joint_embeddings = self.norm2(joint_embeddings + attn_output)
        
        # FFN
        ffn_output = self.ffn(joint_embeddings)
        joint_embeddings = self.norm3(joint_embeddings + ffn_output)
        
        return joint_embeddings


class MotionGenerator(nn.Module):
    """
    Generate actual joint values for target robot.
    
    Takes the semantically-informed joint embeddings and generates
    the actual joint position/velocity commands.
    """
    
    def __init__(self, joint_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        # Per-joint decoder
        self.joint_decoder = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1),  # Single value per joint
        )
        
        # Activation predictor (which joints should move?)
        self.activation_predictor = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, joint_features: torch.Tensor, max_dof: int) -> torch.Tensor:
        """
        Args:
            joint_features: [batch, num_joints, 256] - Semantically-informed joints
            max_dof: Maximum DOF to output (for padding)
        
        Returns:
            joint_actions: [batch, max_dof] - Joint position/velocity commands
        """
        batch_size, num_joints, _ = joint_features.shape
        
        # Decode joint values
        joint_values = self.joint_decoder(joint_features).squeeze(-1)  # [batch, num_joints]
        
        # Predict activation
        activations = self.activation_predictor(joint_features).squeeze(-1)  # [batch, num_joints]
        
        # Apply activation gating
        joint_actions = joint_values * activations  # [batch, num_joints]
        
        # Pad to max_dof
        if num_joints < max_dof:
            padding = torch.zeros(batch_size, max_dof - num_joints, device=joint_actions.device)
            joint_actions = torch.cat([joint_actions, padding], dim=1)
        elif num_joints > max_dof:
            joint_actions = joint_actions[:, :max_dof]
        
        return joint_actions


class UniversalSemanticMapper(nn.Module):
    """
    THE CORE INNOVATION: Universal Semantic Mapper
    
    Maps semantic actions to ANY robot without robot-pair-specific training.
    
    Architecture:
      1. Encode semantic action (robot-agnostic)
      2. Encode target robot structure (GNN)
      3. Match semantics to joints (cross-attention)
      4. Generate joint commands
    
    Training Strategy:
      Train on semantic primitives across diverse robots:
      - "wrist_flexion" + UR5 → [0,0,0,0,0.26,0]
      - "wrist_flexion" + Franka → [0,0,0,0,0,0.26]
      - "elbow_bend" + G1_right_arm → [0,0,0,...,0.5,...]
      
      NOT on robot pairs:
      - Franka → G1 (pair-specific, doesn't generalize)
    
    This allows zero-shot to new robots if primitives are well-learned.
    """
    
    def __init__(
        self,
        max_dof: int = 32,
        semantic_dim: int = 256,
        joint_dim: int = 256,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.max_dof = max_dof
        
        # 1. Semantic Action Encoder
        self.semantic_encoder = SemanticActionEncoder(
            input_dim=128,  # Semantic fingerprint
            hidden_dim=256,
            output_dim=semantic_dim
        )
        
        # 2. URDF Graph Encoder (for target robot)
        self.urdf_encoder = URDFGraphEncoder(
            node_feature_dim=32,
            hidden_dim=128,
            output_dim=joint_dim
        )
        
        # 3. Joint Semantic Matcher
        self.joint_matcher = JointSemanticMatcher(
            semantic_dim=semantic_dim,
            joint_dim=joint_dim,
            num_heads=num_attention_heads
        )
        
        # 4. Motion Generator
        self.motion_generator = MotionGenerator(
            joint_dim=joint_dim,
            hidden_dim=128
        )
    
    def forward(
        self,
        semantic_fingerprint: torch.Tensor,
        motion_type_embedding: torch.Tensor,
        target_node_features: torch.Tensor,
        target_adjacency: torch.Tensor,
        target_dof: int
    ) -> torch.Tensor:
        """
        Map semantic action to target robot joint commands.
        
        Args:
            semantic_fingerprint: [batch, 128] - From SemanticExtractor
            motion_type_embedding: [batch, 16] - Motion type encoding
            target_node_features: [batch, num_nodes, 16] - Target robot joint features
            target_adjacency: [batch, num_nodes, num_nodes] - Target kinematic tree
            target_dof: Number of DOF in target robot
        
        Returns:
            target_joint_actions: [batch, max_dof] - Joint commands for target robot
        """
        # 1. Encode semantic action
        semantic_encoded = self.semantic_encoder(
            semantic_fingerprint,
            motion_type_embedding
        )  # [batch, 256]
        
        # 2. Encode target robot structure
        global_robot_embedding, joint_embeddings = self.urdf_encoder(
            target_node_features,
            target_adjacency
        )  # [batch, 256], [batch, num_nodes, 256]
        
        # 3. Match semantics to joints
        matched_joints = self.joint_matcher(
            joint_embeddings,
            semantic_encoded
        )  # [batch, num_nodes, 256]
        
        # 4. Generate joint commands
        joint_actions = self.motion_generator(
            matched_joints,
            self.max_dof
        )  # [batch, max_dof]
        
        return joint_actions
    
    def map_semantic_to_robot(
        self,
        semantic_action: SemanticAction,
        target_urdf_data: Dict
    ) -> np.ndarray:
        """
        High-level API: Map semantic action to target robot.
        
        Args:
            semantic_action: SemanticAction from SemanticExtractor
            target_urdf_data: Dict with target robot structure
        
        Returns:
            joint_actions: numpy array of joint commands
        """
        # Prepare inputs
        semantic_fingerprint = torch.FloatTensor(semantic_action.semantic_fingerprint).unsqueeze(0)
        motion_type_embedding = torch.FloatTensor(semantic_action.motion_type_embedding).unsqueeze(0)
        
        # TODO: Extract from target_urdf_data
        # For now, use dummy data
        num_nodes = target_urdf_data['num_joints']
        target_node_features = torch.randn(1, num_nodes, 16)
        target_adjacency = torch.eye(num_nodes).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            joint_actions = self(
                semantic_fingerprint,
                motion_type_embedding,
                target_node_features,
                target_adjacency,
                target_urdf_data['dof']
            )
        
        return joint_actions.squeeze(0).numpy()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_urdf_graph_data(urdf_path: str) -> Dict:
    """
    Extract graph structure from URDF for the mapper.
    
    Returns:
        Dict with node_features, adjacency matrix, etc.
    """
    import pybullet as p
    import pybullet_data
    
    # Load URDF
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=client)
    
    num_joints_total = p.getNumJoints(robot, physicsClientId=client)
    
    # Extract only movable joints
    node_features = []
    joint_indices = []
    
    for i in range(num_joints_total):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            # Create 16-dim feature vector for this joint
            features = np.zeros(16)
            features[0] = 1.0 if info[2] == p.JOINT_REVOLUTE else 0.0  # Type
            features[1:4] = info[13]  # Axis
            features[4] = info[8]  # Lower limit
            features[5] = info[9]  # Upper limit
            features[6] = info[11]  # Max velocity
            features[7] = info[10]  # Max effort
            
            node_features.append(features)
            joint_indices.append(i)
    
    num_movable = len(node_features)
    
    # Build adjacency matrix (ONLY for movable joints)
    adjacency = np.zeros((num_movable, num_movable))
    
    for idx, joint_i in enumerate(joint_indices):
        info_i = p.getJointInfo(robot, joint_i, physicsClientId=client)
        parent_i = info_i[16]
        
        # Check if parent is also a movable joint
        if parent_i in joint_indices:
            parent_idx = joint_indices.index(parent_i)
            adjacency[idx, parent_idx] = 1
            adjacency[parent_idx, idx] = 1
    
    p.disconnect(physicsClientId=client)
    
    return {
        'node_features': np.array(node_features),
        'adjacency': adjacency,
        'num_joints': num_movable,
        'dof': num_movable
    }
