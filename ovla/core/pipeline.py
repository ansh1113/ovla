"""
COMPLETE O-VLA PIPELINE WITH STRATEGY LAYER

Full 7-layer architecture:
  Layer 0: Semantic Extractor
  Layer 0.5: Strategy Extractor (NEW)
  Layer 1: Universal Semantic Mapper
  Layer 1.5: Strategy Mapper (NEW)
  Layer 2: Constraint Extractor
  Layer 3: Hierarchical Optimizer + Whole-Body Coordinator (NEW)
  Layer 4: Trajectory Generator
"""
import numpy as np
import torch
from typing import Dict, Optional

from ovla.semantic_extractor import SemanticExtractor
from ovla.strategy_extractor import StrategyExtractor, TaskStrategy
from ovla.universal_semantic_mapper import UniversalSemanticMapper, extract_urdf_graph_data
from ovla.strategy_mapper import StrategyMapper
from ovla.constraint_extractor import GeometricConstraintNet
from ovla.hierarchical_optimizer import HierarchicalOptimizer
from ovla.whole_body_coordinator import WholeBodyCoordinator
from ovla.trajectory_generator import TrajectoryGenerator
from ovla.topology_morphology_parser import TopologyMorphologyParser


class CompleteOVLAPipeline:
    """
    COMPLETE O-VLA PIPELINE
    
    Handles ANY VLA → ANY Robot with full strategy correction.
    """
    
    def __init__(
        self,
        source_urdf: str,
        target_urdf: str,
        device: str = 'cuda'
    ):
        self.source_urdf = source_urdf
        self.target_urdf = target_urdf
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Parse morphologies
        self.source_parser = TopologyMorphologyParser(source_urdf)
        self.target_parser = TopologyMorphologyParser(target_urdf)
        
        self.source_morph = self.source_parser.get_structure()
        self.target_morph = self.target_parser.get_structure()
        
        # Layer 0: Semantic Extractor
        self.semantic_extractor = SemanticExtractor(source_urdf, verbose=False)
        
        # Layer 0.5: Strategy Extractor
        self.strategy_extractor = StrategyExtractor(self.source_morph)
        
        # Layer 1: Universal Mapper
        self.universal_mapper = UniversalSemanticMapper(
            max_dof=32, semantic_dim=256, joint_dim=256, num_attention_heads=8
        ).to(self.device)
        
        self.universal_mapper.load_state_dict(
            torch.load('/scratch/anshb3/ovla/models/universal_mapper_240k.pt')
        )
        self.universal_mapper.eval()
        
        # Layer 1.5: Strategy Mapper
        self.strategy_mapper = StrategyMapper(
            strategy_dim=64, morphology_dim=256, hidden_dim=256
        ).to(self.device)
        
        self.strategy_mapper.load_state_dict(
            torch.load('/scratch/anshb3/ovla/models/strategy_mapper_MASSIVE.pt')
        )
        self.strategy_mapper.eval()
        
        # Layer 2: Constraint Extractor
        self.constraint_net = GeometricConstraintNet(embedding_dim=128).to(self.device)
        
        # Layer 3: Hierarchical Optimizer
        self.optimizer = HierarchicalOptimizer(target_urdf, device=self.device)
        
        # Layer 3.5: Whole-Body Coordinator
        self.coordinator = WholeBodyCoordinator(target_urdf)
        
        # Layer 4: Trajectory Generator
        robot_profile = {'name': 'target', 'dof': self.target_morph['total_dof']}
        self.traj_gen = TrajectoryGenerator(robot_profile=robot_profile, control_freq=50)
        
        # Extract target URDF data
        self.target_urdf_data = extract_urdf_graph_data(target_urdf)
        
    def process(
        self,
        vla_output: np.ndarray,
        current_source_state: np.ndarray,
        action_format: str = 'joint_position',
        visual_observation: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Complete pipeline processing.
        
        Returns full results including intermediate layers.
        """
        
        results = {}
        
        # ====================================================================
        # LAYER 0: SEMANTIC EXTRACTION
        # ====================================================================
        
        semantic_action = self.semantic_extractor.extract_semantics(
            vla_action=vla_output,
            current_state=current_source_state,
            action_format=action_format
        )
        
        results['layer0'] = {
            'description': semantic_action.description,
            'fingerprint_shape': semantic_action.semantic_fingerprint.shape
        }
        
        # ====================================================================
        # LAYER 0.5: STRATEGY EXTRACTION
        # ====================================================================
        
        source_strategy = self.strategy_extractor.extract_strategy(
            semantic_action={
                'description': semantic_action.description,
                'magnitude': np.linalg.norm(vla_output - current_source_state),
                'speed': 0.5,
                'component_weights': {}
            }
        )
        
        results['layer0.5'] = {
            'strategy': source_strategy.description,
            'stability_required': source_strategy.stability_required,
            'bilateral': source_strategy.bilateral
        }
        
        # ====================================================================
        # LAYER 1: UNIVERSAL MAPPING
        # ====================================================================
        
        semantic_fp = torch.FloatTensor(semantic_action.semantic_fingerprint).unsqueeze(0).to(self.device)
        motion_emb = torch.FloatTensor(semantic_action.motion_type_embedding).unsqueeze(0).to(self.device)
        node_feat = torch.FloatTensor(self.target_urdf_data['node_features']).unsqueeze(0).to(self.device)
        adjacency = torch.FloatTensor(self.target_urdf_data['adjacency']).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            target_actions = self.universal_mapper(
                semantic_fp, motion_emb, node_feat, adjacency, self.target_urdf_data['dof']
            )
        
        target_joint_commands = target_actions[0, :self.target_urdf_data['dof']].cpu().numpy()
        
        results['layer1'] = {
            'actions_shape': target_joint_commands.shape,
            'mean': float(target_joint_commands.mean()),
            'std': float(target_joint_commands.std())
        }
        
        # ====================================================================
        # LAYER 1.5: STRATEGY MAPPING
        # ====================================================================
        
        # Encode strategies
        source_strategy_vec = np.array([
            1.0 if source_strategy.stability_required else 0.0,
            1.0 if source_strategy.bilateral else 0.0,
            1.0 if source_strategy.locomotion_required else 0.0,
            source_strategy.motion_magnitude,
            source_strategy.motion_speed,
        ] + [0.0] * 59)
        
        # Simplified morphology features
        source_morph_vec = np.zeros(256)
        source_morph_vec[0] = self.source_morph['total_dof'] / 61.0
        
        target_morph_vec = np.zeros(256)
        target_morph_vec[0] = self.target_morph['total_dof'] / 61.0
        target_morph_vec[2] = 1.0 if self.target_morph['type'] == 'humanoid' else 0.0
        
        source_strat_t = torch.FloatTensor(source_strategy_vec).unsqueeze(0).to(self.device)
        source_morph_t = torch.FloatTensor(source_morph_vec).unsqueeze(0).to(self.device)
        target_morph_t = torch.FloatTensor(target_morph_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            target_strategy_vec, corrections = self.strategy_mapper(
                source_strat_t, source_morph_t, target_morph_t
            )
        
        target_strategy_vec = target_strategy_vec[0].cpu().numpy()
        
        # Decode target strategy
        target_strategy = {
            'stability_required': target_strategy_vec[0] > 0.5,
            'bilateral': target_strategy_vec[1] > 0.5,
            'locomotion_required': target_strategy_vec[2] > 0.5,
            'motion_magnitude': float(target_strategy_vec[3]),
            'motion_speed': float(target_strategy_vec[4]),
        }
        
        results['layer1.5'] = {
            'target_strategy': target_strategy,
            'strategy_changed': float(corrections['strategy_delta'].abs().mean().item())
        }
        
        # ====================================================================
        # LAYER 2: CONSTRAINT EXTRACTION
        # ====================================================================
        
        # Mock visual embedding (in real use, from vision encoder)
        visual_emb = torch.randn(1, 128).to(self.device)
        
        with torch.no_grad():
            constraints = self.constraint_net(visual_emb)
        
        results['layer2'] = {
            'center': constraints['center_xyz'].cpu().numpy().tolist(),
            'radius': float(constraints['tolerance_r'].item())
        }
        
        # ====================================================================
        # LAYER 3: WHOLE-BODY COORDINATION + OPTIMIZATION
        # ====================================================================
        
        # Apply whole-body coordination based on strategy
        current_target_state = np.zeros(self.target_urdf_data['dof'])
        
        coordinated_actions = self.coordinator.coordinate(
            primary_action=target_joint_commands,
            strategy=target_strategy,
            current_state=current_target_state
        )
        
        # Optimize
        optimized_actions = self.optimizer.optimize(
            coordinated_actions,
            current_target_state
        )
        
        results['layer3'] = {
            'coordination_delta': float(np.abs(coordinated_actions - target_joint_commands).mean()),
            'optimization_delta': float(np.abs(optimized_actions - coordinated_actions).mean()),
            'final_actions_shape': optimized_actions.shape
        }
        
        # ====================================================================
        # LAYER 4: TRAJECTORY GENERATION
        # ====================================================================
        
        trajectory = self.traj_gen.generate_smooth_path(
            current_joints=current_target_state,
            target_joints=optimized_actions,
            duration=1.0
        )
        
        results['layer4'] = {
            'trajectory_shape': trajectory.shape,
            'control_hz': 50
        }
        
        # Final outputs
        results['final'] = {
            'trajectory': trajectory,
            'optimized_actions': optimized_actions,
            'target_robot_dof': self.target_urdf_data['dof']
        }
        
        return results
    
    def cleanup(self):
        """Clean up all PyBullet connections"""
        try:
            if hasattr(self, 'optimizer'):
                self.optimizer.cleanup()
            if hasattr(self, 'coordinator'):
                self.coordinator.cleanup()
        except:
            pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

