"""
Kinematic Tree Analyzer - Parse URDF kinematic structure properly

Identifies robot components by analyzing the kinematic tree:
- Arms: chains ending in gripper/hand/end_effector
- Legs: chains ending in foot/toe
- Torso: central body connecting limbs
- Mobile base: prismatic/continuous joints at root
"""
import pybullet as p
from typing import Dict, List, Tuple, Set
import numpy as np


class KinematicTreeAnalyzer:
    """
    Analyze URDF kinematic tree to identify robot morphology
    """
    
    # Keywords for identifying link/joint types
    ARM_KEYWORDS = ['arm', 'shoulder', 'elbow', 'wrist', 'gripper', 'hand', 'finger', 'ee', 'end_effector']
    LEG_KEYWORDS = ['leg', 'hip', 'knee', 'ankle', 'foot', 'toe', 'thigh', 'calf', 'shin']
    TORSO_KEYWORDS = ['torso', 'spine', 'chest', 'waist', 'pelvis', 'trunk', 'body']
    BASE_KEYWORDS = ['base', 'mobile', 'wheel', 'chassis']
    GRIPPER_KEYWORDS = ['gripper', 'finger', 'hand', 'palm']
    
    def __init__(self, urdf_path: str, robot_id: int, physics_client: int):
        """
        Args:
            urdf_path: Path to URDF file
            robot_id: PyBullet robot ID
            physics_client: PyBullet physics client ID
        """
        self.urdf_path = urdf_path
        self.robot_id = robot_id
        self.p = physics_client
        self.num_joints = p.getNumJoints(robot_id, physicsClientId=physics_client)
        
        # Build kinematic tree
        self.joint_info = self._get_all_joint_info()
        self.link_info = self._get_all_link_info()
        self.tree = self._build_tree()
        
    def _get_all_joint_info(self) -> List[Dict]:
        """Extract detailed joint information"""
        joints = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.p)
            joints.append({
                'index': i,
                'name': info[1].decode('utf-8').lower(),
                'type': info[2],  # 0=REVOLUTE, 1=PRISMATIC, 4=FIXED
                'parent_link_index': info[16],
                'child_link_name': info[12].decode('utf-8').lower(),
            })
        return joints
    
    def _get_all_link_info(self) -> List[Dict]:
        """Extract link information"""
        links = []
        for i in range(-1, self.num_joints):
            if i == -1:
                name = p.getBodyInfo(self.robot_id, physicsClientId=self.p)[0].decode('utf-8').lower()
            else:
                name = p.getJointInfo(self.robot_id, i, physicsClientId=self.p)[12].decode('utf-8').lower()
            
            links.append({
                'index': i,
                'name': name
            })
        return links
    
    def _build_tree(self) -> Dict[int, List[int]]:
        """
        Build kinematic tree: parent_link_index -> [child_joint_indices]
        """
        tree = {}
        for joint in self.joint_info:
            parent_idx = joint['parent_link_index']
            if parent_idx not in tree:
                tree[parent_idx] = []
            tree[parent_idx].append(joint['index'])
        return tree
    
    def _find_chains(self) -> List[List[int]]:
        """
        Find all kinematic chains (root to leaf)
        Returns list of chains, where each chain is [joint_idx, joint_idx, ...]
        """
        chains = []
        
        def traverse(joint_idx, current_chain):
            """DFS to find all paths to leaves"""
            current_chain = current_chain + [joint_idx]
            
            # Get child link index
            child_link_idx = joint_idx  # In PyBullet, joint i connects to link i
            
            # Check if this link has children
            if child_link_idx in self.tree:
                # Has children - continue traversing
                for child_joint_idx in self.tree[child_link_idx]:
                    traverse(child_joint_idx, current_chain)
            else:
                # Leaf node - save this chain
                chains.append(current_chain)
        
        # Start from base link (-1)
        if -1 in self.tree:
            for root_joint in self.tree[-1]:
                traverse(root_joint, [])
        
        return chains
    
    def _classify_chain(self, chain: List[int]) -> str:
        """
        Classify a kinematic chain as 'arm', 'leg', 'torso', or 'other'
        
        Strategy:
        1. Look at end-effector (last link) name
        2. Look at joint names in chain
        3. Count keyword matches
        """
        if len(chain) == 0:
            return 'other'
        
        # Get all names in this chain
        chain_names = []
        for joint_idx in chain:
            joint_name = self.joint_info[joint_idx]['name']
            link_name = self.joint_info[joint_idx]['child_link_name']
            chain_names.extend([joint_name, link_name])
        
        combined_names = ' '.join(chain_names)
        
        # Count keyword matches
        arm_score = sum(1 for kw in self.ARM_KEYWORDS if kw in combined_names)
        leg_score = sum(1 for kw in self.LEG_KEYWORDS if kw in combined_names)
        torso_score = sum(1 for kw in self.TORSO_KEYWORDS if kw in combined_names)
        
        # Classify based on highest score
        if arm_score > leg_score and arm_score > torso_score:
            return 'arm'
        elif leg_score > arm_score and leg_score > torso_score:
            return 'leg'
        elif torso_score > 0:
            return 'torso'
        else:
            return 'other'
    
    def _is_mobile_base(self) -> bool:
        """Check if robot has mobile base (prismatic or continuous joints at root)"""
        if -1 not in self.tree:
            return False
        
        root_joints = self.tree[-1]
        for joint_idx in root_joints[:3]:  # Check first 3 joints
            if joint_idx < len(self.joint_info):
                joint = self.joint_info[joint_idx]
                # PRISMATIC=1, CONTINUOUS=0 (some mobile bases use continuous for rotation)
                if joint['type'] in [1]:
                    return True
                if 'wheel' in joint['name'] or 'base' in joint['name']:
                    return True
        return False
    
    def analyze(self) -> Dict:
        """
        Analyze kinematic tree and identify robot structure
        
        Returns:
            {
                'type': 'single_arm' | 'dual_arm' | 'humanoid' | 'quadruped' | 'mobile_manipulator',
                'has_mobile_base': bool,
                'components': [
                    {
                        'name': 'left_arm',
                        'type': 'arm',
                        'chain': [joint_indices],
                        'dof': int
                    },
                    ...
                ]
            }
        """
        # Find all kinematic chains
        chains = self._find_chains()
        
        # Classify each chain
        components = []
        arm_count = 0
        leg_count = 0
        
        for i, chain in enumerate(chains):
            chain_type = self._classify_chain(chain)
            
            if chain_type == 'other':
                continue
            
            # Filter to controllable joints only (exclude fixed)
            controllable_chain = [
                j for j in chain 
                if j < len(self.joint_info) and self.joint_info[j]['type'] in [0, 1]
            ]
            
            if len(controllable_chain) == 0:
                continue
            
            # Name based on type and count
            if chain_type == 'arm':
                arm_count += 1
                if arm_count == 1:
                    name = 'left_arm'
                elif arm_count == 2:
                    name = 'right_arm'
                else:
                    name = f'arm_{arm_count}'
            
            elif chain_type == 'leg':
                leg_count += 1
                if leg_count == 1:
                    name = 'front_left_leg'
                elif leg_count == 2:
                    name = 'front_right_leg'
                elif leg_count == 3:
                    name = 'rear_left_leg'
                elif leg_count == 4:
                    name = 'rear_right_leg'
                else:
                    name = f'leg_{leg_count}'
            
            else:
                name = f'{chain_type}_{i}'
            
            components.append({
                'name': name,
                'type': chain_type,
                'joints': controllable_chain,
                'dof': len(controllable_chain)
            })
        
        # Determine overall robot type
        has_mobile_base = self._is_mobile_base()
        
        if leg_count == 4:
            robot_type = 'quadruped'
        elif leg_count == 2 and arm_count >= 2:
            robot_type = 'humanoid'
        elif arm_count == 2 and leg_count == 0:
            robot_type = 'dual_arm'
        elif arm_count == 1 and leg_count == 0:
            if has_mobile_base:
                robot_type = 'mobile_manipulator'
            else:
                robot_type = 'single_arm'
        elif arm_count >= 1 and has_mobile_base:
            robot_type = 'mobile_manipulator'
        else:
            robot_type = 'unknown'
        
        # Calculate total DOF
        total_dof = sum(comp['dof'] for comp in components)
        
        return {
            'type': robot_type,
            'total_dof': total_dof,
            'has_mobile_base': has_mobile_base,
            'components': components,
            'analysis': {
                'num_arms': arm_count,
                'num_legs': leg_count,
                'num_chains': len(chains)
            }
        }
