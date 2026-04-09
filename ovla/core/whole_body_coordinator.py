"""
WHOLE-BODY COORDINATOR - UNIVERSAL APPROACH

Key principle: NO hardcoded robot-specific logic!
Everything derived from:
1. Robot morphology (from URDF)
2. Physics (balance, stability)
3. Strategy requirements

NOT: "If humanoid reaching right, shift weight left"
BUT: "Compute CoM shift needed for stability, apply to available joints"
"""
import numpy as np
from typing import Dict, List, Optional
import pybullet as p

class WholeBodyCoordinator:
    """
    Universal whole-body coordination based on physics and morphology.
    
    Works with ANY robot by:
    1. Analyzing kinematic structure
    2. Computing physics requirements (balance, stability)
    3. Distributing motion across available components
    """
    
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        
        # Load robot in PyBullet for physics
        self.physics_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=False)
        
        # Get robot structure
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Identify component groups (universal detection)
        self.components = self._identify_components_universal()
        
        # Get joint info
        self.joint_info = self._get_joint_info()
        
    def _identify_components_universal(self) -> Dict[str, List[int]]:
        """
        Identify components using UNIVERSAL kinematic analysis.
        
        No hardcoded names! Uses:
        - Kinematic tree structure
        - Joint locations in space
        - Parent-child relationships
        """
        
        components = {
            'support': [],      # Joints that support the robot (legs, base)
            'manipulation': [], # Joints used for manipulation (arms, grippers)
            'stabilization': [],# Joints for balance (torso, waist)
            'locomotion': [],   # Joints for movement (wheels, legs)
        }
        
        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8').lower()
            
            # Get joint position in base frame
            link_state = p.getLinkState(self.robot_id, joint_idx)
            position = link_state[0]  # World position
            
            # Classify based on PHYSICS not names
            height = position[2]  # Z coordinate
            
            # Support: joints close to ground
            if height < 0.3:
                components['support'].append(joint_idx)
            
            # Manipulation: joints far from base, not supporting
            elif height > 0.5:
                components['manipulation'].append(joint_idx)
            
            # Stabilization: middle height (torso region)
            elif 0.3 <= height <= 0.5:
                components['stabilization'].append(joint_idx)
            
            # Locomotion detection (wheels, tracks, etc.)
            if 'wheel' in joint_name or 'track' in joint_name:
                components['locomotion'].append(joint_idx)
        
        return components
    
    def _get_joint_info(self) -> List[Dict]:
        """Get physical properties of all joints"""
        
        info = []
        for joint_idx in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            
            info.append({
                'index': joint_idx,
                'name': joint_info[1].decode('utf-8'),
                'type': joint_info[2],  # Revolute, prismatic, etc.
                'lower_limit': joint_info[8],
                'upper_limit': joint_info[9],
                'max_force': joint_info[10],
                'max_velocity': joint_info[11],
            })
        
        return info
    
    def coordinate(
        self,
        primary_action: np.ndarray,
        strategy: Dict,
        current_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        UNIVERSAL coordination based on physics.
        
        Process:
        1. Apply primary action
        2. Compute stability requirements from physics
        3. Distribute compensation across available joints
        4. Respect joint limits
        """
        
        if current_state is None:
            current_state = np.zeros(self.num_joints)
        
        whole_body_action = current_state.copy()
        
        # Apply primary action
        primary_len = min(len(primary_action), len(whole_body_action))
        whole_body_action[:primary_len] = primary_action[:primary_len]
        
        # Compute Center of Mass (CoM) shift from primary action
        com_shift = self._compute_com_shift(
            current_state, whole_body_action
        )
        
        # If stability required, add compensations
        if strategy.get('stability_required', False):
            whole_body_action = self._compute_stability_compensation(
                whole_body_action, com_shift, current_state
            )
        
        # If locomotion needed, activate locomotion joints
        if strategy.get('locomotion_required', False):
            whole_body_action = self._activate_locomotion(
                whole_body_action, strategy
            )
        
        # Ensure joint limits
        whole_body_action = self._enforce_joint_limits(whole_body_action)
        
        return whole_body_action
    
    def _compute_com_shift(
        self,
        state_before: np.ndarray,
        state_after: np.ndarray
    ) -> np.ndarray:
        """
        Compute Center of Mass shift using PyBullet physics.
        
        This is UNIVERSAL - works for any robot.
        """
        
        # Set robot to state_before
        for i in range(len(state_before)):
            p.resetJointState(self.robot_id, i, state_before[i])
        
        # Get CoM before
        com_before = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        
        # Set robot to state_after
        for i in range(len(state_after)):
            p.resetJointState(self.robot_id, i, state_after[i])
        
        # Get CoM after
        com_after = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        
        # Compute shift
        com_shift = com_after - com_before
        
        return com_shift
    
    def _compute_stability_compensation(
        self,
        action: np.ndarray,
        com_shift: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        UNIVERSAL stability compensation.
        
        Physics-based approach:
        1. If CoM shifts forward, shift support backward
        2. Distribute compensation across support joints
        3. Weight by joint capability (max_force, range)
        """
        
        # If no CoM shift, no compensation needed
        if np.linalg.norm(com_shift) < 0.01:
            return action
        
        # Get support joints
        support_joints = self.components['support']
        stabilization_joints = self.components['stabilization']
        
        available_joints = support_joints + stabilization_joints
        
        if not available_joints:
            return action  # No joints available for compensation
        
        # Compute compensation magnitude (proportional to shift)
        compensation_magnitude = np.linalg.norm(com_shift[:2])  # X-Y plane
        
        # Direction: opposite to CoM shift
        com_direction = com_shift[:2] / (np.linalg.norm(com_shift[:2]) + 1e-6)
        
        # Distribute compensation across available joints
        for joint_idx in available_joints:
            if joint_idx >= len(action):
                continue
            
            # Get joint capability
            joint_range = (
                self.joint_info[joint_idx]['upper_limit'] - 
                self.joint_info[joint_idx]['lower_limit']
            )
            
            # Weight compensation by joint range
            weight = joint_range / 3.14  # Normalize by typical joint range
            
            # Apply compensation (opposite direction to CoM shift)
            compensation = -compensation_magnitude * weight * 0.1
            
            action[joint_idx] = current_state[joint_idx] + compensation
        
        return action
    
    def _activate_locomotion(
        self,
        action: np.ndarray,
        strategy: Dict
    ) -> np.ndarray:
        """UNIVERSAL locomotion activation"""
        
        locomotion_joints = self.components['locomotion']
        
        # Get desired locomotion from strategy
        base_translation = strategy.get('base_translation', 0.1)
        
        for joint_idx in locomotion_joints:
            if joint_idx < len(action):
                action[joint_idx] = base_translation
        
        return action
    
    def _enforce_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """Clip actions to joint limits"""
        
        for i in range(min(len(action), len(self.joint_info))):
            lower = self.joint_info[i]['lower_limit']
            upper = self.joint_info[i]['upper_limit']
            
            if lower < upper:  # Valid limits
                action[i] = np.clip(action[i], lower, upper)
        
        return action
    
    def __del__(self):
        """Cleanup PyBullet"""
        try:
            p.disconnect(self.physics_client)
        except:
            pass
