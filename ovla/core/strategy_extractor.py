"""
STRATEGY EXTRACTOR - Layer 1.5

Extracts HIGH-LEVEL task strategy from semantic actions.
Goes beyond "wrist flexion" to understand "reaching while maintaining balance"

Key Innovation:
- Low-level: "Move joints X, Y, Z"
- High-level: "Reach forward while maintaining stability"

This bridges the gap between motion and strategy.
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TaskStrategy:
    """High-level task strategy representation"""
    
    # Primary goal
    primary_goal: str  # "reach", "grasp", "navigate", "manipulate"
    
    # Secondary constraints
    stability_required: bool  # Does this need balance?
    bilateral: bool  # Does this use both sides?
    locomotion_required: bool  # Does this need base movement?
    
    # Coordination requirements
    components_involved: List[str]  # ["left_arm", "right_leg", "torso"]
    coordination_type: str  # "sequential", "parallel", "coordinated"
    
    # Motion characteristics
    motion_magnitude: float  # How big is the motion?
    motion_speed: float  # How fast?
    precision_required: bool  # High precision task?
    
    # Workspace
    workspace_region: str  # "far", "near", "overhead", "ground"
    
    # Strategy description
    description: str


class StrategyExtractor:
    """
    Extracts high-level task strategy from semantic action.
    
    Works with ANY robot morphology by analyzing:
    1. Which components are active
    2. What they're doing
    3. What constraints are implied
    """
    
    def __init__(self, robot_morphology: Dict):
        """
        Args:
            robot_morphology: Dict with robot structure info
                - type: 'arm', 'humanoid', 'quadruped', etc.
                - components: List of component dicts
                - total_dof: int
        """
        self.morphology = robot_morphology
        
    def extract_strategy(
        self,
        semantic_action: Dict,
        current_state: Optional[np.ndarray] = None
    ) -> TaskStrategy:
        """
        Extract high-level strategy from semantic action.
        
        Args:
            semantic_action: Output from SemanticExtractor
            current_state: Current robot state (optional)
            
        Returns:
            TaskStrategy describing high-level intent
        """
        
        # Analyze what components are moving
        active_components = self._identify_active_components(semantic_action)
        
        # Determine primary goal
        primary_goal = self._infer_primary_goal(semantic_action, active_components)
        
        # Check stability requirements
        stability_required = self._requires_stability(active_components)
        
        # Check if bilateral
        bilateral = self._is_bilateral(active_components)
        
        # Check locomotion
        locomotion_required = self._requires_locomotion(semantic_action)
        
        # Coordination type
        coordination_type = self._determine_coordination(active_components)
        
        # Motion characteristics
        motion_magnitude = semantic_action.get('magnitude', 0.0)
        motion_speed = semantic_action.get('speed', 0.0)
        precision_required = motion_magnitude < 0.05  # Small motions = precision
        
        # Workspace
        workspace_region = self._determine_workspace(semantic_action)
        
        # Generate description
        description = self._generate_strategy_description(
            primary_goal, active_components, stability_required,
            locomotion_required, coordination_type
        )
        
        return TaskStrategy(
            primary_goal=primary_goal,
            stability_required=stability_required,
            bilateral=bilateral,
            locomotion_required=locomotion_required,
            components_involved=active_components,
            coordination_type=coordination_type,
            motion_magnitude=motion_magnitude,
            motion_speed=motion_speed,
            precision_required=precision_required,
            workspace_region=workspace_region,
            description=description
        )
    
    def _identify_active_components(self, semantic_action: Dict) -> List[str]:
        """Which robot components are involved?"""
        
        active = []
        
        # Check component activation weights
        if 'component_weights' in semantic_action:
            for comp, weight in semantic_action['component_weights'].items():
                if weight > 0.1:  # Threshold for "active"
                    active.append(comp)
        
        return active
    
    def _infer_primary_goal(
        self,
        semantic_action: Dict,
        active_components: List[str]
    ) -> str:
        """What is the main goal?"""
        
        description = semantic_action.get('description', '').lower()
        
        # Keyword matching
        if 'reach' in description or 'extend' in description:
            return 'reach'
        elif 'grasp' in description or 'grip' in description or 'close' in description:
            return 'grasp'
        elif 'place' in description or 'release' in description:
            return 'place'
        elif 'walk' in description or 'step' in description:
            return 'locomotion'
        elif 'turn' in description or 'rotate' in description:
            return 'reorient'
        else:
            # Default: reaching
            return 'reach'
    
    def _requires_stability(self, active_components: List[str]) -> bool:
        """Does this task require balance/stability?"""
        
        robot_type = self.morphology.get('type', '')
        
        # Humanoids and bipeds always need stability
        if robot_type in ['humanoid', 'biped']:
            return True
        
        # Quadrupeds need stability if lifting a leg
        if robot_type == 'quadruped':
            leg_components = [c for c in active_components if 'leg' in c.lower()]
            return len(leg_components) > 0
        
        # Arms don't need stability
        return False
    
    def _is_bilateral(self, active_components: List[str]) -> bool:
        """Are both sides involved?"""
        
        left_active = any('left' in c.lower() for c in active_components)
        right_active = any('right' in c.lower() for c in active_components)
        
        return left_active and right_active
    
    def _requires_locomotion(self, semantic_action: Dict) -> bool:
        """Does this need base movement?"""
        
        # Check if base is translating
        base_motion = semantic_action.get('base_translation', 0.0)
        return base_motion > 0.01
    
    def _determine_coordination(self, active_components: List[str]) -> str:
        """What type of coordination is needed?"""
        
        if len(active_components) == 0:
            return 'none'
        elif len(active_components) == 1:
            return 'single'
        elif len(active_components) == 2:
            return 'bilateral'
        else:
            return 'whole_body'
    
    def _determine_workspace(self, semantic_action: Dict) -> str:
        """Where is the action happening?"""
        
        # Check end-effector position if available
        ee_pos = semantic_action.get('end_effector_position', None)
        
        if ee_pos is not None:
            x, y, z = ee_pos
            
            # Far/near
            distance = np.sqrt(x**2 + y**2)
            if distance > 0.5:
                return 'far'
            elif distance < 0.2:
                return 'near'
            
            # Height
            if z > 0.5:
                return 'overhead'
            elif z < 0.2:
                return 'ground'
        
        return 'reachable'
    
    def _generate_strategy_description(
        self,
        goal: str,
        components: List[str],
        stability: bool,
        locomotion: bool,
        coordination: str
    ) -> str:
        """Generate human-readable strategy description"""
        
        parts = []
        
        # Goal
        parts.append(f"Primary goal: {goal}")
        
        # Components
        if components:
            parts.append(f"Using: {', '.join(components)}")
        
        # Coordination
        if coordination == 'whole_body':
            parts.append("Whole-body coordination required")
        elif coordination == 'bilateral':
            parts.append("Bilateral coordination")
        
        # Constraints
        constraints = []
        if stability:
            constraints.append("maintain stability")
        if locomotion:
            constraints.append("include locomotion")
        
        if constraints:
            parts.append(f"Must: {', '.join(constraints)}")
        
        return "; ".join(parts)
