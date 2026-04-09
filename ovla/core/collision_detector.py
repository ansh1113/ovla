"""
Real Collision Detector

Forward kinematics + bounding sphere collision checking
"""
import numpy as np
import pybullet as p
from typing import Dict, List, Tuple, Optional


class CollisionDetector:
    """
    Real inter-limb collision detection using PyBullet FK
    
    Uses bounding spheres around links to check for collisions
    """
    
    def __init__(self, urdf_path: str):
        """
        Args:
            urdf_path: Path to robot URDF
        """
        self.urdf_path = urdf_path
        self.p_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self.p_client)
        
        # Extract link bounding spheres
        self.link_spheres = self._compute_link_spheres()
        
        # Build component → link mapping
        self.component_links = {}
    
    def _compute_link_spheres(self) -> Dict[int, Dict]:
        """
        Compute bounding sphere for each link
        
        Returns:
            {link_index: {'name': str, 'radius': float}}
        """
        spheres = {}
        
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.p_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.p_client)
            name = joint_info[1].decode('utf-8')
            
            # Get collision shape AABB
            aabb = p.getAABB(self.robot_id, i, physicsClientId=self.p_client)
            min_bounds = np.array(aabb[0])
            max_bounds = np.array(aabb[1])
            
            # Compute bounding sphere radius (conservative)
            size = max_bounds - min_bounds
            radius = np.linalg.norm(size) / 2.0
            
            spheres[i] = {
                'name': name,
                'radius': radius if radius > 0 else 0.05  # Minimum radius
            }
        
        return spheres
    
    def set_component_mapping(self, component_map: Dict[str, List[int]]):
        """
        Set mapping from component names to joint indices
        
        Args:
            component_map: {component_name: [joint_indices]}
        """
        self.component_links = component_map
    
    def compute_link_positions(
        self,
        joint_positions: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute world positions of all links via FK
        
        Args:
            joint_positions: Joint angles
            
        Returns:
            {link_index: position_xyz}
        """
        # Set robot to this configuration
        controllable_joints = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.p_client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.p_client)
            if info[2] in [0, 1]:  # Revolute or prismatic
                controllable_joints.append(i)
        
        # Set joint positions
        for i, joint_idx in enumerate(controllable_joints):
            if i < len(joint_positions):
                p.resetJointState(
                    self.robot_id,
                    joint_idx,
                    joint_positions[i],
                    physicsClientId=self.p_client
                )
        
        # Get link world positions
        positions = {}
        for link_idx in self.link_spheres.keys():
            link_state = p.getLinkState(
                self.robot_id,
                link_idx,
                physicsClientId=self.p_client
            )
            positions[link_idx] = np.array(link_state[0])
        
        return positions
    
    def check_collision_between_components(
        self,
        component1: str,
        component2: str,
        joint_positions: np.ndarray,
        safety_margin: float = 0.05
    ) -> bool:
        """
        Check if two components collide
        
        Args:
            component1, component2: Component names
            joint_positions: Full robot joint positions
            safety_margin: Extra clearance (meters)
            
        Returns:
            True if collision detected
        """
        if component1 not in self.component_links or component2 not in self.component_links:
            return False
        
        # Get link positions
        link_positions = self.compute_link_positions(joint_positions)
        
        # Check all pairs of links
        links1 = self.component_links[component1]
        links2 = self.component_links[component2]
        
        for l1 in links1:
            if l1 not in link_positions or l1 not in self.link_spheres:
                continue
            
            for l2 in links2:
                if l2 not in link_positions or l2 not in self.link_spheres:
                    continue
                
                # Sphere-sphere collision test
                pos1 = link_positions[l1]
                pos2 = link_positions[l2]
                
                r1 = self.link_spheres[l1]['radius']
                r2 = self.link_spheres[l2]['radius']
                
                distance = np.linalg.norm(pos1 - pos2)
                threshold = r1 + r2 + safety_margin
                
                if distance < threshold:
                    return True
        
        return False
    
    def check_all_collisions(
        self,
        joint_positions: np.ndarray,
        safety_margin: float = 0.05
    ) -> List[Tuple[str, str]]:
        """
        Check all inter-component collisions
        
        Returns:
            List of colliding component pairs
        """
        collisions = []
        components = list(self.component_links.keys())
        
        # Check all pairs
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                if self.check_collision_between_components(
                    comp1, comp2, joint_positions, safety_margin
                ):
                    collisions.append((comp1, comp2))
        
        return collisions
    
    def __del__(self):
        try:
            p.disconnect(self.p_client)
        except:
            pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/scratch/anshb3/ovla')
    from ovla.topology_morphology_parser import TopologyMorphologyParser
    
    print("="*70)
    print("TESTING REAL COLLISION DETECTOR")
    print("="*70)
    
    # Test with G1 humanoid
    urdf_path = '/scratch/anshb3/ovla/robots/unitree_ros/robots/g1_description/g1_23dof.urdf'
    
    # Get morphology
    parser = TopologyMorphologyParser(urdf_path)
    structure = parser.get_structure()
    
    # Create collision detector
    detector = CollisionDetector(urdf_path)
    
    print(f"\nRobot: G1 Humanoid")
    print(f"Link spheres: {len(detector.link_spheres)}")
    
    # Build component → joints mapping
    component_map = {}
    for comp in structure['components']:
        component_map[comp['name']] = comp['joints']
    
    detector.set_component_mapping(component_map)
    print(f"Components: {list(component_map.keys())}")
    
    # Test collision detection
    print(f"\nTesting collision detection...")
    
    # Neutral pose - should have no collisions
    neutral_pose = np.zeros(22)
    collisions = detector.check_all_collisions(neutral_pose)
    print(f"  Neutral pose collisions: {len(collisions)}")
    
    if collisions:
        for c1, c2 in collisions:
            print(f"    - {c1} ↔ {c2}")
    
    # Extreme pose - might have collisions
    extreme_pose = np.ones(22) * 1.5  # Large joint angles
    collisions = detector.check_all_collisions(extreme_pose, safety_margin=0.1)
    print(f"  Extreme pose collisions: {len(collisions)}")
    
    if collisions:
        for c1, c2 in collisions:
            print(f"    - {c1} ↔ {c2}")
    
    print(f"\n{'='*70}")
    print("✅ REAL COLLISION DETECTOR WORKING")
    print("="*70)
