"""
SEMANTIC EXTRACTOR V2 - Foundation of Universal O-VLA

Purpose: Extract robot-agnostic CONTINUOUS semantic descriptions from ANY VLA output

Key Philosophy: NO FIXED CATEGORIES. Pure continuous kinematic description.
"""
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class JointSemantics:
    """Semantic understanding of a single joint"""
    joint_name: str
    joint_index: int
    joint_type: str
    axis: np.ndarray
    limits: Tuple[float, float]
    max_velocity: float
    max_effort: float
    role_label: str
    component_label: str
    workspace_contribution: float
    dof_index: int

@dataclass
class SemanticAction:
    """CONTINUOUS semantic representation of robot motion"""
    active_joints: List[str]
    joint_positions_current: Dict[str, float]
    joint_positions_target: Dict[str, float]
    joint_deltas: Dict[str, float]
    joint_velocities: Dict[str, float]
    component_activations: Dict[str, float]
    has_end_effector: bool
    ee_pose_current: Optional[np.ndarray]
    ee_pose_target: Optional[np.ndarray]
    ee_translation_delta: Optional[np.ndarray]
    ee_rotation_delta: Optional[np.ndarray]
    ee_velocity_linear: Optional[np.ndarray]
    ee_velocity_angular: Optional[np.ndarray]
    motion_magnitude: float
    motion_magnitude_ee: float
    motion_direction: np.ndarray
    motion_speed: float
    motion_type_embedding: np.ndarray
    workspace_center: np.ndarray
    workspace_extent: float
    workspace_region_embedding: np.ndarray
    expected_contact: bool
    force_direction: Optional[np.ndarray]
    force_magnitude: Optional[float]
    estimated_duration: float
    motion_smoothness: float
    semantic_fingerprint: np.ndarray
    description: str
    confidence: float
    extraction_metadata: Dict

class SemanticExtractor:
    """Extract continuous semantic meaning from VLA outputs"""
    
    def __init__(self, urdf_path: str, verbose: bool = False):
        self.urdf_path = urdf_path
        self.verbose = verbose
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load robot
        self.robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, 
                               physicsClientId=self.physics_client)
        
        # Extract structure
        self.num_joints = p.getNumJoints(self.robot, physicsClientId=self.physics_client)
        self.joint_info = self._extract_joint_info()
        
        # CRITICAL FIX: Find EE BEFORE computing joint semantics
        self.ee_link_index = self._find_end_effector()
        
        # Now compute joint semantics (needs ee_link_index)
        self.joint_semantics = self._compute_joint_semantics()
        
        # Build components
        self.components = self._identify_components()
        
        if self.verbose:
            print(f"✓ Semantic Extractor initialized")
            print(f"  URDF: {urdf_path}")
            print(f"  Joints: {self.num_joints}")
            print(f"  End-effector: {self.ee_link_index}")
            print(f"  Components: {list(self.components.keys())}")
    
    def _extract_joint_info(self) -> List[Dict]:
        """Extract joint information from URDF"""
        joints = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.physics_client)
            
            if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joints.append({
                    'index': i,
                    'name': info[1].decode('utf-8'),
                    'type': info[2],
                    'lower_limit': info[8],
                    'upper_limit': info[9],
                    'max_velocity': info[11],
                    'max_effort': info[10],
                    'axis': info[13],
                })
        return joints
    
    def _find_end_effector(self) -> Optional[int]:
        """Find end-effector link"""
        for i in range(self.num_joints - 1, -1, -1):
            joint_info = p.getJointInfo(self.robot, i, physicsClientId=self.physics_client)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                return i
        return None
    
    def _compute_joint_semantics(self) -> Dict[int, JointSemantics]:
        """Compute semantic understanding of each joint"""
        semantics = {}
        
        for joint in self.joint_info:
            role = self._classify_joint_role(joint['name'])
            component = self._classify_joint_component(joint['name'])
            workspace_contrib = self._compute_workspace_contribution(joint['index'])
            
            semantics[joint['index']] = JointSemantics(
                joint_name=joint['name'],
                joint_index=joint['index'],
                joint_type='revolute' if joint['type'] == p.JOINT_REVOLUTE else 'prismatic',
                axis=np.array(joint['axis']),
                limits=(joint['lower_limit'], joint['upper_limit']),
                max_velocity=joint['max_velocity'],
                max_effort=joint['max_effort'],
                role_label=role,
                component_label=component,
                workspace_contribution=workspace_contrib,
                dof_index=len(semantics)
            )
        
        return semantics
    
    def _classify_joint_role(self, joint_name: str) -> str:
        """Heuristic joint role classification"""
        name_lower = joint_name.lower()
        
        if any(x in name_lower for x in ['wrist', 'hand']):
            return 'wrist'
        elif 'elbow' in name_lower:
            return 'elbow'
        elif 'shoulder' in name_lower:
            return 'shoulder'
        elif 'ankle' in name_lower:
            return 'ankle'
        elif 'knee' in name_lower:
            return 'knee'
        elif 'hip' in name_lower:
            return 'hip'
        elif any(x in name_lower for x in ['waist', 'torso', 'spine', 'pelvis']):
            return 'torso'
        elif any(x in name_lower for x in ['finger', 'gripper']):
            return 'end_effector'
        
        return 'other'
    
    def _classify_joint_component(self, joint_name: str) -> str:
        """Heuristic component classification"""
        name_lower = joint_name.lower()
        
        if any(x in name_lower for x in ['left', '_l_', 'fl_']):
            if any(x in name_lower for x in ['arm', 'shoulder', 'elbow', 'wrist']):
                return 'left_arm'
            elif any(x in name_lower for x in ['leg', 'hip', 'knee', 'ankle']):
                return 'left_leg'
        
        if any(x in name_lower for x in ['right', '_r_', 'fr_']):
            if any(x in name_lower for x in ['arm', 'shoulder', 'elbow', 'wrist']):
                return 'right_arm'
            elif any(x in name_lower for x in ['leg', 'hip', 'knee', 'ankle']):
                return 'right_leg'
        
        if any(x in name_lower for x in ['rear', 'rr_', 'rl_', 'back']):
            return 'rear_leg'
        
        if any(x in name_lower for x in ['torso', 'waist', 'spine', 'base', 'pelvis']):
            return 'torso'
        
        return 'other'
    
    def _compute_workspace_contribution(self, joint_index: int) -> float:
        """Compute how much this joint affects workspace"""
        if self.ee_link_index is None:
            return 0.0
        
        joint_state = p.getJointState(self.robot, joint_index, physicsClientId=self.physics_client)
        original_pos = joint_state[0]
        
        link_state = p.getLinkState(self.robot, self.ee_link_index, physicsClientId=self.physics_client)
        current_ee_pos = np.array(link_state[0])
        
        delta = 0.05
        p.resetJointState(self.robot, joint_index, original_pos + delta, physicsClientId=self.physics_client)
        
        new_link_state = p.getLinkState(self.robot, self.ee_link_index, physicsClientId=self.physics_client)
        new_ee_pos = np.array(new_link_state[0])
        
        displacement = np.linalg.norm(new_ee_pos - current_ee_pos)
        
        p.resetJointState(self.robot, joint_index, original_pos, physicsClientId=self.physics_client)
        
        return displacement / delta
    
    def _identify_components(self) -> Dict[str, List[int]]:
        """Identify kinematic components"""
        components = {}
        for joint_idx, sem in self.joint_semantics.items():
            comp = sem.component_label
            if comp not in components:
                components[comp] = []
            components[comp].append(joint_idx)
        return components
    


    def detokenize_action(self, tokenized_action):
        """
        Convert tokenized actions [0-255] to continuous values
        
        Used for RT-1, RT-2 style VLAs
        Assumes standard joint range [-π, π] for most joints
        """
        import numpy as np
        
        continuous = []
        for i, token in enumerate(tokenized_action):
            # Normalize token to [0, 1]
            normalized = token / 255.0
            
            # For most robot joints, assume [-π, π] range
            # This is a reasonable default for revolute joints
            min_val = -np.pi
            max_val = np.pi
            
            # Scale to joint range
            value = normalized * (max_val - min_val) + min_val
            continuous.append(value)
        
        return np.array(continuous)


    def _add_morphology_context(self, base_description, robot_urdf_path):
        """
        Add morphology-specific context to semantic description
        """
        import os
        
        # Detect morphology from URDF path or structure
        urdf_name = os.path.basename(robot_urdf_path).lower()
        
        morphology_vocab = {
            'snake': {
                'keywords': ['snake', 'serpent', 'slither'],
                'joints': 'body segments',
                'motion': 'undulating',
                'workspace': 'sinusoidal path',
            },
            'hexapod': {
                'keywords': ['hexapod', 'hex', 'insect'],
                'joints': 'leg joints',
                'motion': 'multi-leg coordination',
                'workspace': 'stable platform',
            },
            'quadruped': {
                'keywords': ['quadruped', 'quad', 'dog', 'laikago'],
                'joints': 'leg joints',
                'motion': 'gait pattern',
                'workspace': 'ground-based',
            },
            'wheeled': {
                'keywords': ['wheel', 'mobile', 'base'],
                'joints': 'wheels and manipulator',
                'motion': 'mobile manipulation',
                'workspace': 'planar motion',
            },
            'humanoid': {
                'keywords': ['humanoid', 'human', 'g1', 'atlas'],
                'joints': 'anthropomorphic joints',
                'motion': 'bipedal',
                'workspace': 'human-like reach',
            }
        }
        
        # Detect morphology
        detected = None
        for morph, config in morphology_vocab.items():
            if any(kw in urdf_name for kw in config['keywords']):
                detected = morph
                break
        
        if detected:
            vocab = morphology_vocab[detected]
            
            # Replace generic terms with specific ones
            description = base_description
            description = description.replace('joints', vocab['joints'])
            
            # Add morphology-specific prefix
            if 'Moving' in description:
                description = f"{vocab['motion'].capitalize()} - " + description
            
            return description
        
        return base_description

    def extract_semantics(self, vla_action: np.ndarray, current_state: np.ndarray,
                         action_format: str = 'joint_position', dt: float = 0.1) -> SemanticAction:
        """Extract continuous semantic representation"""
        
        self._set_robot_state(current_state)
        
        joint_data = self._analyze_joint_space(vla_action, current_state, action_format, dt)
        ee_data = self._analyze_end_effector(vla_action, current_state, action_format, dt)
        motion_data = self._analyze_motion_characteristics(joint_data, ee_data)
        workspace_data = self._analyze_workspace(ee_data, joint_data)
        component_activations = self._compute_component_activations(joint_data['velocities'])
        motion_embedding = self._compute_motion_type_embedding(joint_data, ee_data, motion_data)
        semantic_fingerprint = self._compute_semantic_fingerprint(joint_data, ee_data, motion_data, workspace_data, motion_embedding)
        description = self._generate_description(joint_data, ee_data, motion_data, workspace_data)
        confidence = self._compute_confidence(joint_data, ee_data)
        
        return SemanticAction(
            active_joints=joint_data['active_joints'],
            joint_positions_current=joint_data['positions_current'],
            joint_positions_target=joint_data['positions_target'],
            joint_deltas=joint_data['deltas'],
            joint_velocities=joint_data['velocities'],
            component_activations=component_activations,
            has_end_effector=ee_data['has_ee'],
            ee_pose_current=ee_data['pose_current'],
            ee_pose_target=ee_data['pose_target'],
            ee_translation_delta=ee_data['translation_delta'],
            ee_rotation_delta=ee_data['rotation_delta'],
            ee_velocity_linear=ee_data['velocity_linear'],
            ee_velocity_angular=ee_data['velocity_angular'],
            motion_magnitude=motion_data['magnitude'],
            motion_magnitude_ee=motion_data['magnitude_ee'],
            motion_direction=motion_data['direction'],
            motion_speed=motion_data['speed'],
            motion_type_embedding=motion_embedding,
            workspace_center=workspace_data['center'],
            workspace_extent=workspace_data['extent'],
            workspace_region_embedding=workspace_data['region_embedding'],
            expected_contact=False,
            force_direction=None,
            force_magnitude=None,
            estimated_duration=dt,
            motion_smoothness=1.0,
            semantic_fingerprint=semantic_fingerprint,
            description=description,
            confidence=confidence,
            extraction_metadata={'action_format': action_format, 'source_urdf': self.urdf_path, 'dt': dt}
        )
    
    def _set_robot_state(self, joint_positions: np.ndarray):
        """Set robot to given state"""
        for i, pos in enumerate(joint_positions):
            if i < self.num_joints:
                p.resetJointState(self.robot, i, pos, physicsClientId=self.physics_client)
    
    def _analyze_joint_space(self, vla_action, current_state, action_format, dt):
        """Analyze joint space motion"""
        
        # Auto-detect tokenized actions (RT-1, RT-2 style)
        if vla_action.dtype == np.uint8 or (vla_action.max() > 10 and vla_action.max() <= 255 and vla_action.min() >= 0):
            # Likely tokenized action
            vla_action = self.detokenize_action(vla_action)
            action_format = 'joint_position'  # After detokenization

        if action_format == 'joint_position':
            target_positions = vla_action
        elif action_format == 'joint_velocity':
            target_positions = current_state + vla_action * dt
        else:
            target_positions = current_state
        
        deltas = target_positions - current_state
        velocities = deltas / dt
        
        joint_data = {'positions_current': {}, 'positions_target': {}, 'deltas': {}, 
                     'velocities': {}, 'active_joints': []}
        
        for i, sem in self.joint_semantics.items():
            if i < len(current_state):
                joint_data['positions_current'][sem.joint_name] = float(current_state[i])
                joint_data['positions_target'][sem.joint_name] = float(target_positions[i])
                joint_data['deltas'][sem.joint_name] = float(deltas[i])
                joint_data['velocities'][sem.joint_name] = float(velocities[i])
                
                if abs(deltas[i]) > 0.01:
                    joint_data['active_joints'].append(sem.joint_name)
        
        return joint_data
    
    def _analyze_end_effector(self, vla_action, current_state, action_format, dt):
        """Analyze end-effector motion"""
        ee_data = {'has_ee': self.ee_link_index is not None, 'pose_current': None, 
                  'pose_target': None, 'translation_delta': None, 'rotation_delta': None,
                  'velocity_linear': None, 'velocity_angular': None}
        
        if self.ee_link_index is None:
            return ee_data
        
        self._set_robot_state(current_state)
        link_state = p.getLinkState(self.robot, self.ee_link_index, physicsClientId=self.physics_client)
        ee_current = np.array(list(link_state[0]) + list(link_state[1]))
        ee_data['pose_current'] = ee_current
        
        if action_format == 'end_effector_pose':
            ee_target = vla_action[:7]
        else:
            target_positions = vla_action if action_format == 'joint_position' else current_state + vla_action * dt
            self._set_robot_state(target_positions)
            link_state = p.getLinkState(self.robot, self.ee_link_index, physicsClientId=self.physics_client)
            ee_target = np.array(list(link_state[0]) + list(link_state[1]))
        
        ee_data['pose_target'] = ee_target
        ee_data['translation_delta'] = ee_target[:3] - ee_current[:3]
        ee_data['rotation_delta'] = np.array([0.0, 0.0, 0.0])  # TODO: proper quat math
        ee_data['velocity_linear'] = ee_data['translation_delta'] / dt
        ee_data['velocity_angular'] = np.array([0.0, 0.0, 0.0])
        
        self._set_robot_state(current_state)
        return ee_data
    
    def _analyze_motion_characteristics(self, joint_data, ee_data):
        """Analyze overall motion characteristics"""
        velocities = np.array(list(joint_data['velocities'].values()))
        magnitude = np.linalg.norm(velocities)
        direction = velocities / (magnitude + 1e-8)
        magnitude_ee = np.linalg.norm(ee_data['translation_delta']) if ee_data['has_ee'] and ee_data['translation_delta'] is not None else 0.0
        speed = magnitude_ee if ee_data['has_ee'] else magnitude
        
        return {'magnitude': magnitude, 'magnitude_ee': magnitude_ee, 'direction': direction, 'speed': speed}
    
    def _analyze_workspace(self, ee_data, joint_data):
        """Analyze workspace characteristics"""
        if ee_data['has_ee'] and ee_data['pose_current'] is not None:
            center = ee_data['pose_current'][:3]
            extent = np.linalg.norm(ee_data['translation_delta']) if ee_data['translation_delta'] is not None else 0.0
            region_embedding = center / (np.linalg.norm(center) + 1e-8)
        else:
            center = np.zeros(3)
            extent = 0.0
            region_embedding = np.zeros(3)
        
        return {'center': center, 'extent': extent, 'region_embedding': region_embedding}
    
    def _compute_component_activations(self, joint_velocities):
        """Compute continuous activation level for each component"""
        activations = {}
        
        for comp_name, joint_indices in self.components.items():
            total_activation = 0.0
            for joint_idx in joint_indices:
                if joint_idx in self.joint_semantics:
                    joint_name = self.joint_semantics[joint_idx].joint_name
                    if joint_name in joint_velocities:
                        total_activation += abs(joint_velocities[joint_name])
            activations[comp_name] = min(total_activation, 1.0)
        
        total = sum(activations.values())
        if total > 0:
            activations = {k: v/total for k, v in activations.items()}
        
        return activations
    
    def _compute_motion_type_embedding(self, joint_data, ee_data, motion_data):
        """Compute continuous motion type embedding"""
        embedding = np.zeros(16)
        
        if ee_data['has_ee'] and ee_data['translation_delta'] is not None:
            embedding[0] = min(np.linalg.norm(ee_data['translation_delta']) / 0.5, 1.0)
        
        if ee_data['has_ee'] and ee_data['rotation_delta'] is not None:
            embedding[1] = min(np.linalg.norm(ee_data['rotation_delta']) / 0.5, 1.0)
        
        embedding[2] = min(motion_data['magnitude'] / 2.0, 1.0)
        embedding[3] = min(motion_data['speed'] / 1.0, 1.0)
        embedding[4] = len(joint_data['active_joints']) / max(len(self.joint_semantics), 1)
        
        return embedding
    
    def _compute_semantic_fingerprint(self, joint_data, ee_data, motion_data, workspace_data, motion_embedding):
        """Compute high-dimensional semantic fingerprint"""
        fingerprint_parts = [motion_embedding, workspace_data['region_embedding'], 
                            motion_data['direction'][:min(3, len(motion_data['direction']))]]
        
        comp_vector = np.zeros(10)
        for i, activation in enumerate(joint_data['velocities'].values()):
            if i < 10:
                comp_vector[i] = activation
        fingerprint_parts.append(comp_vector)
        
        fingerprint = np.concatenate(fingerprint_parts)
        
        if len(fingerprint) < 128:
            fingerprint = np.pad(fingerprint, (0, 128 - len(fingerprint)))
        else:
            fingerprint = fingerprint[:128]
        
        return fingerprint
    
    def _generate_description(self, joint_data, ee_data, motion_data, workspace_data):
        """Generate natural language description"""
        parts = []
        
        active_components = [k for k, v in joint_data['velocities'].items() if abs(v) > 0.01]
        if len(active_components) > 0:
            parts.append(f"Moving {len(active_components)} joints")
        
        if ee_data['has_ee'] and ee_data['translation_delta'] is not None:
            dist = np.linalg.norm(ee_data['translation_delta'])
            if dist > 0.01:
                direction_words = self._direction_to_words(ee_data['translation_delta'])
                parts.append(f"End-effector moving {direction_words} by {dist:.3f}m")
        
        if motion_data['speed'] > 0.1:
            parts.append(f"Speed: {motion_data['speed']:.2f}")
        
        return "; ".join(parts) if parts else "Minimal motion"
    
    def _direction_to_words(self, delta):
        """Convert direction vector to words"""
        dx, dy, dz = delta
        words = []
        if abs(dx) > 0.01:
            words.append("forward" if dx > 0 else "backward")
        if abs(dy) > 0.01:
            words.append("left" if dy > 0 else "right")
        if abs(dz) > 0.01:
            words.append("up" if dz > 0 else "down")
        return " and ".join(words) if words else "in place"
    
    def _compute_confidence(self, joint_data, ee_data):
        """Compute confidence in extraction"""
        confidence = 0.5
        if len(joint_data['active_joints']) > 0:
            confidence += 0.2
        if ee_data['has_ee']:
            confidence += 0.3
        return min(confidence, 1.0)
    
    def __del__(self):
        """Cleanup"""
        try:
            p.disconnect(physicsClientId=self.physics_client)
        except:
            pass

def compute_semantic_similarity(action1: SemanticAction, action2: SemanticAction) -> float:
    """Compute similarity between two semantic actions"""
    fp1 = action1.semantic_fingerprint
    fp2 = action2.semantic_fingerprint
    return np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2) + 1e-8)

def visualize_semantic_action(action: SemanticAction):
    """Print human-readable visualization"""
    print("="*70)
    print("SEMANTIC ACTION")
    print("="*70)
    print(f"\nDescription: {action.description}")
    print(f"Confidence: {action.confidence:.2f}")
    print(f"\nJoint Space:")
    print(f"  Active joints: {len(action.active_joints)}")
    print(f"  Motion magnitude: {action.motion_magnitude:.3f}")
    if action.has_end_effector:
        print(f"\nEnd-Effector:")
        print(f"  Translation: {action.ee_translation_delta}")
        print(f"  Velocity: {action.ee_velocity_linear}")
    print(f"\nWorkspace:")
    print(f"  Center: {action.workspace_center}")
    print(f"  Extent: {action.workspace_extent:.3f}m")
    print(f"\nComponents:")
    for comp, activation in action.component_activations.items():
        if activation > 0.01:
            print(f"  {comp}: {activation:.2f}")
    print(f"\nMotion embedding: {action.motion_type_embedding[:5]}...")
    print("="*70)
