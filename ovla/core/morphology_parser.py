"""
Topology-Based Parser - FINAL CLEAN VERSION
Pure topology analysis, no keywords (except for left/right hints)
"""
import pybullet as p
from typing import Dict, List, Tuple, Set


class TopologyBasedMorphologyParser:
    """
    Pure topology-based morphology parser
    
    NO keywords for detection - only for left/right naming
    """
    
    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.p_client = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self.p_client)
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.p_client)
        
        self.joints = self._extract_joint_info()
        self.graph = self._build_graph()
        self.structure = self._analyze_topology()
        self.structure['urdf_path'] = urdf_path
    
    def _extract_joint_info(self) -> List[Dict]:
        joints = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.p_client)
            joints.append({
                'index': i,
                'name': info[1].decode('utf-8'),
                'type': info[2],
                'parent': info[16],
            })
        return joints
    
    def _build_graph(self) -> Dict[int, List[int]]:
        graph = {}
        for joint in self.joints:
            parent = joint['parent']
            if parent not in graph:
                graph[parent] = []
            graph[parent].append(joint['index'])
        return graph
    
    def _get_controllable_children(self, link_idx: int) -> List[int]:
        if link_idx not in self.graph:
            return []
        return [j for j in self.graph[link_idx] if self.joints[j]['type'] in [0, 1]]
    
    def _trace_chain_from(self, start_joint: int) -> List[int]:
        chain = [start_joint]
        current = start_joint
        
        while True:
            controllable_children = self._get_controllable_children(current)
            
            if len(controllable_children) == 0:
                break
            elif len(controllable_children) > 1:
                break
            
            next_joint = controllable_children[0]
            chain.append(next_joint)
            current = next_joint
        
        return chain
    
    def _is_connector_joint(self, joint_idx: int) -> bool:
        """
        Check if this is a connector joint (torso/waist), not a limb
        
        Connector = has multiple controllable children (branches to multiple limbs)
        """
        controllable_children = self._get_controllable_children(joint_idx)
        return len(controllable_children) >= 2
    
    def _find_all_limb_starts(self) -> List[int]:
        limb_starts = []
        
        for joint in self.joints:
            if joint['type'] == 4:
                continue
            
            parent = joint['parent']
            
            if parent == -1:
                # Attached to base - but check if it's a connector
                if not self._is_connector_joint(joint['index']):
                    limb_starts.append(joint['index'])
                continue
            
            # Parent is a branch point
            parent_controllable_children = self._get_controllable_children(parent)
            if len(parent_controllable_children) >= 2:
                limb_starts.append(joint['index'])
        
        return limb_starts
    
    def _find_all_chains(self) -> List[List[int]]:
        limb_starts = self._find_all_limb_starts()
        
        chains = []
        processed = set()
        
        for start in limb_starts:
            if start in processed:
                continue
            
            chain = self._trace_chain_from(start)
            chains.append(chain)
            processed.update(chain)
        
        return chains
    
    def _get_chain_dof(self, chain: List[int]) -> int:
        return len([j for j in chain if self.joints[j]['type'] in [0, 1]])
    
    def _find_symmetric_pairs(self, chains: List[List[int]]) -> List[Tuple[int, int]]:
        pairs = []
        used = set()
        
        for i, chain1 in enumerate(chains):
            if i in used:
                continue
            
            dof1 = self._get_chain_dof(chain1)
            parent1 = self.joints[chain1[0]]['parent']
            
            for j, chain2 in enumerate(chains):
                if j <= i or j in used:
                    continue
                
                dof2 = self._get_chain_dof(chain2)
                parent2 = self.joints[chain2[0]]['parent']
                
                if dof1 == dof2 and parent1 == parent2:
                    pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    break
        
        return pairs
    
    def _classify_chain(self, chain_dof: int, num_chains: int, num_pairs: int) -> str:
        # 4 pairs = quadruped or octopus
        if num_pairs >= 4:
            return 'leg'
        
        # 2 pairs with 4 chains = humanoid or quadruped
        if num_chains == 4 and num_pairs == 2:
            # Higher DOF = arms
            return 'arm' if chain_dof >= 5 else 'leg'
        
        # 1 pair = dual-arm
        if num_chains == 2:
            return 'arm'
        
        # Single chain
        if num_chains == 1:
            return 'arm'
        
        # Default
        return 'arm' if chain_dof >= 5 else 'leg'
    
    def _get_side_from_name(self, joint_idx: int) -> str:
        """Only for left/right hints - this is acceptable"""
        name = self.joints[joint_idx]['name'].lower()
        
        # Determine side
        if 'left' in name or '_l_' in name or name.startswith('fl'):
            side = 'left'
        elif 'right' in name or '_r_' in name or name.startswith('fr'):
            side = 'right'
        else:
            return None
        
        # Check for front/rear
        if 'front' in name or name.startswith('f'):
            return f'front_{side}'
        elif 'rear' in name or 'rr' in name or 'rl' in name or name.startswith('r') and not name.startswith('right'):
            return f'rear_{side}'
        
        return side
    
    def _assign_sides(self, chains: List[List[int]], pairs: List[Tuple[int, int]]) -> Dict[int, str]:
        sides = {}
        
        unpaired = set(range(len(chains))) - {i for pair in pairs for i in pair}
        for idx in unpaired:
            sides[idx] = None
        
        for idx1, idx2 in pairs:
            side1 = self._get_side_from_name(chains[idx1][0])
            side2 = self._get_side_from_name(chains[idx2][0])
            
            if side1 and side2:
                sides[idx1] = side1
                sides[idx2] = side2
            elif side1:
                sides[idx1] = side1
                if 'left' in side1:
                    sides[idx2] = side1.replace('left', 'right')
                else:
                    sides[idx2] = side1.replace('right', 'left')
            elif side2:
                sides[idx2] = side2
                if 'left' in side2:
                    sides[idx1] = side2.replace('left', 'right')
                else:
                    sides[idx1] = side2.replace('right', 'left')
            else:
                sides[idx1] = 'left'
                sides[idx2] = 'right'
        
        return sides
    
    def _analyze_topology(self) -> Dict:
        chains = self._find_all_chains()
        
        if len(chains) == 0:
            controllable = [j for j in self.joints if j['type'] in [0, 1]]
            if len(controllable) > 0:
                chains = [[j['index'] for j in controllable]]
        
        pairs = self._find_symmetric_pairs(chains)
        sides = self._assign_sides(chains, pairs)
        
        components = []
        arm_count = 0
        leg_count = 0
        
        for idx, chain in enumerate(chains):
            dof = self._get_chain_dof(chain)
            chain_type = self._classify_chain(dof, len(chains), len(pairs))
            side = sides.get(idx)
            
            if chain_type == 'arm':
                arm_count += 1
                name = f'{side}_arm' if side else 'arm'
            else:
                leg_count += 1
                name = f'{side}_leg' if side else 'leg'
            
            components.append({
                'name': name,
                'type': chain_type,
                'joints': chain,
                'dof': dof
            })
        
        if leg_count == 4:
            robot_type = 'quadruped'
        elif leg_count == 2 and arm_count >= 2:
            robot_type = 'humanoid'
        elif arm_count == 2:
            robot_type = 'dual_arm'
        elif arm_count == 1:
            robot_type = 'single_arm'
        else:
            robot_type = 'unknown'
        
        total_dof = sum(c['dof'] for c in components)
        
        return {
            'type': robot_type,
            'total_dof': total_dof,
            'components': components,
            'analysis': {
                'num_arms': arm_count,
                'num_legs': leg_count,
                'num_chains': len(chains),
                'num_pairs': len(pairs)
            }
        }
    
    def get_structure(self) -> Dict:
        return self.structure
    
    def __repr__(self):
        return (f"TopologyParser(type={self.structure['type']}, "
                f"dof={self.structure['total_dof']}, "
                f"arms={self.structure['analysis']['num_arms']}, "
                f"legs={self.structure['analysis']['num_legs']})")
    
    def __del__(self):
        try:
            p.disconnect(self.p_client)
        except:
            pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/scratch/anshb3/ovla')
    
    print("TOPOLOGY-BASED MORPHOLOGY PARSER - FINAL")
    print("="*70)
    
    robots = [
        ("Franka", "/scratch/anshb3/ovla/robots/franka/franka_simple.urdf"),
        ("G1 Dual-Arm", "/scratch/anshb3/ovla/robots/unitree_ros/robots/g1_description/g1_dual_arm.urdf"),
        ("G1 Humanoid", "/scratch/anshb3/ovla/robots/unitree_ros/robots/g1_description/g1_23dof.urdf"),
        ("H1", "/scratch/anshb3/ovla/robots/unitree_ros/robots/h1_description/urdf/h1.urdf"),
        ("Laikago", "/scratch/anshb3/ovla/robots/unitree_ros/robots/laikago_description/urdf/laikago.urdf"),
        ("Spot", "/scratch/anshb3/ovla/robots/spot_ros/spot_description/urdf/spot.urdf.xacro"),
    ]
    
    for name, urdf in robots:
        print(f"\n{name}:")
        try:
            parser = TopologyBasedMorphologyParser(urdf)
            print(f"  {parser}")
            for c in parser.get_structure()['components']:
                print(f"    {c['name']:20s} {c['type']:8s} {c['dof']} DOF")
        except Exception as e:
            print(f"  Error: {str(e)[:80]}")
