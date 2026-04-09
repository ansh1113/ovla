"""
Basic O-VLA Transfer Example

Demonstrates transferring a VLA action from Franka arm to G1 humanoid
"""

import numpy as np
from ovla import OVLAPipeline

def main():
    print("="*70)
    print("O-VLA: Basic Transfer Example")
    print("="*70)
    
    # Initialize pipeline: Franka (7-DOF) → G1 Humanoid (23-DOF)
    print("\nInitializing pipeline...")
    pipeline = OVLAPipeline(
        source_urdf='ovla/examples/robots/franka/franka_simple.urdf',
        target_urdf='ovla/examples/robots/humanoid/g1.urdf'
    )
    
    # Simulate VLA output for Franka
    print("\nVLA outputs 7-DOF action for Franka arm...")
    vla_action = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 1.0])
    current_state = np.zeros(7)
    
    print(f"  VLA action: {vla_action}")
    
    # O-VLA translates to G1
    print("\nO-VLA translating to G1 humanoid (23-DOF)...")
    result = pipeline.process(
        vla_output=vla_action,
        current_source_state=current_state,
        action_format='joint_position'
    )
    
    # Extract results
    trajectory = result['layer4']['trajectory']
    
    print(f"\n✅ Success!")
    print(f"  Input: {vla_action.shape} (Franka)")
    print(f"  Output: {trajectory.shape} (G1 Humanoid)")
    print(f"  Trajectory: 50 timesteps at 50Hz = 1 second of motion")
    
    print("\n" + "="*70)
    print("Transfer complete!")
    print("="*70)

if __name__ == "__main__":
    main()
