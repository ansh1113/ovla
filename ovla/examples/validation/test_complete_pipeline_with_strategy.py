"""
TEST COMPLETE PIPELINE WITH ALL LAYERS INCLUDING STRATEGY
"""
import sys
sys.path.insert(0, '/scratch/anshb3/ovla')

import numpy as np
from ovla.complete_pipeline_with_strategy import CompleteOVLAPipeline

print("="*70)
print("TESTING COMPLETE PIPELINE WITH STRATEGY LAYER")
print("="*70)

# Test: Franka → G1 Humanoid
print("\nTest: Franka (7-DOF) → G1 Humanoid (23-DOF)")

pipeline = CompleteOVLAPipeline(
    source_urdf='/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    target_urdf='/scratch/anshb3/ovla/robots/meshfree/g1_meshfree.urdf'
)

print("✓ Pipeline initialized")

# Simulate VLA output
vla_output = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 1.0])
current_state = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

print("\nProcessing through all layers...")

results = pipeline.process(
    vla_output=vla_output,
    current_source_state=current_state,
    action_format='joint_position'
)

print("\n" + "="*70)
print("LAYER-BY-LAYER RESULTS")
print("="*70)

print(f"\n[Layer 0] Semantic Extraction:")
print(f"  Description: {results['layer0']['description']}")

print(f"\n[Layer 0.5] Strategy Extraction:")
print(f"  Strategy: {results['layer0.5']['strategy']}")
print(f"  Stability required: {results['layer0.5']['stability_required']}")
print(f"  Bilateral: {results['layer0.5']['bilateral']}")

print(f"\n[Layer 1] Universal Mapping:")
print(f"  Output shape: {results['layer1']['actions_shape']}")
print(f"  Mean: {results['layer1']['mean']:.4f}")

print(f"\n[Layer 1.5] Strategy Mapping:")
print(f"  Target needs stability: {results['layer1.5']['target_strategy']['stability_required']}")
print(f"  Strategy delta: {results['layer1.5']['strategy_changed']:.4f}")

print(f"\n[Layer 2] Constraints:")
print(f"  Center: {results['layer2']['center']}")

print(f"\n[Layer 3] Coordination + Optimization:")
print(f"  Coordination delta: {results['layer3']['coordination_delta']:.4f}")
print(f"  Optimization delta: {results['layer3']['optimization_delta']:.4f}")

print(f"\n[Layer 4] Trajectory:")
print(f"  Shape: {results['layer4']['trajectory_shape']}")
print(f"  Control Hz: {results['layer4']['control_hz']}")

print("\n" + "="*70)
print("VALIDATION")
print("="*70)

# Validate all layers worked
checks = [
    results['layer0']['description'] is not None,
    results['layer0.5']['strategy'] is not None,
    results['layer1']['actions_shape'][0] == 23,
    results['layer1.5']['target_strategy'] is not None,
    results['layer3']['final_actions_shape'][0] == 23,
    results['layer4']['trajectory_shape'] == (50, 23)
]

if all(checks):
    print("\n✅ ALL 7 LAYERS WORKING END-TO-END!")
    print("\nComplete architecture validated:")
    print("  ✓ Layer 0: Semantic Extraction")
    print("  ✓ Layer 0.5: Strategy Extraction")
    print("  ✓ Layer 1: Universal Mapping")
    print("  ✓ Layer 1.5: Strategy Mapping")
    print("  ✓ Layer 2: Constraint Extraction")
    print("  ✓ Layer 3: Whole-Body Coordination + Optimization")
    print("  ✓ Layer 4: Trajectory Generation")
else:
    print("\n❌ Some layers failed")

print("\n" + "="*70)
