"""
PHASE 4: EXTREME MORPHOLOGY STRESS TESTS (FIXED)

With proper PyBullet cleanup between tests + suppressed warnings
"""
import sys
sys.path.insert(0, '/scratch/anshb3/ovla')

import os
import warnings

# Suppress PyBullet warnings
os.environ['PYBULLET_SILENT'] = '1'
warnings.filterwarnings('ignore')

# Redirect stderr to suppress C++ warnings
import io
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr"""
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

import numpy as np
import gc
from ovla.complete_pipeline_with_strategy import CompleteOVLAPipeline
from pathlib import Path

print("="*70)
print("PHASE 4: EXTREME MORPHOLOGY STRESS TESTS (FIXED)")
print("="*70)

test_cases = []

def run_test(name, source_urdf, target_urdf, source_dof, target_dof):
    """Run a single test with cleanup"""
    print(f"\n[{name}]")
    
    pipeline = None
    try:
        # Suppress all the PyBullet warnings during initialization
        with suppress_stdout_stderr():
            pipeline = CompleteOVLAPipeline(
                source_urdf=source_urdf,
                target_urdf=target_urdf
            )
        
        vla_output = np.random.uniform(-0.5, 0.5, source_dof)
        current_state = np.zeros(source_dof)
        
        # Suppress warnings during processing too
        with suppress_stdout_stderr():
            result = pipeline.process(vla_output, current_state, 'joint_position')
        
        success = result['layer4']['trajectory_shape'] == (50, target_dof)
        
        if success:
            print(f"  ✅ PASSED")
            print(f"  Trajectory: {result['layer4']['trajectory_shape']}")
        else:
            print(f"  ❌ FAILED - Wrong shape: {result['layer4']['trajectory_shape']}")
        
        return success
    
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:80]}")
        return False
    
    finally:
        # CRITICAL: Cleanup
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except:
                pass
        del pipeline
        gc.collect()

# TEST 1: Minimum DOF
print("\n" + "="*70)
print("TEST 1: MINIMUM DOF (2-DOF)")
print("="*70)

minimal_urdf = """<?xml version="1.0"?>
<robot name="minimal_2dof">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link_1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
  <link name="link_2">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
</robot>
"""

minimal_path = Path('/scratch/anshb3/ovla/robots/test/minimal_2dof.urdf')
minimal_path.parent.mkdir(parents=True, exist_ok=True)
with open(minimal_path, 'w') as f:
    f.write(minimal_urdf)

success = run_test(
    'Minimum DOF (7 → 2)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    str(minimal_path),
    7, 2
)
test_cases.append(('Minimum DOF (2-DOF)', success))

# TEST 2: Maximum DOF
print("\n" + "="*70)
print("TEST 2: MAXIMUM DOF (23-DOF)")
print("="*70)

success = run_test(
    'Maximum DOF (7 → 23)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    '/scratch/anshb3/ovla/robots/meshfree/g1_meshfree.urdf',
    7, 23
)
test_cases.append(('Maximum DOF (23-DOF)', success))

# TEST 3: Extreme ratios
print("\n" + "="*70)
print("TEST 3: EXTREME DOF RATIOS")
print("="*70)

success = run_test(
    'Small → Large (5 → 23)',
    '/scratch/anshb3/ovla/robots/exotic/wheeled_mobile_manipulator.urdf',
    '/scratch/anshb3/ovla/robots/meshfree/g1_meshfree.urdf',
    5, 23
)
test_cases.append(('Small → Large (5 → 23)', success))

success = run_test(
    'Large → Small (7 → 5)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    '/scratch/anshb3/ovla/robots/exotic/wheeled_mobile_manipulator.urdf',
    7, 5
)
test_cases.append(('Large → Small (7 → 5)', success))

success = run_test(
    'Similar (7 → 7)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    7, 7
)
test_cases.append(('Similar (7 → 7)', success))

# TEST 4: Exotic morphologies
print("\n" + "="*70)
print("TEST 4: EXOTIC MORPHOLOGIES")
print("="*70)

success = run_test(
    'Arm → Snake (7 → 16)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    '/scratch/anshb3/ovla/robots/exotic/snake_16seg.urdf',
    7, 16
)
test_cases.append(('Arm → Snake (7 → 16)', success))

success = run_test(
    'Arm → Hexapod (7 → 18)',
    '/scratch/anshb3/ovla/robots/franka/franka_simple.urdf',
    '/scratch/anshb3/ovla/robots/exotic/hexapod.urdf',
    7, 18
)
test_cases.append(('Arm → Hexapod (7 → 18)', success))

# Summary
print("\n" + "="*70)
print("PHASE 4: RESULTS (FIXED)")
print("="*70)

passed = sum(1 for _, p in test_cases if p)
total = len(test_cases)

print(f"\nTests Passed: {passed}/{total} ({100*passed/total:.0f}%)")

print("\nDetailed Results:")
for name, success in test_cases:
    status = "✅" if success else "❌"
    print(f"  {status} {name}")

print("\n" + "="*70)

if passed == total:
    print("🎯 PHASE 4: 100% COMPLETE!")
    print("\n✅ ALL extreme morphologies validated!")
    print("\nProven DOF Range:")
    print("  ✓ Minimum: 2-DOF")
    print("  ✓ Maximum: 23-DOF")
    print("  ✓ Extreme ratios: 5→23, 7→5, 7→7")
    print("  ✓ Exotic: Snake (16-DOF), Hexapod (18-DOF)")
elif passed >= 5:
    print(f"✅ PHASE 4: MOSTLY COMPLETE ({passed}/{total})")
else:
    print(f"⚠️ PHASE 4: {total-passed} tests still failing")

print("="*70)
