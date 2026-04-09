# O-VLA Architecture

## System Overview

O-VLA is a 7-layer pipeline that enables universal VLA-to-robot transfer through semantic primitive learning.
```
┌─────────────────────────────────────────────────────────────────┐
│                        Any VLA Model                            │
│              (RT-1, RT-2, OpenVLA, Octo, etc.)                  │
└────────────────────────┬────────────────────────────────────────┘
│ N-DOF Action (any dimension)
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0: Semantic Extractor                                    │
│  Purpose: Extract robot-agnostic semantic representation        │
│  Type: Analytical (no training required)                        │
│  ───────────────────────────────────────────────────────────    │
│  • Forward kinematics analysis                                  │
│  • Joint space characterization                                 │
│  • Workspace geometry computation                               │
│  • Component activation detection                               │
│  ───────────────────────────────────────────────────────────    │
│  Works with: ANY URDF file, ANY action dimension                │
│  Output: 128-dim semantic vector + natural language             │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0.5: Strategy Extractor                                  │
│  Purpose: Extract high-level task execution strategy            │
│  Type: Rule-based semantic analysis                             │
│  ───────────────────────────────────────────────────────────    │
│  • Stability requirements detection                             │
│  • Bilateral coordination analysis                              │
│  • Locomotion involvement detection                             │
│  • Motion characteristics extraction                            │
│  ───────────────────────────────────────────────────────────    │
│  Output: 64-dim strategy vector                                 │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Universal Semantic Mapper                             │
│  Purpose: Map source semantics to target robot                  │
│  Type: Learned (GNN + Transformer)                              │
│  ───────────────────────────────────────────────────────────    │
│  Architecture:                                                  │
│    • Graph Neural Network (topology understanding)              │
│    • Transformer (semantic mapping)                             │
│    • 1.5M parameters                                            │
│  ───────────────────────────────────────────────────────────    │
│  Training:                                                      │
│    • 240,000 samples                                            │
│    • 60 universal primitives                                    │
│    • 50 robot morphologies                                      │
│  ───────────────────────────────────────────────────────────    │
│  Key Innovation: Learns primitives, NOT robot pairs             │
│  Output: Predicted semantics for target robot                   │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1.5: Strategy Mapper                                     │
│  Purpose: Cross-class strategy correction                       │
│  Type: Learned (MLP)                                            │
│  ───────────────────────────────────────────────────────────    │
│  Architecture:                                                  │
│    • Multi-layer perceptron                                     │
│    • 625K parameters                                            │
│  ───────────────────────────────────────────────────────────    │
│  Training:                                                      │
│    • 4,430 strategy examples                                    │
│    • Multiple robot classes                                     │
│  ───────────────────────────────────────────────────────────    │
│  Purpose: Detect when execution requirements change             │
│  Example: Fixed-base arm → Mobile humanoid (add stability)      │
│  Output: Corrected strategy for target robot                    │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Constraint Extractor                                  │
│  Purpose: Extract physical constraints from target URDF         │
│  Type: URDF parser                                              │
│  ───────────────────────────────────────────────────────────    │
│  Extracts:                                                      │
│    • Joint position limits                                      │
│    • Joint velocity limits                                      │
│    • Joint effort limits                                        │
│    • Collision geometry                                         │
│  ───────────────────────────────────────────────────────────    │
│  Output: Complete constraint set for optimizer                  │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Hierarchical Optimizer + Whole-Body Coordinator       │
│  Purpose: Generate physically feasible joint trajectories       │
│  Type: Physics-based optimization                               │
│  ───────────────────────────────────────────────────────────    │
│  Whole-Body Coordinator:                                        │
│    • Component identification (arms, legs, torso, etc.)         │
│    • Multi-component synchronization                            │
│    • Task priority assignment                                   │
│  ───────────────────────────────────────────────────────────    │
│  Hierarchical Optimizer (PyBullet):                             │
│    • Balance checking (will robot fall?)                        │
│    • Collision detection (self-collision avoidance)             │
│    • Inverse kinematics solving                                 │
│    • Multi-objective optimization                               │
│  ───────────────────────────────────────────────────────────    │
│  Optimization Hierarchy:                                        │
│    1. Primary: Achieve manipulation goal                        │
│    2. Secondary: Maintain stability                             │
│    3. Tertiary: Avoid collisions                                │
│  ───────────────────────────────────────────────────────────    │
│  Output: Optimized joint positions for target robot             │
└────────────────────────┬────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Trajectory Generator                                  │
│  Purpose: Generate smooth executable trajectory                 │
│  Type: Interpolation + smoothing                                │
│  ───────────────────────────────────────────────────────────    │
│  Process:                                                       │
│    • Interpolate from current state to goal                     │
│    • Velocity smoothing                                         │
│    • Acceleration limiting                                      │
│    • 50Hz trajectory generation                                 │
│  ───────────────────────────────────────────────────────────    │
│  Output: (50, M) trajectory where M = target robot DOF          │
│          50 timesteps at 50Hz = 1 second of motion              │
└────────────────────────┬────────────────────────────────────────┘
│ M-DOF Trajectory (any dimension)
▼
┌─────────────────────────────────────────────────────────────────┐
│                      Target Robot Execution                     │
│                         (Any Robot)                             │
└─────────────────────────────────────────────────────────────────┘
```
---

## Key Design Principles

### 1. Robot-Agnostic by Design

**Layer 0 is analytical, not learned:**
- Works with any URDF file
- No training data required for new robots
- Universal across all morphologies

**Layer 1 learns primitives, not robots:**
- "reach_forward" is a semantic concept
- Applies to ANY robot configuration
- Generalizes to unseen morphologies

### 2. Handles Any DOF Range

**Architecture supports 2-61 DOF:**
- Validated up to 23-DOF
- No hard-coded dimension constraints
- GNN handles arbitrary topology

### 3. Multi-Objective Optimization

**Hierarchical task priorities:**
1. Achieve the manipulation goal (primary)
2. Maintain robot stability (secondary)
3. Avoid self-collisions (tertiary)

**Physics-based validation:**
- Real PyBullet simulation
- Balance checking for stability
- Collision detection for safety

### 4. Format-Agnostic VLA Support

**Automatic action format detection:**
- Continuous actions (OpenVLA, Octo)
- Tokenized [0-255] (RT-1, RT-2)
- Action chunking (multi-timestep)

**No VLA-specific code required.**

---

## Component Details

### Semantic Extractor (Layer 0)

**Input:** VLA action (N-dim) + Source robot URDF

**Process:**
1. Load robot kinematics from URDF
2. Compute forward kinematics
3. Analyze joint space motion
4. Extract end-effector trajectory
5. Compute workspace characteristics
6. Generate semantic fingerprint

**Output:**
- 128-dimensional continuous semantic vector
- Natural language description
- Motion type classification

**Key Feature:** Completely analytical - works with any robot immediately.

---

### Universal Semantic Mapper (Layer 1)

**Architecture:**Input (128-dim semantics)
↓
Graph Neural Network
• Message passing over robot topology
• Node features: joint types, limits, axis
• Edge features: kinematic connections
↓
Transformer
• Self-attention over semantic features
• Cross-attention with robot structure
↓
Output (128-dim target semantics)

**Training Objective:** Learn universal primitives that transfer across robots

**Key Insight:** 
- NOT learning: "If Franka does X, UR5 does Y"
- LEARNING: "What does 'reach_forward' mean for any robot?"

---

### Hierarchical Optimizer (Layer 3)

**Optimization Problem:**minimize: ||q_target - q_desired||²
subject to:
• q_min ≤ q ≤ q_max           (joint limits)
• ||q̇|| ≤ v_max               (velocity limits)
• CoM within support polygon  (stability)
• No self-collisions          (safety)

**Solving Method:**
- Sequential quadratic programming
- PyBullet physics validation
- Iterative constraint satisfaction

---

## Performance Characteristics

### Latency Breakdown

| Layer | Operation | Time |
|-------|-----------|------|
| 0 | Semantic extraction | 9ms |
| 0.5 | Strategy extraction | <1ms |
| 1 | Semantic mapping | 106ms |
| 1.5 | Strategy mapping | <1ms |
| 2 | Constraint extraction | 1ms |
| 3 | Optimization | 343ms |
| 4 | Trajectory generation | <1ms |
| **Total** | **End-to-end** | **~460ms** |

### Scalability

**DOF Scaling:**
- 2-DOF: ~300ms
- 7-DOF: ~460ms
- 23-DOF: ~600ms

**Bottleneck:** Layer 3 optimization (physics simulation)

---

## Extensibility

### Adding New Layers

The modular architecture allows easy extension:

```pythonclass CustomLayer:
def process(self, input_data):
# Your processing logic
return output_dataInsert into pipeline
pipeline.insert_layer(position=2.5, layer=CustomLayer())

### Custom Primitives

Train on your own manipulation primitives:

```pythonfrom ovla.training import add_primitivesadd_primitives([
"precise_insertion",
"tool_use",
"bimanual_assembly"
])
```
---

## Implementation Notes

### Dependencies

**Core:**
- PyTorch 2.0+ (neural networks)
- PyBullet 3.2+ (physics simulation)
- torch-geometric (GNN operations)

**Robotics:**
- urdfpy (URDF parsing)
- numpy (numerical operations)
- scipy (optimization)

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (speeds up Layer 1)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM

---

## Future Directions

### Architectural Improvements

1. **Real-time Layer 3:**
   - GPU-accelerated physics
   - Parallel optimization
   - Target: <100ms end-to-end

2. **Online Learning:**
   - Adapt to robot-specific dynamics
   - Learn from execution feedback
   - Continuous improvement

3. **Multi-Robot Coordination:**
   - Extend to robot teams
   - Collaborative manipulation
   - Distributed optimization

### Research Extensions

1. **Vision Integration:**
   - Direct image → semantics
   - Skip VLA entirely for some tasks
   - End-to-end perception-action

2. **Task Generalization:**
   - Beyond manipulation
   - Locomotion primitives
   - Mobile manipulation

3. **Sim-to-Real:**
   - Domain adaptation
   - Reality gap mitigation
   - Physical deployment validation
