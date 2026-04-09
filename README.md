# O-VLA: Universal Vision-Language-Action Transfer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A middleware system enabling ANY vision-language-action (VLA) model to control ANY robot through semantic primitive learning and zero-shot transfer.**

![O-VLA Architecture](docs/architecture.png)

## 🎯 The Problem

Current VLA models (RT-1, RT-2, OpenVLA, Octo) are trained on specific robots. A VLA trained on a 7-DOF Franka arm cannot control a 23-DOF humanoid or a 16-DOF snake robot without retraining.

## 💡 Our Solution

O-VLA is a **robot-agnostic middleware** that translates VLA outputs to work on completely different robots - even robots the VLA has never seen.

### Key Innovation: Semantic Primitive Learning

Instead of memorizing robot-to-robot mappings, O-VLA learns **universal manipulation primitives**:
- `reach_forward` means different joint configurations for different robots
- But the *semantic meaning* transfers across morphologies

## 🏗️ Architecture

O-VLA consists of 7 layers:
VLA Model (e.g., OpenVLA)
↓ [7-DOF action]
Layer 0: Semantic Extractor (analytical, URDF-agnostic)
↓ [128-dim semantic vector]
Layer 0.5: Strategy Extractor (high-level task understanding)
↓ [64-dim strategy vector]
Layer 1: Universal Semantic Mapper (GNN + Transformer)
↓ [target robot semantics]
Layer 1.5: Strategy Mapper (cross-class correction)
↓ [corrected strategy]
Layer 2: Constraint Extractor (geometric limits)
↓ [robot constraints]
Layer 3: Whole-Body Coordinator + Hierarchical Optimizer
↓ [optimized joint positions]
Layer 4: Trajectory Generator (50Hz smooth output)
↓ [final trajectory]
Target Robot (e.g., G1 Humanoid)
↓ [executes 23-DOF motion]

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from ovla import OVLAPipeline
import numpy as np

# Initialize pipeline: Franka arm → G1 humanoid
pipeline = OVLAPipeline(
    source_urdf='robots/franka/franka.urdf',
    target_urdf='robots/humanoid/g1.urdf'
)

# VLA outputs 7-DOF action for Franka
vla_action = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 1.0])
current_state = np.zeros(7)

# O-VLA translates to 23-DOF action for G1
result = pipeline.process(vla_action, current_state, action_format='joint_position')

# Extract final trajectory
trajectory = result['trajectory']  # Shape: (50, 23) at 50Hz
```

### Supported VLA Models

O-VLA automatically handles different action formats:

```python
# RT-1/RT-2 (tokenized actions [0-255])
rt1_action = np.array([140, 91, 136, 91, 131, 122, 171], dtype=np.uint8)
result = pipeline.process(rt1_action, current_state)  # Auto-detects tokenized format

# OpenVLA (continuous actions)
openvla_action = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 1.0])
result = pipeline.process(openvla_action, current_state)

# Octo (action chunking)
octo_action = np.random.randn(4, 7)  # 4 timesteps
result = pipeline.process(octo_action[0], current_state)  # Takes first timestep
```

## 📊 Validation Results

| Phase | Score | Status |
|-------|-------|--------|
| Cross-Class Strategy | 60% (3/5) | ⚠️ Manipulation focus validated |
| Exotic Primitives | 100% | ✅ Transfer working |
| VLA Diversity | 100% | ✅ 4 VLA models |
| Extreme Morphologies | 100% | ✅ 2-23 DOF range |

### Validated Capabilities

✅ **DOF Range:** 2-23 degrees of freedom  
✅ **VLA Models:** RT-1, RT-2, OpenVLA, Octo  
✅ **Action Formats:** Continuous, tokenized, action chunking  
✅ **Robot Types:** Arms, humanoids, mobile manipulators, snake robots  
✅ **Zero-Shot:** Works on unseen robots without retraining  

## 🎓 Training

### Pre-trained Models

We provide pre-trained models:
- `universal_mapper_240k.pt`: Trained on 240K samples, 60 primitives, 50 robots
- `strategy_mapper_MASSIVE.pt`: Trained on 4.4K strategy examples

### Training Your Own

```python
from ovla.training import train_universal_mapper

# Prepare your training data
train_universal_mapper(
    training_samples_path='data/semantic_samples.pkl',
    output_model_path='models/my_mapper.pt',
    num_epochs=100
)
```

See [TRAINING.md](docs/TRAINING.md) for details.

## 🧪 Examples

### Transfer OpenVLA from Franka to G1 Humanoid

```python
from ovla import OVLAPipeline
from openvla import OpenVLA  # Example VLA model

# Load VLA
vla = OpenVLA.from_pretrained('openvla-7b')

# Initialize O-VLA pipeline
pipeline = OVLAPipeline(
    source_urdf='robots/franka/franka.urdf',
    target_urdf='robots/humanoid/g1.urdf'
)

# Run VLA on observation
vla_action = vla.predict(observation)

# Translate to G1
result = pipeline.process(vla_action, current_robot_state)

# Execute on robot
robot.execute_trajectory(result['trajectory'])
```

### Extreme Morphology Transfer

```python
# Snake robot (16-DOF)
pipeline = OVLAPipeline(
    source_urdf='robots/franka/franka.urdf',  # 7-DOF
    target_urdf='robots/exotic/snake_16seg.urdf'  # 16-DOF
)

result = pipeline.process(vla_action, current_state)
# ✓ Works! Transfers 7-DOF → 16-DOF
```

## 📁 Repository Structure
ovla/
├── core/                       # Core O-VLA components
│   ├── semantic_extractor.py  # Layer 0: URDF-agnostic semantics
│   ├── strategy_extractor.py  # Layer 0.5: Task strategy
│   ├── universal_semantic_mapper.py  # Layer 1: GNN + Transformer
│   ├── strategy_mapper.py     # Layer 1.5: Strategy correction
│   ├── constraint_extractor.py # Layer 2: Robot constraints
│   ├── hierarchical_optimizer.py # Layer 3: Physics-based optimization
│   ├── whole_body_coordinator.py # Layer 3: Multi-component coordination
│   ├── trajectory_generator.py # Layer 4: Smooth trajectory
│   └── pipeline.py            # Complete pipeline
├── models/
│   └── pretrained/            # Pre-trained models
├── examples/
│   ├── robots/                # Example URDF files
│   └── validation/            # Validation scripts
└── utils/                     # Utilities

## 🔬 Technical Details

### Semantic Extractor (Layer 0)

**Analytical (no learning required):**
- Forward kinematics for end-effector motion
- Joint space analysis
- Workspace geometry
- Component activation patterns

Output: 128-dim continuous semantic vector

### Universal Semantic Mapper (Layer 1)

**Architecture:**
- Graph Neural Network (GNN) for topology understanding
- Transformer for semantic mapping
- 1.5M parameters

**Training:**
- 240K samples
- 60 universal primitives
- 50 different robots

### Strategy Mapper (Layer 1.5)

**Purpose:** Cross-class strategy correction

Example: Arm → Humanoid requires adding stability constraints

**Training:**
- 4.4K strategy examples
- 625K parameters

### Hierarchical Optimizer (Layer 3)

**Physics-based optimization:**
- Balance checking
- Collision detection
- Inverse kinematics
- Multi-objective optimization

## 🚧 Current Limitations

- **Physical deployment:** Validated in simulation only
- **Legged robots:** Strategy detection for quadrupeds/hexapods not validated (VLAs focus on manipulation)
- **Real VLA models:** RT-1/RT-2 validated with simulated outputs (format accurate)

## 📝 Citation

If you use O-VLA in your research, please cite:

```bibtex
@misc{bhansali2025ovla,
  title={O-VLA: Universal Vision-Language-Action Transfer through Semantic Primitive Learning},
  author={Bhansali, Ansh},
  year={2025},
  institution={University of Illinois Urbana-Champaign}
}
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- UIUC DAIS Lab
- Campus Cluster Computing Resources
- OpenVLA, RT-1, RT-2, Octo teams for inspiration

## 📧 Contact

Ansh Bhansali - anshb3@illinois.edu  
Portfolio: [anshbhansali.com](https://anshbhansali.com)

---

**Built with ❤️ at UIUC**
