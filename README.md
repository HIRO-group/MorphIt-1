# MorphIt: Flexible Spherical Approximation of Robot Morphology

<img src="imgs/overview.png" alt="MorphIt Overview" width="500">

MorphIt is a novel algorithm for approximating robot morphology using spherical primitives that balances geometric accuracy with computational efficiency. Unlike existing approaches that rely on either labor-intensive manual specification or inflexible computational methods, MorphIt implements an automatic gradient-based optimization framework with tunable parameters that provides explicit control over the physical fidelity versus computational cost tradeoff.

## Key Features

- **Automatic spherical approximation** of any mesh with gradient-based optimization
- **Flexible configuration** for different task requirements
- **Fast computation** - up to 2 orders of magnitude faster than baseline methods
- **Physical fidelity** - better mesh approximation with fewer spheres
- **Easy integration** - generates URDF files compatible with popular robotics simulators

## Quick Start

### Setup Environment

```bash
uv venv venv-morphit --python 3.10
source venv-morphit/bin/activate
uv pip install -r requirements.txt
```

### Compile Helper Module

Compile an efficient helper module used to check whether a point lies inside a mesh:

```bash
sudo apt-get install python3.10-dev
cd src
python setup.py build_ext --inplace
```

This should generate a file such as `triangle_hash.cpython-310-x86_64-linux-gnu.so`

### Run Tests

Test the system to ensure everything is working:

```bash
cd src
python test_morphit.py
```

This runs some tests including imports, configuration, model creation, loss computation, and minimal training to verify the installation.

### Run Example

```bash
cd src
python main.py
```

## Usage

### Basic Usage

```python
from morphit import MorphIt
from config import get_config

# Create configuration
config = get_config("MorphIt-B")  # Balanced configuration
config.model.mesh_path = "path/to/your/mesh.obj"
config.model.num_spheres = 15

# Create and train model
model = MorphIt(config)
tracker = model.train()

# Save results
model.save_results("output.json")
```

<img src="imgs/sphere-num-compare.png" alt="Number of spheres comparison" width="500">

### Batch Processing

Process multiple mesh files:

```bash
cd src
python morphit_panda.py
```

### Visualization

Visualize sphere packing results:

```bash
cd src
python visualize_packing.py results/output/morphit_results.json
```

### Convert to URDF and Visualize the Panda Robot in Drake

Create the sheres, populate the URDF file, then use a simulator like Drake to use your new robot mophology.

```bash
cd src
python morphit_panda.py
cd scripts
python create_panda_urdf.py
python run_spherical_panda_drake_example.py
```

## File Overview

### Core System

- `morphit.py` - Main MorphIt class implementing the spherical approximation algorithm
- `config.py` - Configuration management with predefined variants (V, S, B)
- `losses.py` - Loss functions for optimization (coverage, overlap, boundary, surface, containment, SQEM)
- `training.py` - Training loop and optimization logic
- `density_control.py` - Adaptive sphere pruning and addition mechanisms

### Utilities

- `inside_mesh.py` - Efficient mesh containment checking using Cython
- `visualization.py` - PyVista-based visualization functions
- `convergence_tracker.py` - Training metrics tracking and analysis
- `logger.py` - Evolution logging and rendering thread management

### Scripts

- `main.py` - Main example script
- `test_morphit.py` - Comprehensive system tests
- `morphit_panda.py` - Batch processing for Franka Panda robot
- `visualize_packing.py` - Standalone visualization tool
- `scripts/` - URDF generation and integration scripts for Drake, CuRobo

### Build Files

- `setup.py` - Cython compilation setup
- `triangle_hash.pyx` - Cython implementation for fast mesh queries
- `triangle_hash.cpp` - Generated C++ code

## Configuration Variants

- **MorphIt-V**: Volume-focused, conservative approximations for collision avoidance
- **MorphIt-S**: Surface-focused, precise approximations for contact-rich manipulation
- **MorphIt-B**: Balanced approximation for general-purpose applications

<img src="imgs/bvs-compare.png" alt="V-S-B visual comparson" width="500">


## Applications

- Collision detection and avoidance
- Contact-rich manipulation planning
- Navigation through cluttered spaces
- Whole-body contact planning
- Real-time motion generation

## Tested on a system with the following requirements

- Ubuntu 22.04.5 LTS (recommended)
- Python 3.10
- CUDA-capable GPU (recommended, for acceleration)

## Baselines
Special thanks to

## Citation

If you use MorphIt in your research, please cite our paper:

```
@misc{nechyporenko2025morphit,
      title={MorphIt: Flexible Spherical Approximation of Robot Morphology for Representation-driven Adaptation},
      author={Nataliya Nechyporenko and Yutong Zhang and Sean Campbell and Alessandro Roncone},
      year={2025},
      eprint={2507.14061},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.14061},
}
```

## Acknowledgments

To test our work, we have used several outstanding open-source robotics libraries and tools that have significantly accelerated research and development in robot motion planning, dynamics, and geometry approximation. We gratefully acknowledge the contributions of the following projects:

### Motion Planning and Control

- **[cuRobo (NVlabs)](https://github.com/NVlabs/curobo)** – A CUDA-accelerated robotics library providing high-performance motion generation, collision-free trajectory optimization, and inverse kinematics that operates within milliseconds on NVIDIA GPUs.

- **[Drake (Robot Locomotion Group, MIT/Toyota Research Institute)](https://github.com/RobotLocomotion/drake)** – A comprehensive C++/Python toolbox for model-based design, multibody dynamics simulation, and optimization-based control systems analysis.

- **[IDTO (Toyota Research Institute)](https://github.com/ToyotaResearchInstitute/idto)** – Inverse Dynamics Trajectory Optimization implementation for contact-implicit model predictive control, enabling real-time robotic manipulation and locomotion planning.

### Geometric Approximation and Collision Detection

- **[sphere-set-approximation](https://github.com/CK1201/sphere-set-approximation)** and its **[parent repository](https://github.com/111116/sphere-set-approximation)** – Algorithms for approximating triangular meshes with minimal-volume sphere sets, useful for efficient collision detection and shape representation.

- **[spheretree (mlund)](https://github.com/mlund/spheretree)** – A sphere-tree construction toolkit for generating hierarchical sphere representations of polygonal models to accelerate interruptible collision detection.

- **[foam (CoMMA Lab, Purdue University)](https://github.com/CoMMALab/foam)** – A specialized tool for spherical approximation of robot geometry, enabling fast proximity queries and motion planning for complex articulated systems.



## License

This project is licensed under the MIT License.
