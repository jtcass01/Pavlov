# Pavlov

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0.2-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)


## Pavlov Environments

Pavlov is a collection of environments for reinforcement learning research and benchmarking. It is built on top of the IsaacSim physics engine and provides a set of environments for training agents in a variety of tasks. The environments are designed to be easy to use and modify, and are built with the goal of enabling research in reinforcement learning.

list of environments:
- [x] `PitchBot`: A baseball pitching environment where the agent learns to throw a ball to a target. [IN PROGRESS]

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
