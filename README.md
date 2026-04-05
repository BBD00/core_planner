# CORE

English | [中文](readme-zh.md)

This repository contains the training code for CORE Planner (Contextual-memory Oriented Reinforcement-learning).

**Note:** This repository only includes training code. ROS-based evaluation code is available at https://github.com/marmotlab/ARiADNE-ROS-Planner.

### Demo of CORE

<div>
   <img src="gifs/demo01.gif" width="360"/><img src="gifs/demo02.gif" width="360"/>
</div>

### Demo Video

<video src="video/core.mp4" controls width="720">
  Your browser does not support the video tag.
</video>

- [Watch `core.mp4`](video/core.mp4)

## 0. Clone the repository

```bash
git clone https://github.com/marmotlab/ARiADNE.git
cd core_planner
```

## 1. Environment setup

It is recommended to create an isolated Conda environment to avoid dependency conflicts. Python 3.10 is recommended.

```bash
# Create an environment named core with Python 3.10
conda create -n core python=3.10

# Activate environment
conda activate core
```

## 2. Install dependencies

All required Python dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

*Note: If you encounter version issues related to PyTorch or Ray, please install the matching PyTorch build according to your CUDA version from the official PyTorch website.*

## 3. Start training

The training entry script is `driver.py`. Before running, you can adjust hyperparameters in `parameter.py` (e.g., learning rate, GPU settings, number of training steps, etc.).

Run:

```bash
RAY_DEDUP_LOGS=0 python3 driver.py
```

**Note:** The success rate usually starts to improve around step `8000` and reaches its peak near step `10000`.

## 4. Project structure

- **driver.py**: Main training driver, responsible for maintaining and updating the global network.
- **parameter.py**: Training-related hyperparameter configuration.
- **worker.py**: Worker logic for interacting with the environment and collecting experience.
- **model.py**: Attention-based deep neural network model definition.
- **env.py**: Environment definition for autonomous exploration.
- **requirements.txt**: Project dependency list.

## 5. Acknowledgement

This project is based on the ARiADNE codebase. If this work helps your research, please cite the original paper:

```bibtex
@INPROCEEDINGS{cao2023ariadne,
  author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  title={ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration},
  year={2023},
  pages={10219-10225},
  doi={10.1109/ICRA48891.2023.10160565}}
```
