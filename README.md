<br>
<div align="center">
  <h1><strong>CORE Planner: Contextual-memory Oriented Reinforcement-learning for autonomous exploration</strong></h1>
  <p>
    Training code release
  </p>
</div>

<div align="center">

[![python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#-getting-started)
[![arXiv](https://img.shields.io/badge/arXiv-2606.29222-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.29222)
[![video](https://img.shields.io/badge/Video-Bilibili-00A1D6?logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV1Ks7G64ESp/)
[![status](https://img.shields.io/badge/Code-Training_Only-6A5ACD)](#-todo)

</div>

<div align="center">
  <a href="https://www.bilibili.com/video/BV1Ks7G64ESp/">
    <img src="gifs/CORE.jpg" width="760" alt="CORE Planner project video: two robots navigating through a forest"/>
  </a>
  <br>
  <sub><strong>Project Video:</strong> Click the image to watch CORE Planner navigate unknown environments.</sub>
</div>

## 🏠 About
**CORE Planner** is a reinforcement-learning-based exploration framework for target-directed navigation using visibility graphs. The current release includes the training code for the planner, with components such as environment simulation, graph-based policy learning, distributed rollout collection, and visualization utilities.

The planner utilizes a contextual memory mechanism over graph-structured observations and is optimized using a distributed SAC-style training pipeline built on **PyTorch** and **Ray**.

> ROS-based evaluation and deployment code will be released later.

## 🎬 Demo
<div align="center">
  <img src="gifs/demo01.gif" width="360"/>
  <img src="gifs/demo02.gif" width="360"/>
</div>

## 📢 News
- [2026-06-30] Refreshed the README with a conference-style open-source layout and clearer training instructions.
- [2026-06-30] The repository currently releases the training pipeline only; ROS testing code is still under preparation.

## ✨ Highlights
- Visibility Graph Representation: CORE Planner utilizes visibility graphs for efficient environment representation, improving computational efficiency in large-scale environments.
- Contextual Memory: The planner integrates a memory mechanism to mitigate local optima and enhance decision-making across long horizons.
- Zero-shot Sim-to-Real Transfer: The planner achieves real-world navigation without fine-tuning, trained entirely on image-based environments.

## 🛠 Getting Started
We recommend the following environment:

- Python `3.10`
- PyTorch `2.1+`
- CUDA-compatible GPU for training

1. **Clone this repository**
   ```bash
   git clone https://github.com/BBD00/core_planner.git
   cd core_planner
   ```

2. **Create a conda environment**
   ```bash
   conda create -n core python=3.10
   conda activate core
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

If `torch` or `ray` has version conflicts on your machine, please install a matching PyTorch build according to your CUDA version first, then re-run the dependency installation.

## 📁 Project Structure
```bash
core/
├── driver.py          # main training entry
├── worker.py          # rollout worker and episode collection
├── runner.py          # Ray remote runner wrapper
├── env.py             # map-based exploration environment
├── agent.py           # planning agent
├── model.py           # policy / Q-network definitions
├── node_manager.py    # graph node management
├── parameter.py       # hyper-parameters and path configuration
├── requirements.txt   # Python dependencies
├── maps/              # training maps
└── gifs/              # demo and training visualization
```

## 📦 Data Preparation
Training maps are loaded from `./maps` by default. The environment reads grayscale map images and converts them into binary occupancy maps during training.

Please make sure:

- your training maps are placed under `maps/`
- the map images contain valid free / occupied regions
- the start position marker used by the environment is consistent with the preprocessing logic in `env.py`

## 🚀 Training
Before launching training, you will usually want to adjust a few settings:

1. **Edit hyper-parameters in `parameter.py`**
   You can modify learning rate, replay buffer size, episode length, worker number, GPU usage, sensor range, movement distance, and logging paths there.

2. **Check GPU selection in `driver.py`**
   The current script explicitly sets:
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"
   ```
   Please change or remove this line based on your machine.

3. **Launch training**
   ```bash
   RAY_DEDUP_LOGS=0 python3 driver.py
   ```

Training artifacts will be written to:

- `model/<FOLDER_NAME>/` for checkpoints
- `train/<FOLDER_NAME>/` for TensorBoard logs
- `gifs/<FOLDER_NAME>/` for visualization results

Based on the current training setup, the success rate typically starts to improve around `step=8000` and may peak around `step=10000`.

## 📊 Monitoring and Visualization
- TensorBoard logs are saved to `train/<FOLDER_NAME>/`.
- Exploration GIFs can be generated during rollout when image saving is enabled.

To inspect training curves:

```bash
tensorboard --logdir train
```

## 📝 TODO
- [✅] Release the training code
- [✅] Release demo visualizations
- [ ] Release ROS-based testing code

## 🙋 Questions or Issues
If you encounter bugs or reproduction issues, please feel free to [open an issue](https://github.com/BBD00/core_planner/issues). When reporting a problem, including your environment and parameter settings will make it much easier to trace dependency or map-format issues.

## 🔗 Citation
If you find **CORE Planner** useful in your research, please cite our paper:

```bibtex
@misc{kong2026coreplanner,
  title         = {{CORE Planner}: Contextual-Memory-Oriented Reinforcement Learning in Unknown Environments for Robot Navigation},
  author        = {Jintao Kong and Zhihao Zhang and Weihuang Chen and Liming Chen and Zhongyu Guo and Shuaiyu Liu and Hongbin Sun},
  year          = {2026},
  eprint        = {2606.29222},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.29222}
}
```
## 🙏 Acknowledgement
This implementation is built upon ideas and code from **ARiADNE**. If this repository is helpful to your work, please also consider citing the original paper:

```bibtex
@INPROCEEDINGS{cao2023ariadne,
  author={Cao, Yuhong and Hou, Tianxiang and Wang, Yizhuo and Yi, Xian and Sartoretti, Guillaume},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  title={ARiADNE: A Reinforcement learning approach using Attention-based Deep Networks for Exploration},
  year={2023},
  pages={10219-10225},
  doi={10.1109/ICRA48891.2023.10160565}
}
```

