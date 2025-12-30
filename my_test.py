import random
import numpy as np
from worker import Worker
from parameter import *
from model import PolicyNet
import matplotlib
# logging_config.py
from loguru import logger
import torch

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
# 配置日志（只需执行一次）
logger.add("log/app.log", rotation="10 MB")
logger.debug("this is debug beginng")

matplotlib.use('TkAgg')  # 指定 TkAgg 后端

local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
model_file = r"/home/iair/kjt_workspace/cmu_ws/core/src/core_ros_planner/scripts/model/checkpoint14_5w5.pth"
local_network.load_state_dict(torch.load(model_file, map_location="cpu")['policy_model'])
worker = Worker(1, local_network, 1, save_image=True)
worker.run_episode()
# node_inputs = torch.stack(rollouts[0]).to(device)
# node_padding_mask = torch.stack(rollouts[1]).to(device)
# edge_mask = torch.stack(rollouts[2]).to(device)
# current_index = torch.stack(rollouts[3]).to(device)
# current_edge = torch.stack(rollouts[4]).to(device)
# edge_padding_mask = torch.stack(rollouts[5]).to(device)
# action = torch.stack(rollouts[6]).to(device)
# reward = torch.stack(rollouts[7]).to(device)
# done = torch.stack(rollouts[8]).to(device)

job_results = worker.episode_buffer
perf_metrics = worker.perf_metrics
print(job_results[0])