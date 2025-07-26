from worker import Worker
from parameter import *
from model import PolicyNet
import matplotlib
# logging_config.py
from loguru import logger

# 配置日志（只需执行一次）
logger.add("log/app.log", rotation="10 MB")
logger.debug("this is debug beginng")

# matplotlib.use('TkAgg')  # 指定 TkAgg 后端

local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
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