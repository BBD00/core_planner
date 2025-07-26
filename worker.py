import matplotlib.pyplot as plt
import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from loguru import logger

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        """
        初始化数据集工作器

        Args:
            meta_agent_id: 元智能体ID
            policy_net: 模型网络
            global_step: 全局步骤计数
            device: 运行设备(cpu或cuda)
            save_image: 是否保存图像用于可视化
        """
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device
        # 初始化环境、节点管理器和智能体  默认只有一个agent
        self.env = Env(global_step, plot=self.save_image)
        # 创建智能体
        self.robot = Agent(policy_net, self.device, self.save_image)
        self.robot.set_goal_point(self.env.ground_truth_info, self.env.robot_cell)
        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        """
        运行一个完整的探索episode，并收集训练数据

        包括初始化环境，智能体与环境交互，收集状态-动作数据等
        """
        done = False
        # 更新每个机器人的图形表示和规划状态
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)


        for i in range(MAX_EPISODE_STEP):

            self.save_observation(observation)

            next_location, action_index = self.robot.select_next_waypoint(observation)
            self.save_action(action_index)

            node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
            check = np.array(list(node.data.neighbor_edges_set)).reshape(-1, 2)
            assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(self.global_step,next_location, self.robot.location, check)
            assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

            reward = self.env.step(next_location, self.robot.goal_point)  # 更新env中的机器人位置、进行新的传感器检测

            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            if np.linalg.norm(self.robot.location - self.robot.goal_point) <= END_MIN_DISTANCE: # np.array_equal(self.robot.location, self.robot.goal_point) self.robot.utility.sum() == 0
                done = True
                reward += 20
            self.save_reward_done(reward, done)

            observation = self.robot.get_observation()
            self.save_next_observations(observation)

            if self.save_image:
                # 两个都有用
                self.robot.plot_env()
                self.env.plot_env(i+1)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, action_index):
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)

    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 77, save_image=True)
    worker.run_episode()
