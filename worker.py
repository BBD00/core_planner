import matplotlib
matplotlib.use('Agg')
import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet
from log_config import logger
import numpy as np

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)

        self.robot = Agent(policy_net, self.device, self.save_image)

        if self.global_step < 10000:
            if np.random.random() < 0.7:
                distance = np.random.randint(100, 121)
            else:
                distance = np.random.randint(120, 201)
        else:
            if np.random.random() < 0.3:
                distance = np.random.randint(100, 121)
            else:
                distance = np.random.randint(120, 201)
        self.robot.set_goal_point(self.env.ground_truth_info, self.env.robot_cell, distance)
        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False

        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation = self.robot.get_observation()

        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)

        for i in range(MAX_EPISODE_STEP):
            try:
                next_location, action_index = self.robot.select_next_waypoint(observation)

                node = self.robot.node_manager.id_to_node.get(self.robot.node_manager.robot_id)
                check = np.array(node.get_neighbor_coords()).reshape(-1, 2)

                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, logger.warning(f"{self.global_step} invalid next_location: {next_location}, robot_location: {self.robot.location}, neighbors: {check}, dist:{self.robot.node_manager.get_node_by_id(1).neighbor_coords_dist}, id:{self.robot.node_manager.get_node_by_id(1).neighbor_ids}")

                self.save_observation(observation)
                self.save_action(action_index)

                reward = self.env.step(next_location, self.robot.goal_point, self.robot.updating_map_info)

                self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
                if np.linalg.norm(self.robot.location - self.robot.goal_point) <= END_MIN_DISTANCE:
                    done = True
                    reward += 20
                    logger.info(f"{self.global_step} done_reward:{reward}")
                self.save_reward_done(reward, done)

                observation = self.robot.get_observation()
                self.save_next_observations(observation)

                if self.save_image:
                    self.robot.plot_env()
                    self.env.plot_env(i+1)
                if done:
                    break
            except Exception as e:
                print(f'{self.global_step} failed \n')
                logger.error(f'{self.global_step} failed: {e}\n')
                self.robot.plot_env()
                if not self.env.plot:
                    self.env.frame_files = []
                self.env.plot_env(i+1)
                self._remove_incomplete_transition()
                break

        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, done)
        else:
            print(f'{self.global_step} complete | success:{done}\n')

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

    def _remove_incomplete_transition(self):
        if len(self.episode_buffer[0]) == 0:
            return

        min_len = min(len(buf) for buf in self.episode_buffer)
        for i in range(len(self.episode_buffer)):
            self.episode_buffer[i] = self.episode_buffer[i][:min_len]


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)

    worker = Worker(0, model, 77, save_image=True)
    worker.run_episode()
