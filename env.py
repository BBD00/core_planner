import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import parameter

from sensor import sensor_work
from utils import *
import pickle
import datetime
from log_config import logger


class Env:
    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        self.stagnation_time = 0
        self.map_name = None

        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)
        self.cell_size = CELL_SIZE

        self.robot_location = np.array([0.0, 0.0])

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)

        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE
        self.travel_dist = 0
        self.explored_rate = 0
        self.previous_distance = 0

        self.robot_belief = sensor_work(self.robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        self.old_belief = deepcopy(self.robot_belief)

        self.belief_info = MapInfo(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        if self.plot:
            self.frame_files = []
        self.trajectory_x = [self.robot_location[0]]
        self.trajectory_y = [self.robot_location[1]]

    def import_ground_truth(self, episode_index):
        map_dir = map_path
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)


        self.map_name = map_list[map_index]
        ground_truth = io.imread(map_dir + '/' + map_list[map_index], as_gray=True)

        if ground_truth.dtype in (np.float32, np.float64):
            ground_truth = (ground_truth * 255).astype(int)
        else:
            ground_truth = ground_truth.astype(int)

        target_episode = 25000
        if episode_index < target_episode:

            downsample_prob = 1.0 - (episode_index / target_episode)

            if np.random.random() < downsample_prob:
                ground_truth = block_reduce(ground_truth, 2, np.min)

        robot_cell = np.nonzero(ground_truth == 208)

        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1
        if episode_index > 15000:
            free_points = np.argwhere(ground_truth == 255)
            if len(free_points) > 0:
                selected_index = np.random.choice(len(free_points))
                robot_cell = np.array([free_points[selected_index][1], free_points[selected_index][0]])
        return ground_truth, robot_cell


    def debug_save_info(self, map_info, temp_info, updating_map_info, new_cell):
        error_info = {
                "map_info": map_info,
                "temp_info": temp_info,
                "updating_map_info": updating_map_info,
                "new_cell": new_cell
            }

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assert_error_update_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(error_info, f)

    def update_robot_location(self, robot_location, updating_map_info):
        direction = np.array(robot_location) - self.robot_location
        distance = np.linalg.norm(direction)

        if distance > 0:
            unit_direction = direction / distance
            move_distance = min(MOVE_DISTANCE, distance)

            new_location = self.robot_location + unit_direction * move_distance
            new_location = np.round(new_location, 1)

            new_cell_global = np.array([round((new_location[0] - self.belief_origin_x) / self.cell_size),
                             round((new_location[1] - self.belief_origin_y) / self.cell_size)])
            new_cell = get_cell_position_from_coords(new_location, updating_map_info)
            if updating_map_info.map[new_cell[1], new_cell[0]] == FREE and self.ground_truth[new_cell_global[1], new_cell_global[0]] == FREE:

                self.robot_location = new_location
            else:
                for step in np.arange(int(move_distance), 0, -0.5):
                    test_location = self.robot_location + unit_direction * step.item()
                    test_location = np.round(test_location, 1)
                    test_cell = np.array([round((test_location[0] - self.belief_origin_x) / self.cell_size),
                                        round((test_location[1] - self.belief_origin_y) / self.cell_size)])
                    new_cell = get_cell_position_from_coords(test_location, updating_map_info)
                    if (updating_map_info.map[new_cell[1], new_cell[0]] == FREE and
                        self.ground_truth[test_cell[1], test_cell[0]] == FREE):
                        self.robot_location = test_location
                        break
                    if step == 0:
                        logger.error("error!!!")

        self.robot_cell = np.array([round((self.robot_location[0] - self.belief_origin_x) / self.cell_size),
                                round((self.robot_location[1] - self.belief_origin_y) / self.cell_size)])
        try:
            assert self.ground_truth[self.robot_cell[1],self.robot_cell[0]] == FREE, print(f"Robot in obstacle!{self.robot_cell},{self.robot_location}")
        except AssertionError:
            self.debug_save_info(self.ground_truth_info, self.belief_info, updating_map_info, new_cell)
            logger.error(f"Robot in obstacle!{self.robot_cell},{self.robot_location}")
            raise
        self.trajectory_x.append(self.robot_location[0])
        self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self, dist, goal_point):
        reward = 0
        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)
        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        current_distance = np.linalg.norm(self.robot_location - goal_point)
        reward += max(self.previous_distance - current_distance, 0) * 0.5

        self.previous_distance = current_distance

        if len(self.trajectory_x) >= 2:
            prev_x, prev_y = self.trajectory_x[-2], self.trajectory_y[-2]
            current_x, current_y = self.robot_location[0], self.robot_location[1]
            movement = np.hypot(current_x-prev_x, current_y-prev_y)
            reward -= movement / UPDATING_MAP_SIZE * 5
            if movement < STAGNATION_THRESHOLD:
                self.stagnation_time += 1
            else:
                self.stagnation_time = 0
            if self.stagnation_time > 5:
                reward -= min(0.5 * (self.stagnation_time - 5), 5)
        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, goal_point, updating_map_info):
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        self.update_robot_location(next_waypoint, updating_map_info)
        self.update_robot_belief()
        self.travel_dist += dist
        self.evaluate_exploration_rate()
        reward = self.calculate_reward(dist, goal_point)
        return reward

    def plot_env(self, step):
        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}   Map Name:{}'.format(self.explored_rate, self.travel_dist, self.map_name))
        plt.tight_layout()

        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)
