import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy

from sensor import sensor_work
from utils import *


class Env:
    def __init__(self, episode_index, plot=False):
        self.episode_index = episode_index
        self.plot = plot
        # 导入地面真值地图(1 or 255)和初始机器人单元格位置 255白色为可通行 initial_cell为大于0数
        self.ground_truth, self.robot_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)  # cell
        self.cell_size = CELL_SIZE  # meter

        self.robot_location = np.array([0.0, 0.0])  # meter

        self.robot_belief = np.ones(self.ground_truth_size) * 127
        self.belief_origin_x = -np.round(self.robot_cell[0] * self.cell_size, 1)   # meter
        self.belief_origin_y = -np.round(self.robot_cell[1] * self.cell_size, 1)  # meter

        self.global_frontiers = set()

        self.sensor_range = SENSOR_RANGE  # meter
        self.travel_dist = 0  # meter
        self.explored_rate = 0
        # 栅格数据
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
        """
        导入地面真值地图

        Args:
            episode_index: 场景索引，用于选择不同的地图

        Returns:
            ground_truth: 地面真值地图
            robot_cell: 机器人初始单元格位置
        """
        map_dir = f'maps'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        print(map_index)
        # 读取地图文件、灰度图
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1) * 255).astype(int)
        # 使用block_reduce进行2x2的降采样(取最小值) 最小池化
        ground_truth = block_reduce(ground_truth, 2, np.min)
        # 找到标记为208的像素点作为机器人初始位置
        robot_cell = np.nonzero(ground_truth == 208)
        # 这里交换是因为np.nonzero返回的是（行、列）索引，而x是列索引，y是行索引
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])
        # 将地图转换为二值地图: 值>150或者50<=值<=80的像素设为可通行(254+1)
        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        # 计算从当前位置到目标位置的方向向量
        direction = np.array(robot_location) - self.robot_location
        distance = np.linalg.norm(direction)

        # 如果距离大于0，则移动固定步长
        if distance > 0:
            # 计算单位方向向量
            unit_direction = direction / distance

            # 计算实际移动距离（不超过到目标的距离）
            move_distance = min(MOVE_DISTANCE, distance)

            # 更新机器人位置
            self.robot_location += unit_direction * move_distance
            self.robot_location = np.round(self.robot_location)


        # self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])
        if self.plot:
            self.trajectory_x.append(self.robot_location[0])
            self.trajectory_y.append(self.robot_location[1])

    def update_robot_belief(self):
        self.robot_belief = sensor_work(self.robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self, dist):
        reward = 0
        reward -= dist / UPDATING_MAP_SIZE * 5

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint):
        """
        执行环境的一步更新

        Args:
            next_waypoint: 下一个路径点坐标(米)
            agent_id: 执行动作的智能体ID
        """
        dist = np.linalg.norm(self.robot_location - next_waypoint)
        # 更新指定智能体的位置
        self.update_robot_location(next_waypoint)
        # 更新新的传感器
        self.update_robot_belief()

        self.travel_dist += dist
        # 计算当前探索率
        self.evaluate_exploration_rate()

        reward = self.calculate_reward(dist)

        return reward

    def plot_env(self, step):

        plt.subplot(1, 3, 1)
        plt.imshow(self.robot_belief, cmap='gray')
        plt.axis('off')
        plt.plot((self.robot_location[0] - self.belief_origin_x) / self.cell_size,
                 (self.robot_location[1] - self.belief_origin_y) / self.cell_size, 'mo', markersize=4, zorder=5)
        plt.plot((np.array(self.trajectory_x) - self.belief_origin_x) / self.cell_size,
                 (np.array(self.trajectory_y) - self.belief_origin_y) / self.cell_size, 'b', linewidth=2, zorder=1)
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.explored_rate, self.travel_dist))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.episode_index, step)
        plt.close()
        self.frame_files.append(frame)

