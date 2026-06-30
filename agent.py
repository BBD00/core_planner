import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

from utils import *
from parameter import *
from node_manager import NodeManager
import pickle
import datetime
from log_config import logger


class Agent:
    def __init__(self, policy_net, device='cpu', plot=False):
        self.device = device
        self.policy_net = policy_net
        self.plot = plot

        self.goal_point = None

        self.location = None


        self.cell_size = CELL_SIZE
        self.node_resolution = NODE_RESOLUTION
        self.updating_map_size = UPDATING_MAP_SIZE


        self.map_info = None
        self.updating_map_info = None


        self.frontier = set()


        self.node_manager = NodeManager(plot=self.plot)


        self.node_coords, self.utility, self.guidepost, self.goal_distance, self.stay_count = None, None, None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        self.trajectory_x = []
        self.trajectory_y = []

    def update_map(self, map_info):

        self.map_info = map_info

    def update_updating_map(self, location):
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        self.location = location
        self.trajectory_x.append(location[0])
        self.trajectory_y.append(location[1])

    def update_frontiers(self):
        self.frontier = get_frontier_in_map(self.updating_map_info)

    def get_updating_map(self, location):
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size

        min_x = self.map_info.map_origin_x
        min_y = self.map_info.map_origin_y
        max_x = (self.map_info.map_origin_x + self.cell_size * (self.map_info.map.shape[1] - 1))
        max_y = (self.map_info.map_origin_y + self.cell_size * (self.map_info.map.shape[0] - 1))

        if updating_map_origin_x < min_x:
            updating_map_origin_x = min_x
        if updating_map_origin_y < min_y:
            updating_map_origin_y = min_y
        if updating_map_top_x > max_x:
            updating_map_top_x = max_x
        if updating_map_top_y > max_y:
            updating_map_top_y = max_y

        rel_origin_x = updating_map_origin_x - self.map_info.map_origin_x
        rel_origin_y = updating_map_origin_y - self.map_info.map_origin_y
        rel_top_x = updating_map_top_x - self.map_info.map_origin_x
        rel_top_y = updating_map_top_y - self.map_info.map_origin_y

        grid_origin_x = np.floor(rel_origin_x / self.cell_size) * self.cell_size
        grid_origin_y = np.floor(rel_origin_y / self.cell_size) * self.cell_size

        updating_map_origin_x = self.map_info.map_origin_x + grid_origin_x
        updating_map_origin_y = self.map_info.map_origin_y + grid_origin_y

        grid_top_x = np.ceil(rel_top_x / self.cell_size) * self.cell_size
        grid_top_y = np.ceil(rel_top_y / self.cell_size) * self.cell_size

        updating_map_top_x = self.map_info.map_origin_x + grid_top_x
        updating_map_top_y = self.map_info.map_origin_y + grid_top_y

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)

        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)

        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]

        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        self.update_map(map_info)
        self.update_location(location)
        self.update_updating_map(self.location)
        self.update_frontiers()

        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        self.node_coords, self.utility, self.goal_distance, self.stay_count, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
            self.update_observation()


    def update_observation(self):
        all_node_coords = []
        all_node_ids = []
        robot_node_index = None
        for i, (id, node) in enumerate(self.node_manager.id_to_node.items()):
            all_node_coords.append(node.coords)
            all_node_ids.append(id)
            if id == self.node_manager.robot_id:
                robot_node_index = i
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []

        distance = []
        stay_count = []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)

        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

        for i, coords in enumerate(all_node_coords):
            node = self.node_manager.id_to_node.get(all_node_ids[i])
            utility.append(node.utility)
            distance.append(np.linalg.norm(node.coords - self.goal_point))
            stay_count.append(get_stay_count(node.coords, self.trajectory_x, self.trajectory_y))
            for id, neighbor_coords in node.neighbor_coords_dist.items():

                index = np.argwhere(node_coords_to_check == neighbor_coords[0] + neighbor_coords[1] * 1j)
                if index.size > 0:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0

        utility = np.array(utility)
        distance = np.array(distance)
        stay_count = np.array(stay_count)

        if robot_node_index is not None:
            current_index = robot_node_index
        else:

            current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]


        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, distance, stay_count, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self):
        node_coords = self.node_coords
        node_utility = self.utility.reshape(-1, 1)
        node_goal_distance = self.goal_distance.reshape(-1, 1)
        node_stay_count = self.stay_count.reshape(-1, 1)
        current_index = self.current_index
        edge_mask = self.adjacent_matrix
        current_edge = self.neighbor_indices
        n_node = node_coords.shape[0]

        current_node_coords = node_coords[self.current_index]

        relative_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                            node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1)
        max_abs_coord = np.max(np.abs(relative_coords))
        if max_abs_coord > 1e-6:
            node_coords = relative_coords / max_abs_coord
        else:
            node_coords = relative_coords

        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        node_inputs = np.concatenate((node_coords, node_utility, node_goal_distance, node_stay_count), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)

        assert node_coords.shape[0] < NODE_PADDING_SIZE, logger.error(
            f"Node count {node_coords.shape[0]} exceeds padding size {NODE_PADDING_SIZE}"
        )
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)

        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)

        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def set_goal_point(self, ground_truth_map_info, robot_cell, distance=100):
        ground_truth_map = ground_truth_map_info.map
        free_points_all = np.argwhere(ground_truth_map == 255)
        free_points = free_points_all

        if distance is not None:
            robot_yx = np.array([robot_cell[1], robot_cell[0]])
            distances = np.sqrt(np.sum((free_points - robot_yx) ** 2, axis=1))
            free_points = free_points[distances >= distance]

        if len(free_points) == 0:
            print("Warning: no feasible point satisfies the constraints; sampling a random goal.")
            selected_indices = np.random.choice(len(free_points_all), 1, replace=False)
            selected_points = free_points_all[selected_indices]
        else:
            selected_indices = np.random.choice(len(free_points), 1, replace=False)
            selected_points = free_points[selected_indices]


        self.goal_point = get_coords_from_cell_position(selected_points[:, [1, 0]], ground_truth_map_info)
        self.node_manager.goal_point = self.goal_point


    def select_next_waypoint(self, observation):
        _, _, _, _, current_edge, edge_padding_mask = observation

        valid_edge_count = torch.sum(edge_padding_mask == 0).item()

        if valid_edge_count == 0:
            if IS_TRAIN:
                assert False, logger.error(f"Robot Node {self.location} has no valid edges, moving directly to goal {self.goal_point}")
            print(f"Error: Robot Node {self.location} has no valid edges, moving directly to goal {self.goal_point}")
            logger.warning(f"Robot Node {self.location} has no valid edges, moving directly to goal {self.goal_point}")

            direction = self.goal_point - self.location
            dist = np.linalg.norm(direction)
            if dist > 0:
                step_size = min(MOVE_DISTANCE, dist)
                next_position = self.location + (direction / dist) * step_size
                next_position = np.round(next_position, 1)
            else:
                next_position = self.goal_point

            return next_position, torch.tensor([self.node_manager.goal_id])

        with torch.no_grad():
            logp = self.policy_net(*observation)
        valid_logp_range = logp[0, :valid_edge_count]
        valid_mask = valid_logp_range > -1e7
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            print(f"Error: All action probabilities are invalid {logp}, trying to use valid edges")
            logger.warning(f"All action probabilities are invalid, trying to use valid edges")
            valid_indices = torch.arange(valid_edge_count, device=logp.device)
            valid_probs = torch.ones(valid_edge_count, device=logp.device) / valid_edge_count
        else:
            valid_logp = logp[:, valid_indices]
            valid_probs = valid_logp.softmax(dim=-1)

        valid_node_indices = current_edge[0, valid_indices, 0].cpu().numpy().flatten()
        self.action_probs = valid_probs.cpu().numpy().flatten()
        self.action_node_coords = self.node_coords[valid_node_indices].copy()
        sampled_idx = torch.multinomial(valid_probs, 1).long().squeeze(1)
        action_index = valid_indices[sampled_idx]
        next_node_index = current_edge[0, action_index.item(), 0].item()
        next_position = self.node_coords[next_node_index]
        return next_position, action_index

    def plot_env(self):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 2)
        nodes = get_cell_position_from_coords(self.node_coords, self.map_info)
        if len(self.frontier) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.frontier)), self.map_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)
        robot = get_cell_position_from_coords(self.location, self.map_info)
        goal_cell = get_cell_position_from_coords(self.goal_point, self.map_info)
        plt.imshow(self.map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        for i, (node, utility, stay_count) in enumerate(zip(nodes, self.utility, self.stay_count)):
            plt.text(node[0], node[1], f"({self.node_coords[i][0]},{self.node_coords[i][1]}){stay_count}", fontsize=5, zorder=3)
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        plt.plot(goal_cell[0], goal_cell[1], 'ro', markersize=10, zorder=5)
        for coords in self.node_coords:
            node = self.node_manager.get_node(coords.tolist())
            for neighbor_coords in node.get_neighbor_coords():
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                               (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.subplot(1, 3, 3)
        plt.imshow(self.map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=4, zorder=5)
        plt.plot(goal_cell[0], goal_cell[1], 'ro', markersize=4, zorder=5)
        node = self.node_manager.get_node(self.location.tolist())
        for neighbor_coords in node.get_neighbor_coords():
            end = (np.array(neighbor_coords) - self.location) / 2 + self.location
            plt.plot((np.array([self.location[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                     (np.array([self.location[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        if hasattr(self, 'action_probs') and self.action_probs is not None and hasattr(self, 'action_node_coords') and self.action_node_coords is not None:
            cmap = plt.cm.Reds
            coord_to_prob = {}
            for i, coord in enumerate(self.action_node_coords):
                coord_key = (round(coord[0], 1), round(coord[1], 1))
                coord_to_prob[coord_key] = self.action_probs[i]
            for neighbor_coords in node.get_neighbor_coords():
                end = np.array(neighbor_coords)
                start = self.location
                neighbor_key = (round(neighbor_coords[0], 1), round(neighbor_coords[1], 1))
                if neighbor_key in coord_to_prob:
                    prob = coord_to_prob[neighbor_key]
                    color = cmap(0.3 + 0.7 * prob)
                    linewidth = 1 + 3 * prob
                else:
                    color = 'lightgray'
                    linewidth = 1
                    prob = None
                plt.plot(
                    (np.array([start[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                    (np.array([start[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size,
                    color=color, linewidth=linewidth, zorder=4
                )
            for i, coord in enumerate(self.action_node_coords):
                neighbor_cell = get_cell_position_from_coords(coord, self.map_info)
                prob = self.action_probs[i]
                color = cmap(0.3 + 0.7 * prob)
                plt.scatter(neighbor_cell[0], neighbor_cell[1], c=[color], s=50, edgecolors='black', zorder=6)
                plt.text(neighbor_cell[0] + 2, neighbor_cell[1] + 2, f'{prob:.2f}', fontsize=8, color='blue', zorder=7)
        else:
            if node is not None:
                for neighbor_coords in node.get_neighbor_coords():
                    end = np.array(neighbor_coords)
                    plt.plot((np.array([self.location[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                             (np.array([self.location[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)
        current_node_x = (self.location[0] - self.map_info.map_origin_x) / self.cell_size
        current_node_y = (self.location[1] - self.map_info.map_origin_y) / self.cell_size

        rect_size = UPDATING_MAP_SIZE / CELL_SIZE
        rect = plt.Rectangle((current_node_x - rect_size / 2, current_node_y - rect_size / 2),
                             rect_size, rect_size,
                             linewidth=2, edgecolor='cyan', facecolor='none', zorder=6)
        plt.gca().add_patch(rect)
