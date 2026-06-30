import time
import heapq

import matplotlib.pyplot as plt
import numpy as np
from utils import *
from parameter import *
from loguru import logger
from copy import deepcopy
from rtree import index
from scipy.spatial import cKDTree

class NodeManager:
    def __init__(self, plot=False):
        self.nodes_tree = index.Index()
        self.id_to_node = {}
        self.plot = plot
        self.goal_point = None
        self.goal_id = None
        self.robot_id = 1
        self.id_tracker = 1
        self.id_frontiers_tracker = -1
        self.previous_updating_map = None
        self.previous_map_origin = None
        self._changed_coords_tree = None
        self._update_counter = 0

    def _make_bounds(self, coords):
        return (float(coords[0]), float(coords[1]), float(coords[0]), float(coords[1]))


    def check_node_exist_in_dict(self, coords):
        bounds = self._make_bounds(coords)
        exist = list(self.nodes_tree.intersection(bounds))
        if not exist:
            return None

        for node_id in exist:
            node = self.id_to_node.get(node_id)
            if node and np.allclose(node.coords, coords, atol=1e-6):
                return node
        return None

    def get_node_id_by_coords(self, coords):
        bounds = self._make_bounds(coords)
        result = list(self.nodes_tree.intersection(bounds))
        for node_id in result:
            node = self.id_to_node.get(node_id)
            if node and np.allclose(node.coords, coords, atol=1e-6):
                return node_id
        return None

    def get_node_by_id(self, node_id):
        return self.id_to_node.get(node_id)

    def get_node(self, coords):
        if isinstance(coords, int):
            return self.get_node_by_id(coords)
        else:
            return self.check_node_exist_in_dict(coords)

    def add_node_to_dict(self, coords, frontiers, edges_coords, updating_map_info, is_robot=False, is_goal=False, is_frontier=False):
        bounds = self._make_bounds(coords)
        if is_robot:
            node = LocalNode(self.robot_id, coords, frontiers, edges_coords, updating_map_info)
            if self.robot_id in self.id_to_node:
                old_node = self.id_to_node[self.robot_id]
                for neighbor_id in list(old_node.neighbor_ids):
                    neighbor_node = self.id_to_node.get(neighbor_id)
                    if neighbor_node:
                        neighbor_node.remove_neighbor_by_id(self.robot_id)
                old_node.neighbor_ids.clear()
                old_node.neighbor_coords_dist.clear()

                old_bounds = self._make_bounds(old_node.coords)
                self.nodes_tree.delete(self.robot_id, old_bounds)
                self.id_to_node[self.robot_id] = node
                self.nodes_tree.insert(self.robot_id, bounds)
            else:
                self.id_to_node[self.robot_id] = node
                self.nodes_tree.insert(self.robot_id, bounds)
            return node

        existing = self.check_node_exist_in_dict(coords)
        if existing is not None:
            logger.info("Node already exists, skip adding.")
            return existing
        if is_goal:
            if self.goal_id is None:
                self.id_tracker += 1
                self.goal_id = self.id_tracker
            node = LocalNode(self.goal_id, coords, frontiers, edges_coords, updating_map_info)

            self.id_to_node[self.goal_id] = node
            self.nodes_tree.insert(self.goal_id, bounds)
        elif is_frontier:
            self.id_frontiers_tracker -= 1
            node = LocalNode(self.id_frontiers_tracker, coords, frontiers, edges_coords, updating_map_info)
            self.id_to_node[self.id_frontiers_tracker] = node
            self.nodes_tree.insert(self.id_frontiers_tracker, bounds)
        else:
            self.id_tracker += 1
            node = LocalNode(self.id_tracker, coords, frontiers, edges_coords, updating_map_info)
            self.id_to_node[self.id_tracker] = node
            self.nodes_tree.insert(self.id_tracker, bounds)

        return node

    def _process_pending_edges(self, node):
        edge_coords = node.pending_edge_coords
        if edge_coords is None or len(edge_coords) == 0:
            return
        edge_coords = np.array(edge_coords)
        if edge_coords.ndim == 1:
            edge_coords = edge_coords.reshape(-1, 2)
        for coord in edge_coords:
            neighbor_id = self.get_node_id_by_coords(coord)
            if neighbor_id is not None and neighbor_id != node.id:

                node.add_neighbor(neighbor_id, coord)
                self.id_to_node[neighbor_id].add_neighbor(node.id, node.coords)
        node.pending_edge_coords = None

    def remove_node_from_dict(self, node):
        if node is None:
            return
        node_id = node.id
        neighbor_ids_copy = list(node.neighbor_ids)


        for neighbor_id in neighbor_ids_copy:
            neighbor_node = self.id_to_node.get(neighbor_id)
            if neighbor_node:
                neighbor_node.remove_neighbor_by_id(node_id)

        node.neighbor_ids.clear()
        node.neighbor_coords_dist.clear()

        bounds = self._make_bounds(node.coords)
        self.nodes_tree.delete(node_id, bounds)
        self.id_to_node.pop(node_id, None)

    def within_bb(self, min_x, min_y, max_x, max_y):
        bounds = (min_x, min_y, max_x, max_y)
        result_ids = list(self.nodes_tree.intersection(bounds))
        return [self.id_to_node[nid] for nid in result_ids if nid in self.id_to_node]

    def get_neighbor_nodes(self, node):
        return [self.id_to_node[nid] for nid in node.neighbor_ids if nid in self.id_to_node]

    def remove_history_node(self, center_point, new_points, updating_map_info):
        buffer_size = 3.0
        map_min_x = updating_map_info.map_origin_x + buffer_size
        map_min_y = updating_map_info.map_origin_y + buffer_size
        map_max_x = updating_map_info.map_origin_x + updating_map_info.cell_size * (updating_map_info.map.shape[1] - 1) - buffer_size
        map_max_y = updating_map_info.map_origin_y + updating_map_info.cell_size * (updating_map_info.map.shape[0] - 1) - buffer_size

        nodes_in_range = self.within_bb(
            min_x=map_min_x,
            min_y=map_min_y,
            max_x=map_max_x,
            max_y=map_max_y
        )

        new_points_set = set()
        for point in new_points:
            new_points_set.add((round(point[0], 1), round(point[1], 1)))

        nodes_to_remove = []
        current_step = self._update_counter

        for node in nodes_in_range:
            if node.id == self.robot_id or node.id == self.goal_id:
                continue
            if self.goal_point is not None and np.allclose(node.coords, self.goal_point, atol=0.2):
                continue
            px, py = round(node.coords[0], 1), round(node.coords[1], 1)
            cell = get_cell_position_from_coords(node.coords, updating_map_info, check_negative=False)
            cell_x, cell_y = int(cell[0]), int(cell[1])
            cell_value = updating_map_info.map[cell_y, cell_x]
            if cell_value == OCCUPIED:
                nodes_to_remove.append(node)
                continue
            if (px, py) not in new_points_set:
                steps_since_seen = current_step - node.last_seen
                has_valid_neighbors = len(node.neighbor_ids) > 0
                if not has_valid_neighbors:
                    node.noise_flag += 1
                    if node.noise_flag >= NOISE_THRESHOLD + 1:
                        nodes_to_remove.append(node)
                else:
                    node.noise_flag += 1
                    if node.noise_flag >= NOISE_THRESHOLD + 1:
                        nodes_to_remove.append(node)
            else:
                node.noise_flag = max(0, node.noise_flag - 2)

        for node_to_remove in nodes_to_remove:
            existing_node = self.check_node_exist_in_dict(node_to_remove.coords)
            if existing_node:
                self.remove_node_from_dict(existing_node)

    def update_node_neighbors(self, node):
        if node is None:
            return
        node_id_to_remove = []

        for neighbor_id in list(node.neighbor_ids):
            neighbor_node = self.id_to_node.get(neighbor_id)
            if neighbor_node is None:
                logger.debug(f"NM: neighbor_id {neighbor_id} not in dict, removing from node {node.id}")
                node_id_to_remove.append(neighbor_id)
            else:
                if node.id not in neighbor_node.neighbor_ids:
                    neighbor_node.add_neighbor(node.id, node.coords)
        for neighbor_id in node_id_to_remove:
            node.neighbor_ids.discard(neighbor_id)
            node.neighbor_coords_dist.pop(neighbor_id, None)
        self._process_pending_edges(node)

    def init_node_neighbors(self, node, updating_map_info):
        node_id = node.id
        if node.coords[0] < updating_map_info.map_origin_x or node.coords[1] < updating_map_info.map_origin_y or \
            node.coords[0] > updating_map_info.map_origin_x + updating_map_info.cell_size * updating_map_info.map.shape[1] or \
            node.coords[1] > updating_map_info.map_origin_y + updating_map_info.cell_size * updating_map_info.map.shape[0]:
            return
        nodes_in_range = self.within_bb( min_x=node.coords[0] - UPDATING_MAP_SIZE / 2,
                                            min_y=node.coords[1] - UPDATING_MAP_SIZE / 2,
                                            max_x=node.coords[0] + UPDATING_MAP_SIZE / 2,
                                            max_y=node.coords[1] + UPDATING_MAP_SIZE / 2)

        for neighbor in nodes_in_range:
            if not np.array_equal(neighbor.id, node_id):
                if not check_collision(node.coords, neighbor.coords, updating_map_info):
                    node.add_neighbor(neighbor.id, neighbor.coords)
                    neighbor.add_neighbor(node.id, node.coords)

    def get_changed_region_mask(self, updating_map_info):
        current_map = updating_map_info.map
        current_origin = (updating_map_info.map_origin_x, updating_map_info.map_origin_y)
        cell_size = updating_map_info.cell_size

        if self.previous_updating_map is None:
            known_mask = current_map != UNKNOWN
            y_indices, x_indices = np.where(known_mask)
            world_x = updating_map_info.map_origin_x + x_indices * cell_size
            world_y = updating_map_info.map_origin_y + y_indices * cell_size
            coords = np.round(np.column_stack((world_x, world_y)), 1)
            changed_coords = set(map(tuple, coords))
        else:
            prev_origin_x, prev_origin_y = self.previous_map_origin
            curr_origin_x, curr_origin_y = current_origin
            prev_h, prev_w = self.previous_updating_map.shape
            curr_h, curr_w = current_map.shape
            y_grid, x_grid = np.mgrid[0:curr_h, 0:curr_w]
            world_x = curr_origin_x + x_grid * cell_size
            world_y = curr_origin_y + y_grid * cell_size
            prev_x = np.round((world_x - prev_origin_x) / cell_size).astype(int)
            prev_y = np.round((world_y - prev_origin_y) / cell_size).astype(int)
            in_prev_map = (prev_x >= 0) & (prev_x < prev_w) & (prev_y >= 0) & (prev_y < prev_h)
            valid_prev_x = np.clip(prev_x, 0, prev_w - 1)
            valid_prev_y = np.clip(prev_y, 0, prev_h - 1)
            prev_values = self.previous_updating_map[valid_prev_y, valid_prev_x]
            changed_in_range = in_prev_map & (prev_values != current_map)
            changed_out_range = ~in_prev_map & (current_map != UNKNOWN)
            changed_mask = changed_in_range | changed_out_range
            changed_y, changed_x = np.where(changed_mask)
            if len(changed_x) > 0:
                changed_world_x = curr_origin_x + changed_x * cell_size
                changed_world_y = curr_origin_y + changed_y * cell_size
                coords = np.round(np.column_stack((changed_world_x, changed_world_y)), 1)
                changed_coords = set(map(tuple, coords))
            else:
                changed_coords = set()

        if len(changed_coords) > 0:
            self._changed_coords_tree = cKDTree(np.array(list(changed_coords)))
        else:
            self._changed_coords_tree = None

        self.previous_updating_map = current_map.copy()
        self.previous_map_origin = current_origin

        return changed_coords

    def is_near_changed_region(self, coords, changed_coords=None, threshold=5.0):
        if self._changed_coords_tree is None:
            return False

        coords = np.array(coords).reshape(1, -1)
        distance, _ = self._changed_coords_tree.query(coords, k=1)
        return distance[0] <= threshold

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        self._update_counter += 1
        changed_coords = self.get_changed_region_mask(updating_map_info)
        visible_graphs = extract_visible_graph_from_map(updating_map_info)

        new_node_list = []
        existing_node_list = []
        special_node_list = []

        new_visible_points = get_coords_from_cell_position(
            np.array([coord for contour in visible_graphs for coord in contour["contour_points"]]),
            updating_map_info
        )
        self.remove_history_node(robot_location, new_visible_points, updating_map_info)
        for contour in visible_graphs:
            contour_points = contour["contour_points"]
            contour_edges = contour["edges"]
            assert len(contour_points) == len(contour_edges), print("the length of point and edge is wrong")
            for index in range(len(contour_edges)):
                point_coord = get_coords_from_cell_position(np.array(contour_points[index]), updating_map_info)
                egdes_coord = get_coords_from_cell_position(np.array(contour_edges[index]), updating_map_info)
                node = self.check_node_exist_in_dict(point_coord)
                if node is None:
                    if self.is_near_changed_region(point_coord, changed_coords, threshold=5.0):

                        node = self.add_node_to_dict(point_coord, frontiers, egdes_coord, updating_map_info)
                        if node:
                            if node.id == self.robot_id:
                                continue
                            node.last_seen = self._update_counter
                            new_node_list.append(node)
                else:
                    if node.id == self.robot_id:
                        continue
                    node.noise_flag = max(0, node.noise_flag - 1)
                    node.last_seen = self._update_counter
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
                    node.pending_edge_coords = egdes_coord
                    existing_node_list.append(node)

        special_node_list.append(self.add_node_to_dict(robot_location, frontiers, [], updating_map_info, is_robot=True))

        if not self.check_node_exist_in_dict(self.goal_point):
            special_node_list.append(self.add_node_to_dict(self.goal_point, frontiers, [], updating_map_info, is_goal=True))
        else:
            special_node_list.append(self.check_node_exist_in_dict(self.goal_point))

        for node in existing_node_list:
            self.update_node_neighbors(node)

        for node in new_node_list:
            self._process_pending_edges(node)

        if frontiers:
            cluster_points = cluster_frontiers(frontiers)
            for points in cluster_points:
                points = np.round(np.array(points),1)
                if np.linalg.norm(robot_location - points) < UPDATING_MAP_SIZE//2 - 5:
                    existing_node = self.check_node_exist_in_dict(points)
                    if existing_node is None:
                        node = self.add_node_to_dict(points, frontiers, [], updating_map_info)
                        if node:
                            special_node_list.append(node)

        for node in special_node_list:
            if np.linalg.norm(node.coords - robot_location) < (
                    UPDATING_MAP_SIZE / 2):
                self.init_node_neighbors(node, updating_map_info)

        if IS_DEBUG:
            self.validate_graph_integrity()


    def validate_graph_integrity(self):
        errors = []
        for node_id, node in self.id_to_node.items():
            for neighbor_id in list(node.neighbor_ids):
                neighbor_node = self.id_to_node.get(neighbor_id)
                if neighbor_node is None:
                    errors.append(f"Node {node_id} has neighbor {neighbor_id} which doesn't exist")
                    node.neighbor_ids.discard(neighbor_id)
                    node.neighbor_coords_dist.pop(neighbor_id, None)
                elif node_id not in neighbor_node.neighbor_ids:
                    errors.append(f"Node {node_id} -> {neighbor_id} but {neighbor_id} -> {node_id} missing")

                    neighbor_node.add_neighbor(node_id, node.coords)

        if errors:
            for err in errors:
                logger.warning(f"Graph integrity issue: {err}")
        return len(errors) == 0

    def repair_broken_edges(self):
        repaired_count = 0
        removed_count = 0

        for node_id, node in list(self.id_to_node.items()):
            neighbors_to_remove = []
            for neighbor_id in list(node.neighbor_ids):
                neighbor_node = self.id_to_node.get(neighbor_id)
                if neighbor_node is None:
                    neighbors_to_remove.append(neighbor_id)
                    removed_count += 1
                elif node_id not in neighbor_node.neighbor_ids:
                    neighbor_node.add_neighbor(node_id, node.coords)
                    repaired_count += 1

            for neighbor_id in neighbors_to_remove:
                node.neighbor_ids.discard(neighbor_id)
                node.neighbor_coords_dist.pop(neighbor_id, None)

        if repaired_count > 0 or removed_count > 0:
            print(f"Graph repair: added {repaired_count} missing edges, removed {removed_count} broken edges")
            logger.info(f"Graph repair: added {repaired_count} missing edges, removed {removed_count} broken edges")

        return repaired_count, removed_count
class LocalNode:
    def __init__(self, id, coords, frontiers, edge_coords, updating_map_info):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.visited = 0
        self.noise_flag = 0
        self.id = id
        self.last_seen = 0

        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)

        self.neighbor_ids = set()  # set of node IDs (int)
        self.neighbor_coords_dist = dict()  # dict of (x, y) tuples

        self.pending_edge_coords = edge_coords

    def get_neighbor_coords(self):
        return list(self.neighbor_coords_dist.values())

    def add_neighbor(self, neighbor_id, coords):
        if neighbor_id != self.id:
            self.neighbor_ids.add(neighbor_id)
            self.neighbor_coords_dist[neighbor_id] = coords

    def remove_neighbor_by_id(self, neighbor_id):
        self.neighbor_ids.discard(neighbor_id)
        self.neighbor_coords_dist.pop(neighbor_id, None)

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if self.coords[0] < updating_map_info.map_origin_x or self.coords[1] < updating_map_info.map_origin_y or \
            self.coords[0] > updating_map_info.map_origin_x + updating_map_info.cell_size * updating_map_info.map.shape[1] or \
            self.coords[1] > updating_map_info.map_origin_y + updating_map_info.cell_size * updating_map_info.map.shape[0]:
            return set()

        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()
            frontiers = np.array(list(frontiers)).reshape(-1, 2)

            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)

            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)

            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info):
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if not is_frontier(np.array([frontier[0], frontier[1]]), map_info):
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)

        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)

        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]

        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:
                self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)

        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()
