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
    """
    节点管理器类，负责管理探索环境中的所有节点
    使用四叉树数据结构存储节点，实现路径规划和图更新功能
    """
    def __init__(self, plot=False):
        """
        初始化节点管理器

        参数:
        plot: 是否启用绘图功能
        """
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
        self._update_counter = 0 # not use

    def _make_bounds(self, coords):
        """
        将坐标转换为 R-tree 边界格式
        coords: np.ndarray(2,) or list
        return: (min_x, min_y, max_x, max_y)
        """
        return (float(coords[0]), float(coords[1]), float(coords[0]), float(coords[1]))


    def check_node_exist_in_dict(self, coords):
        """
        检查指定坐标的节点是否存在于节点字典中

        参数:
        coords: np.ndarray(2,)

        返回:
        如果节点存在，返回节点对象；否则返回None
        """
        bounds = self._make_bounds(coords)
        exist = list(self.nodes_tree.intersection(bounds))
        if not exist:
            return None
        # 精确匹配（处理浮点误差）
        for node_id in exist:
            node = self.id_to_node.get(node_id)
            if node and np.allclose(node.coords, coords, atol=1e-6):
                return node
        return None

    def get_node_id_by_coords(self, coords):
        """通过坐标获取节点 ID"""
        bounds = self._make_bounds(coords)
        result = list(self.nodes_tree.intersection(bounds))
        for node_id in result:
            node = self.id_to_node.get(node_id)
            if node and np.allclose(node.coords, coords, atol=1e-6):
                return node_id
        return None

    def get_node_by_id(self, node_id):
        """通过 ID 获取节点"""
        return self.id_to_node.get(node_id)

    def get_node(self, coords):
        """
        获取节点对象
        如果输入是id，则通过id查找节点
        如果输入是坐标，则通过坐标查找节点
        """
        if isinstance(coords, int):
            return self.get_node_by_id(coords)
        else:
            return self.check_node_exist_in_dict(coords)

    def add_node_to_dict(self, coords, frontiers, edges_coords, updating_map_info, is_robot=False, is_goal=False, is_frontier=False):
        """
        向节点字典中添加新节点,

        参数:
        coords: 新节点的坐标 np.ndarray(2,)
        edges: 边坐标
        updating_map_info: 局部地图信息

        返回:
        新创建的节点对象
        """
        
        bounds = self._make_bounds(coords)
        if is_robot:
            node = LocalNode(self.robot_id, coords, frontiers, edges_coords, updating_map_info)
            if self.robot_id in self.id_to_node:
                # 已存在，先删除旧的再插入新的
                old_node = self.id_to_node[self.robot_id]
                old_bounds = self._make_bounds(old_node.coords)
                self.nodes_tree.delete(self.robot_id, old_bounds)
                self.id_to_node[self.robot_id] = node
                self.nodes_tree.insert(self.robot_id, bounds)
            else:
                self.id_to_node[self.robot_id] = node
                self.nodes_tree.insert(self.robot_id, bounds)
            return node
        # 非机器人节点检查是否已存在
        existing = self.check_node_exist_in_dict(coords)
        if existing is not None:
            print("节点已存在，error")
            return existing
        if is_goal:
            if self.goal_id is None:
                self.id_tracker += 1
                self.goal_id = self.id_tracker
            node = LocalNode(self.goal_id, coords, frontiers, edges_coords, updating_map_info)
            print("Goal node added:", self.goal_id, coords)
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
        """
        处理节点的待定边坐标，转换为邻居 ID
        """
        edge_coords = node.pending_edge_coords
        if edge_coords is None or len(edge_coords) == 0:
            return
        
        edge_coords = np.array(edge_coords)
        if edge_coords.ndim == 1:
            edge_coords = edge_coords.reshape(-1, 2)
        for coord in edge_coords:
            neighbor_id = self.get_node_id_by_coords(coord)
            if neighbor_id is not None and neighbor_id != node.id:
                # 建立双向连接
                node.add_neighbor(neighbor_id, coord)
                self.id_to_node[neighbor_id].add_neighbor(node.id, node.coords)

        node.pending_edge_coords = None  # 清除待定数据

    def remove_node_from_dict(self, node):
        """
        从节点字典中移除节点，同时更新相邻节点的连接关系

        参数:
        node: 要移除的节点
        """
        # 先删除邻居节点中的连接
        for neighbor_id in list(node.neighbor_ids):
            neighbor_node = self.id_to_node.get(neighbor_id)
            if neighbor_node:
                neighbor_node.remove_neighbor_by_id(node.id)
        # 从 R-tree 和字典中删除
        bounds = self._make_bounds(node.coords)
        self.nodes_tree.delete(node.id, bounds)
        self.id_to_node.pop(node.id, None)

    def within_bb(self, min_x, min_y, max_x, max_y):
        """范围查询，返回节点列表"""
        bounds = (min_x, min_y, max_x, max_y)
        result_ids = list(self.nodes_tree.intersection(bounds))
        return [self.id_to_node[nid] for nid in result_ids if nid in self.id_to_node]

    def get_neighbor_nodes(self, node):
        """获取节点的所有邻居节点对象"""
        return [self.id_to_node[nid] for nid in node.neighbor_ids if nid in self.id_to_node]

    def remove_history_node(self, center_point, new_points, updating_map_info):
        """
        先清空之前存在而现在不是可视点的点
        :param center_coords: 更新中心点 np.array|list
        :param new_points: 当前观测的可视点
        :return:
        """
        # 计算 updating_map 的有效内部区域（去掉边界缓冲区）
        buffer_size = 3.0  # 边界缓冲区大小（米）
        map_min_x = updating_map_info.map_origin_x + buffer_size
        map_min_y = updating_map_info.map_origin_y + buffer_size
        map_max_x = updating_map_info.map_origin_x + updating_map_info.cell_size * (updating_map_info.map.shape[1] - 1) - buffer_size
        map_max_y = updating_map_info.map_origin_y + updating_map_info.cell_size * (updating_map_info.map.shape[0] - 1) - buffer_size
        
        # 查询 updating_map 内部区域的节点
        nodes_in_range = self.within_bb(
            min_x=map_min_x,
            min_y=map_min_y,
            max_x=map_max_x,
            max_y=map_max_y
        )

        new_points_set = set()
        for point in new_points:
            new_points_set.add((round(point[0], 1), round(point[1], 1)))
        # 筛选并删除不存在的点
        nodes_to_remove = []
        current_step = self._update_counter

        for node in nodes_in_range:
            # 跳过特殊节点
            if node.id == self.robot_id or node.id == self.goal_id:
                continue
            if self.goal_point is not None and np.allclose(node.coords, self.goal_point, atol=0.2):
                continue
            
            px, py = round(node.coords[0], 1), round(node.coords[1], 1)
            
            # 检查节点位置是否仍然是 FREE 区域
            cell = get_cell_position_from_coords(node.coords, updating_map_info, check_negative=False)
            cell_x, cell_y = int(cell[0]), int(cell[1])
            
            cell_value = updating_map_info.map[cell_y, cell_x]
            
            # 条件1：节点位置变成了障碍物 -> 立即删除
            if cell_value == OCCUPIED:
                nodes_to_remove.append(node)
                continue
            
            # 条件2：节点不在新观测点中
            if (px, py) not in new_points_set:
                steps_since_seen = current_step - node.last_seen
                # 检查是否没有邻居
                has_valid_neighbors = len(node.neighbor_ids) > 0
                
                if not has_valid_neighbors:
                    # 没有邻居的孤立节点，使用更严格的阈值
                    # 如果超过 NOISE_THRESHOLD 步没被观测到，增加 noise_flag
                    # if steps_since_seen >= 1:
                    node.noise_flag += 1
                    if node.noise_flag >= NOISE_THRESHOLD + 1:
                        nodes_to_remove.append(node)
                else:
                    # 有邻居但不在新观测中，可能是暂时遮挡
                    # 使用更宽松的阈值
                    # if steps_since_seen >= 2:
                    node.noise_flag += 1
                    if node.noise_flag >= NOISE_THRESHOLD + 1:
                        nodes_to_remove.append(node)
            else:
                # 节点在新观测中，重置噪点标记
                node.noise_flag = max(0, node.noise_flag - 2)
                # node.last_seen = current_step
        
        # 执行删除
        for node_to_remove in nodes_to_remove:
            existing_node = self.check_node_exist_in_dict(node_to_remove.coords)
            if existing_node:
                self.remove_node_from_dict(existing_node)

    def update_node_neighbors(self, node):
        """
        更新传入节点的邻居边
        """
        node_id_to_remove = []
        # 检查现在的邻居边是否还存在
        for neighbor_id in node.neighbor_ids:
            if neighbor_id not in self.id_to_node:
                print("NM: error the neighbor_id is not in dist", neighbor_id, node.neighbor_coords_dist[neighbor_id])
                node_id_to_remove.append(neighbor_id)
        for neighbor_id in node_id_to_remove:
            node.neighbor_ids.discard(neighbor_id)
            node.neighbor_coords_dist.pop(neighbor_id, None)
        # 更新新的邻居边
        self._process_pending_edges(node)

    def init_node_neighbors(self, node, updating_map_info):
        """
        初始化特殊节点的邻居关系
        """
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
        """
        检测地图变化区域（向量化优化版本）
        """
        current_map = updating_map_info.map
        current_origin = (updating_map_info.map_origin_x, updating_map_info.map_origin_y)
        cell_size = updating_map_info.cell_size
        
        if self.previous_updating_map is None:
            # 第一次：获取所有非未知区域
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
            
            # 向量化坐标计算
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
        
        # 构建 KD-Tree
        if len(changed_coords) > 0:
            self._changed_coords_tree = cKDTree(np.array(list(changed_coords)))
        else:
            self._changed_coords_tree = None
        
        self.previous_updating_map = current_map.copy()
        self.previous_map_origin = current_origin
        
        return changed_coords
    
    def is_near_changed_region(self, coords, changed_coords=None, threshold=5.0):
        """
        使用 KD-Tree 快速检查（O(log n)）
        """
        if self._changed_coords_tree is None:
            return False
        
        coords = np.array(coords).reshape(1, -1)
        distance, _ = self._changed_coords_tree.query(coords, k=1)
        return distance[0] <= threshold

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        """
        更新可视节点图，添加新节点并更新已有节点的信息（更新边界、更新邻居）

        参数:
        robot_location: 机器人当前世界位置
        frontiers: 边界点集合
        updating_map_info: 用于更新的局部地图信息
        map_info: 完整地图信息
        """
        self._update_counter += 1  # 更新计数器 
        # self.remove_noise_nodes(robot_location, updating_map_info)
        # 检测地图变化区域
        changed_coords = self.get_changed_region_mask(updating_map_info)
        visible_graphs = extract_visible_graph_from_map(updating_map_info)
        # visualize_contours(updating_map_info.map, visible_graphs)
        all_node_list = []  # 初始化所有节点列表
        node_update_neighbors = []  # 存储需要更新邻居的节点
        # 先清空之前存在而现在不是可视点的点
        new_visible_points = get_coords_from_cell_position(
            np.array([coord for contour in visible_graphs for coord in contour["contour_points"]]), 
            updating_map_info
        )
        self.remove_history_node(robot_location, new_visible_points, updating_map_info)
        # 先添加新节点并将边存入待定列表，久节点则更新边待定列表等待统一更新
        for contour in visible_graphs:
            contour_points = contour["contour_points"]
            contour_edges = contour["edges"]
            assert len(contour_points) == len(contour_edges), print("the length of point and edge is wrong")
            for index in range(len(contour_edges)):
                point_coord = get_coords_from_cell_position(np.array(contour_points[index]), updating_map_info)
                egdes_coord = get_coords_from_cell_position(np.array(contour_edges[index]), updating_map_info)  # N*2
                node = self.check_node_exist_in_dict(point_coord)
                if node is None:
                    if self.is_near_changed_region(point_coord, changed_coords, threshold=5.0):
                        # 创建新节点并添加到字典
                        node = self.add_node_to_dict(point_coord, frontiers, egdes_coord, updating_map_info)
                        if node:
                            node.last_seen = self._update_counter
                else:
                    # 如果存在则更新可见边
                    node.noise_flag = max(0, node.noise_flag - 1) 
                    node.last_seen = self._update_counter  # 更新最后观测时间
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
                    node.pending_edge_coords = egdes_coord
                    node_update_neighbors.append(node)
        # 为终点单独增加
        all_node_list.append(self.add_node_to_dict(robot_location, frontiers, [], updating_map_info, is_robot=True))
        
        if not self.check_node_exist_in_dict(self.goal_point):
            all_node_list.append(self.add_node_to_dict(self.goal_point, frontiers, [], updating_map_info, is_goal=True))
        else:
            all_node_list.append(self.check_node_exist_in_dict(self.goal_point))

        # 更新已有节点的边
        for node in node_update_neighbors:
            self.update_node_neighbors(node)

        # 增加cluster的聚类节点
        if frontiers:
            cluster_points = cluster_frontiers(frontiers)
            for points in cluster_points:
                points = np.round(np.array(points),1)
                if np.linalg.norm(robot_location - points) < UPDATING_MAP_SIZE//2 - 5:
                    existing_node = self.check_node_exist_in_dict(points)
                    if existing_node is None:
                        node = self.add_node_to_dict(points, frontiers, [], updating_map_info)
                        if node:
                            all_node_list.append(node)

        # 更新节点的邻居关系
        for node in all_node_list:  # 遍历所有节点
            # 如果节点需要更新邻居且在传感器范围内（因为这是最新的数据）
            if np.linalg.norm(node.coords - robot_location) < (
                    UPDATING_MAP_SIZE / 2):
                self.init_node_neighbors(node, updating_map_info)


class LocalNode:
    def __init__(self, id, coords, frontiers, edge_coords, updating_map_info):
        """
        初始化节点

        参数:
        coords: 节点坐标
        frontiers: 边界点集合
        updating_map_info: 用于更新的地图信息
        """
        self.coords = coords    # 节点坐标
        self.utility_range = UTILITY_RANGE  # 效用范围（可观察边界点的最大距离）16
        self.utility = 0        # 初始效用值为0  定义为其效用范围内的边界点数量
        self.visited = 0
        self.noise_flag = 0
        self.id = id      # 节点ID
        self.last_seen = 0  # 新增：最后一次被观测到的时间戳
        # 初始化可观察边界点
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        # 可见边节点 只有目标节点 里面全是元组
        self.neighbor_ids = set()  # set of node IDs (int)
        self.neighbor_coords_dist = dict()  # dict of (x, y) tuples
        # 暂存初始边坐标，等 NodeManager 分配 ID 后再处理
        self.pending_edge_coords = edge_coords

    def get_neighbor_coords(self):
        """
        获取邻居坐标
        return List[Tuple[float, float]]
        """
        return list(self.neighbor_coords_dist.values())

    def add_neighbor(self, neighbor_id, coords):
        """通过 ID 添加邻居，并加入字典"""
        if neighbor_id != self.id:
            self.neighbor_ids.add(neighbor_id)
            self.neighbor_coords_dist[neighbor_id] = coords

    def remove_neighbor_by_id(self, neighbor_id):
        """通过 ID 移除邻居，并从字典中删除"""
        self.neighbor_ids.discard(neighbor_id)
        self.neighbor_coords_dist.pop(neighbor_id, None)

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        """
        初始化节点在设定效用范围内(16格子)可观察的边界点

        参数:
        frontiers: 边界点集合
        updating_map_info: 用于更新的地图信息

        返回:
        节点自身可观察边界点集合
        """
        if self.coords[0] < updating_map_info.map_origin_x or self.coords[1] < updating_map_info.map_origin_y or \
            self.coords[0] > updating_map_info.map_origin_x + updating_map_info.cell_size * updating_map_info.map.shape[1] or \
            self.coords[1] > updating_map_info.map_origin_y + updating_map_info.cell_size * updating_map_info.map.shape[0]:
            return set()

        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()    # 初始化可观察边界点集合
            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            # 计算节点到所有边界点的距离
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            # 考察在效用范围内的边界点
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            self.utility = len(observable_frontiers)        # 设置效用值为可观察边界点数量
            # 如果效用值小于最小阈值 1 ，设为0
            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            return observable_frontiers

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info):
        """
        更新节点可观察的边界点，包括删除不是边界的边界点，更新新的传入的frontiers 边界点

        参数:
        frontiers: 边界点集合
        updating_map_info: 用于更新的地图信息
        map_info: 完整地图信息
        """
        # 移除已经不是边界点的点
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if not is_frontier(np.array([frontier[0], frontier[1]]), map_info):
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        # 添加新的可观察边界点
        new_frontiers = frontiers - self.observable_frontiers  # 计算新边界点集合 都是集合所以能相减
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        # 计算到新边界点的距离
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        # 选择在效用范围内的新边界点
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        # 检查每个新边界点是否可见
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:  # 如果没有碰撞
                self.observable_frontiers.add((point[0], point[1]))
        # 更新效用值
        self.utility = len(self.observable_frontiers)
        # 如果效用值小于最小阈值，设为0
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()  # 清空可观察边界点

