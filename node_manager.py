import time
import heapq

import matplotlib.pyplot as plt
import numpy as np
from utils import *
from parameter import *
import quads
from loguru import logger
from copy import deepcopy

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
        # 创建四叉树结构存储节点，初始区域为1000×1000
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.goal_point = None
        # self.frontier = None

    def check_node_exist_in_dict(self, coords):
        """
        检查指定坐标的节点是否存在于节点字典中

        参数:
        coords: 要检查的坐标

        返回:
        如果节点存在，返回节点对象；否则返回None
        """
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, frontiers, edges_coords, updating_map_info):
        """
        向节点字典中添加新节点,

        参数:
        coords: 新节点的坐标
        local_frontiers: 当前更新时的边界点集合 全局地图
        edges: 边坐标
        updating_map_info: 局部地图信息

        返回:
        新创建的节点对象
        """
        key = (coords[0], coords[1])
        node = LocalNode(coords, frontiers, edges_coords, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        """
        从节点字典中移除节点，同时更新相邻节点的连接关系

        参数:
        node: 要移除的节点
        """
        # for neighbor_edge_coords in node.data.neighbor_edges_set:
        #     neighbor_node = self.nodes_dict.find(neighbor_edge_coords)
        #     if neighbor_node:
        #         neighbor_node.data.neighbor_edges_set.discard((node.data.coords[0],node.data.coords[1]))
        # self.nodes_dict.remove(node.data.coords.tolist())

        # 创建集合的副本再遍历
        neighbor_edges_copy = node.data.neighbor_edges_set.copy()
        for neighbor_edge_coords in neighbor_edges_copy:
            neighbor_node = self.nodes_dict.find(neighbor_edge_coords)
            if neighbor_node:
                neighbor_node.data.neighbor_edges_set.discard((node.data.coords[0],node.data.coords[1]))
        self.nodes_dict.remove(node.data.coords.tolist())

    def remove_history_node(self, center_point, new_points):
        """
        先清空之前存在而现在不是可视点的点
        :param center_coords: 更新中心点 np.array|list
        :param new_points: 当前观测的可视点
        :return:
        """
        bb = quads.BoundingBox(min_x=center_point[0] - UPDATING_MAP_SIZE / 2 - 3, min_y=center_point[1] - UPDATING_MAP_SIZE / 2 - 3,
                               max_x=center_point[0] + UPDATING_MAP_SIZE / 2 + 3, max_y=center_point[1] + UPDATING_MAP_SIZE / 2 + 3)
        points_in_range = self.nodes_dict.within_bb(bb)
        new_points_set = set()
        for point in new_points:
            new_points_set.add((point[0], point[1]))
        # 筛选并删除不存在的点
        points_to_remove = []
        for point in points_in_range:
            px, py = point.data.coords[0], point.data.coords[1]
            # 检查点是否在新数据中
            if (((px, py) not in new_points_set and \
                    not np.array_equal(np.array([px, py]), self.goal_point)) and \
                    point.data.noise_flag) or len(point.data.neighbor_edges_set)==0:
                    # np.linalg.norm(np.array([px, py] - center_point)) < parameter.UPDATING_MAP_SIZE / 2:
                    # if point.data.step >  self.step - 3:
                    points_to_remove.append(point)
        # logger.debug(f"remove lenth/all_len:{len(points_to_remove)}/{len(points_in_range)}")
        for point in points_to_remove:
            node = self.nodes_dict.find(point)
            if node:
                self.remove_node_from_dict(node)
            elif node is None:
                print(f"point {point}")
                logger.error(f"point {point}")
                continue


    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        """
        更新可视节点图，添加新节点并更新已有节点的信息（更新边界、更新邻居）

        参数:
        robot_location: 机器人当前世界位置
        frontiers: 边界点集合
        updating_map_info: 用于更新的局部地图信息
        map_info: 完整地图信息
        """
        self.remove_noise_nodes(robot_location, updating_map_info)
        visible_graphs = extract_visible_graph_from_map(updating_map_info)
        # visualize_contours(updating_map_info.map, visible_graphs)
        all_node_list = []  # 初始化所有节点列表
        # 先清空之前存在而现在不是可视点的点
        self.remove_history_node(robot_location, get_coords_from_cell_position(np.array([coord for contour in visible_graphs for coord in contour["contour_points"]]), updating_map_info))
        for contour in visible_graphs:
            contour_points = contour["contour_points"]
            contour_edges = contour["edges"]
            assert len(contour_points) == len(contour_edges), print("the length of point and edge is wrong")
            for index in range(len(contour_edges)):
                point_coord = get_coords_from_cell_position(np.array(contour_points[index]), updating_map_info)
                egdes_coord = get_coords_from_cell_position(np.array(contour_edges[index]), updating_map_info)  # N*2
                node = self.check_node_exist_in_dict(point_coord)
                if node is None:
                    # 创建新节点并添加到字典
                    node = self.add_node_to_dict(point_coord, frontiers, egdes_coord, updating_map_info)
                else:
                    # 如果存在则更新可见边
                    node = node.data
                    node.noise_flag = False
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
                    node.update_neighbor_set(egdes_coord, self.nodes_dict)
                    
                all_node_list.append(node)
        # 为起点单独增加
        if not self.check_node_exist_in_dict(robot_location):
            all_node_list.append(self.add_node_to_dict(deepcopy(robot_location), frontiers, [], updating_map_info))
        if not self.check_node_exist_in_dict(self.goal_point):
            all_node_list.append(self.add_node_to_dict(self.goal_point, frontiers, [], updating_map_info))
            # logger.debug("goal_point is move")
        # 增加cluster的聚类节点
        if frontiers:
            cluster_points = cluster_frontiers(frontiers)
            for points in cluster_points:
                points = np.round(np.array(points),1)
                if np.linalg.norm(robot_location - points) < UPDATING_MAP_SIZE//2 - 5:
                    all_node_list.append(self.add_node_to_dict(points, frontiers, [], updating_map_info))

        # 更新节点的邻居关系
        for node in all_node_list:  # 遍历所有节点
            # 如果节点需要更新邻居且在传感器范围内（因为这是最新的数据）
            if np.linalg.norm(node.coords - robot_location) < (
                    UPDATING_MAP_SIZE / 2):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

    def remove_noise_nodes(self, robot_location, updating_map_info):
        """
        # TODO: 耗时太长
        移除噪声节点，主要基于两个特征：
        1. 共线：三个或更多节点在一条直线上时，移除中间所有节点，保留两端点
        2. 距离过近：与其他节点距离过小的节点
        
        参数:
        robot_location: 机器人当前位置
        """
        # 获取当前更新区域内的所有节点
        bb = quads.BoundingBox(min_x=robot_location[0] - UPDATING_MAP_SIZE / 2 - 10, 
                               min_y=robot_location[1] - UPDATING_MAP_SIZE / 2 - 10,
                               max_x=robot_location[0] + UPDATING_MAP_SIZE / 2 + 10, 
                               max_y=robot_location[1] + UPDATING_MAP_SIZE / 2 + 10)
        nodes_in_range = self.nodes_dict.within_bb(bb)
        
        if len(nodes_in_range) < 3:
            return
        
        nodes_to_remove = set()
        
        # 1. 移除共线的中间节点（保留两端点）
        collinear_nodes = self._find_collinear_middle_nodes_simple(nodes_in_range, updating_map_info)
        nodes_to_remove.update(collinear_nodes)
        # 2. 移除距离过近的节点（保留效用更高的）
        close_nodes = self._find_close_noise_nodes(nodes_in_range)
        nodes_to_remove.update(close_nodes)
        # 3. 执行移除（排除特殊节点）
        for node in nodes_to_remove:
            # 保护目标点和机器人位置
            if (not np.array_equal(node.data.coords, self.goal_point) and 
                not np.array_equal(node.data.coords, robot_location)):
                self.remove_node_from_dict(node)
        
        # if nodes_to_remove:
        #     print(f"移除了 {len(nodes_to_remove)} 个噪声节点")
    def _find_collinear_middle_nodes_simple(self, nodes_in_range, updating_map_info, collinear_threshold=0.5):
        """
        简化版本：只移除明显的共线中间节点
        """
        nodes_to_remove = set()
        nodes_list = list(nodes_in_range)
        
        # 只检查每个节点与其直接邻居的共线性
        for node in nodes_list:
            neighbors = []
            for neighbor_coords in node.data.neighbor_edges_set:
                neighbor_node = self.nodes_dict.find(neighbor_coords)
                if neighbor_node and neighbor_node in nodes_list:
                    neighbors.append(neighbor_node)
            
            # 如果邻居数量 >= 2，检查是否为中间节点
            if len(neighbors) >= 2:
                coords = node.data.coords
                
                # 检查所有邻居对是否共线通过当前节点
                for i, neighbor_a in enumerate(neighbors):
                    for neighbor_b in neighbors[i+1:]:
                        coords_a = neighbor_a.data.coords
                        coords_b = neighbor_b.data.coords
                        
                        # 检查当前节点是否在AB直线上
                        dist_to_line = self._point_to_line_distance(coords, coords_a, coords_b)
                        
                        if dist_to_line < collinear_threshold:
                            # 检查是否在线段中间
                            if self._is_point_between(coords, coords_a, coords_b):
                                nodes_to_remove.add(node)
                                break
        
        return nodes_to_remove

    def _is_point_between(self, point, line_start, line_end, tolerance=0.1):
        """检查点是否在线段中间"""
        dist_start = np.linalg.norm(point - line_start)
        dist_end = np.linalg.norm(point - line_end)
        dist_line = np.linalg.norm(line_end - line_start)
        
        return abs(dist_start + dist_end - dist_line) < tolerance

    def _find_close_noise_nodes(self, nodes_in_range, min_distance=0.5):
        """
        找出距离过近的噪声节点
        
        参数:
        nodes_in_range: 范围内的节点列表
        min_distance: 最小距离阈值
        
        返回:
        需要移除的距离过近的噪声节点集合
        """
        nodes_to_remove = set()
        nodes_list = list(nodes_in_range)
        n = len(nodes_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                node_a = nodes_list[i]
                node_b = nodes_list[j]
                
                # 计算距离
                distance = np.linalg.norm(node_a.data.coords - node_b.data.coords)
                
                if distance <= min_distance:
                    # 保留效用更高的节点，移除效用较低的
                    if node_a.data.utility > node_b.data.utility:
                        nodes_to_remove.add(node_b)
                    elif node_b.data.utility > node_a.data.utility:
                        nodes_to_remove.add(node_a)
                    else:
                        # 效用相同时，保留邻居数量更多的
                        if len(node_a.data.neighbor_edges_set) >= len(node_b.data.neighbor_edges_set):
                            nodes_to_remove.add(node_b)
                        else:
                            nodes_to_remove.add(node_a)
        
        return nodes_to_remove

    def _point_to_line_distance(self, point, line_start, line_end):
        """
        计算点到直线的距离
        
        参数:
        point: 目标点坐标
        line_start: 直线起点
        line_end: 直线终点
        
        返回:
        点到直线的距离
        """
        # 向量计算
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        # 避免除零
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-10:
            return np.linalg.norm(point_vec)
        
        # 计算投影长度
        proj_len = np.dot(point_vec, line_vec) / line_len_sq
        
        # 计算投影点
        proj_point = line_start + proj_len * line_vec
        
        # 返回距离
        return np.linalg.norm(point - proj_point)



class LocalNode:
    def __init__(self, coords, frontiers, edge_coords, updating_map_info):
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
        self.noise_flag = True
        # 初始化可观察边界点
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        # 可见边节点 只有目标节点 里面全是元组
        self.initialize_neighbor_set(edge_coords) # self.neighbor_edges_set


    def initialize_neighbor_set(self, edge_coords):
        edge_coords = np.array(edge_coords) if not isinstance(edge_coords, np.ndarray) else edge_coords

        # 处理不同维度的输入
        if edge_coords.size == 0:  # 空输入（如[]）
            edge_coords = np.empty((0, 2))  # 创建0行2列的空数组
        elif edge_coords.ndim == 1:  # 一维输入（如[38.8, 6.8]）
            edge_coords = edge_coords.reshape(-1, 2)  # 自动重塑为(n,2)形状
        else:  # 二维输入（如[[x1,y1], [x2,y2]]）
            if edge_coords.shape[1] != 2:  # 检查列数是否为2
                raise ValueError("2D input must have exactly 2 columns")
        self.neighbor_edges_set = set(map(tuple, edge_coords))


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
            return
        
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
        
    def update_neighbor_set(self, edge_coords, nodes_dict=None):
        """
        更新节点的邻居关系 和update_neighbor_nodes的区别在于这个是传入邻居边并进行更新
        将传入的 edge_coords 追加进当前邻居集合；
        若提供 nodes_dict，则同时移除在全局节点字典中不存在的邻居。
        参数:
        edge_coords: 邻居边
        """
        # 校验并清理不存在的邻居
        if nodes_dict is not None:
            to_remove = []
            for nei in self.neighbor_edges_set:
                if nodes_dict.find(nei) is None:
                    to_remove.append(nei)
            for nei in to_remove:
                self.neighbor_edges_set.discard(nei)

        # 处理不同维度的输入
        if edge_coords.size == 0:  # 空输入（如[]）
            edge_coords = np.empty((0, 2))  # 创建0行2列的空数组
        elif edge_coords.ndim == 1:  # 一维输入
            edge_coords = edge_coords.reshape(-1, 2)  # 自动重塑为(n,2)形状
        else:  # 二维输入（如[[x1,y1], [x2,y2]]）
            if edge_coords.shape[1] != 2:  # 检查列数是否为2
                raise ValueError("2D input must have exactly 2 columns")
        new_edges = list(map(tuple, edge_coords))
        # 追加新边
        for e in new_edges:
            if e != tuple(self.coords):  # 不连接自身
                self.neighbor_edges_set.add(e)

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        """
        更新节点的邻居关系

        参数:
        updating_map_info: 用于更新的地图信息
        nodes_dict: 节点字典
        """
        # min_x = updating_map_info.map_origin_x
        # min_y = updating_map_info.map_origin_y
        # center_x = min_x + updating_map_info.cell_size * (updating_map_info.map.shape[1] // 2)
        # center_y = min_y + updating_map_info.cell_size * (updating_map_info.map.shape[0] // 2)
        if self.coords[0] < updating_map_info.map_origin_x or self.coords[1] < updating_map_info.map_origin_y or \
            self.coords[0] > updating_map_info.map_origin_x + updating_map_info.cell_size * updating_map_info.map.shape[1] or \
            self.coords[1] > updating_map_info.map_origin_y + updating_map_info.cell_size * updating_map_info.map.shape[0]:
            return
        node_in_range = get_nodes_in_range(updating_map_info, nodes_dict)

        for node in node_in_range:
            node = node.data
            if not np.array_equal(node.coords, self.coords):
                if not check_collision(self.coords, node.coords, updating_map_info):
                    self.neighbor_edges_set.add(tuple(node.coords))
                    node.neighbor_edges_set.add(tuple(self.coords))

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
            self.need_update_neighbor = False  # 不需要更新邻居


    def set_visited(self):
        """
        设置节点为已访问状态 将可观测的边界清除、清除utility、update_neighbor标志位置False
        """
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0
