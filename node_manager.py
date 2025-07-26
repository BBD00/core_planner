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
        bb = quads.BoundingBox(min_x=center_point[0] - UPDATING_MAP_SIZE / 2 - 1, min_y=center_point[1] - UPDATING_MAP_SIZE / 2 - 1,
                               max_x=center_point[0] + UPDATING_MAP_SIZE / 2 + 1, max_y=center_point[1] + UPDATING_MAP_SIZE / 2 + 1)
        points_in_range = self.nodes_dict.within_bb(bb)
        new_points_set = set()
        for point in new_points:
            # 将坐标按容差取整，避免浮点精度问题
            # rounded_x = round(round(point[0] / TOLERANCE) * TOLERANCE)
            # rounded_y = round(round(point[1] / TOLERANCE) * TOLERANCE)
            new_points_set.add((point[0], point[1]))
        # 筛选并删除不存在的点
        points_to_remove = []
        for point in points_in_range:
            px, py = point.data.coords[0], point.data.coords[1]

            # 应用相同容差处理
            # rounded_px = round(round(px / TOLERANCE) * TOLERANCE)
            # rounded_py = round(round(py / TOLERANCE) * TOLERANCE)
            # 检查点是否在新数据中
            # if (px, py) not in new_points_set and \
            #         not np.array_equal(np.array([px, py]), self.goal_point):
            #     points_to_remove.append(point)
            # 新策略在范围内的除了特殊点全部移除重新生成
            if not np.array_equal(np.array([px, py]), self.goal_point):
                points_to_remove.append(point)
        # logger.debug(f"remove lenth/all_len:{len(points_to_remove)}/{len(points_in_range)}")
        for point in points_to_remove:
            node = self.nodes_dict.find(point)
            self.remove_node_from_dict(node)


    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        """
        更新可视节点图，添加新节点并更新已有节点的信息（更新边界、更新邻居）

        参数:
        robot_location: 机器人当前世界位置
        frontiers: 边界点集合
        updating_map_info: 用于更新的局部地图信息
        map_info: 完整地图信息
        """
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
                    # 如果节点效用为0或距离机器人较远，则不更新
                    if np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                        pass
                    else:  # 否则更新节点的可观察边界点 旧的删除，新的添加  同时对新边界进行计算如果周围没有边界了就设置不再更新邻居
                        node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
                all_node_list.append(node)
        # 为起点单独增加
        if not self.check_node_exist_in_dict(robot_location):
            all_node_list.append(self.add_node_to_dict(deepcopy(robot_location), frontiers, [], updating_map_info))
        if not self.check_node_exist_in_dict(self.goal_point):
            all_node_list.append(self.add_node_to_dict(self.goal_point, frontiers, [], updating_map_info))
            # logger.debug("goal_point is move")

        # 更新节点的邻居关系
        for node in all_node_list:  # 遍历所有节点
            # 如果节点需要更新邻居且在传感器范围内（因为这是最新的数据）
            if np.linalg.norm(node.coords - robot_location) < (
                    UPDATING_MAP_SIZE / 2):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)



    def update_graph_bck(self, robot_location, frontiers, updating_map_info, map_info):
        """
        更新节点图，添加新节点并更新已有节点的信息（更新边界、更新邻居）

        参数:
        robot_location: 机器人当前世界位置
        frontiers: 边界点集合
        updating_map_info: 用于更新的局部地图信息
        map_info: 完整地图信息
        """
        # 获取当前位置附近需要更新的节点坐标 世界坐标
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        all_node_list = []      # 初始化所有节点列表
        for coords in node_coords:  # 遍历所有需要更新的节点坐标
            node = self.check_node_exist_in_dict(coords)    # 检查节点是否已存在
            if node is None:         # 如果节点不存在
                # 创建新节点并添加到字典
                node = self.add_node_to_dict(coords, frontiers, updating_map_info)
            else:       # 如果节点已存在即之前传感器扫过
                node = node.data        # 获取节点数据
                # 如果节点效用为0或距离机器人较远，则不更新
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:   # 否则更新节点的可观察边界点 旧的删除，新的添加  同时对新边界进行计算如果周围没有边界了就设置不再更新邻居
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info)
            all_node_list.append(node)  # 将节点添加到列表
        # 更新节点的邻居关系
        for node in all_node_list:  # 遍历所有节点
            # 如果节点需要更新邻居且在传感器范围内（因为这是最新的数据）
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    SENSOR_RANGE + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

    def Dijkstra(self, start, boundary=None):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            # assert self.nodes_dict.find(u) is not None

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_set:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            print("destination is not in Dijkstra graph")
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        # h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.linalg.norm(np.array([coords_1[0] - coords_2[0], coords_1[1] - coords_2[1]]))
        # h = np.round(h, 2)
        return h

    def a_star(self, start, destination, max_dist=None):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        open_heap = []
        heapq.heappush(open_heap, (0, (start[0], start[1])))

        while len(open_list) > 0:
            _, n = heapq.heappop(open_heap)
            n_coords = n
            node = self.nodes_dict.find(n).data

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()

                return path, np.round(length, 2)

            costs = np.linalg.norm(np.array(list(node.neighbor_set)).reshape(-1, 2) - [n_coords[0], n_coords[1]],
                                   axis=1)
            for cost, neighbor_node_coords in zip(costs, node.neighbor_set):
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                    heapq.heappush(open_heap, (g[m], m))
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8


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
        # 初始化可观察边界点
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        # 可见边节点 只有目标节点 里面全是元组
        self.initialize_neighbor_set(edge_coords) # self.neighbor_edges_set

        self.need_update_neighbor = True         # 设置需要更新邻居标志

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
        node_in_range = get_nodes_in_range(updating_map_info, nodes_dict)

        for node in node_in_range:
            node = node.data
            if not np.array_equal(node.coords, self.coords):
                if not check_collision(self.coords, node.coords, updating_map_info):
                    self.neighbor_edges_set.add(tuple(node.coords))
                    node.neighbor_edges_set.add(tuple(self.coords))

        # TODO 有一种可能是点是围城障碍物时落在位置区域了
        if self.utility == 0:
            node_cell = get_cell_position_from_coords(np.array(self.coords), updating_map_info)
            if updating_map_info.map[node_cell[1],node_cell[0]] == FREE:
                # logger.warning(f"the {self.coords} need_update_neighbor {self.need_update_neighbor}")
                self.need_update_neighbor = False

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
