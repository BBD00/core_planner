import time

import numpy as np
import torch
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
        # 终点
        self.goal_point = None

        # location （世界坐标系）
        self.location = None

        # map related parameters
        self.cell_size = CELL_SIZE  # 0.4m
        self.node_resolution = NODE_RESOLUTION  # 4.0m
        self.updating_map_size = UPDATING_MAP_SIZE  # 96m

        # map and updating map
        self.map_info = None
        self.updating_map_info = None   # 以车辆为中心的局部地图 96m*96m

        # frontiers
        self.frontier = set()

        # node managers
        self.node_manager = NodeManager(plot=self.plot)

        # graph
        self.node_coords, self.utility, self.guidepost, self.goal_distance = None, None, None, None
        self.current_index, self.adjacent_matrix, self.neighbor_indices = None, None, None

        # self.travel_dist = 0
        #
        # self.episode_buffer = []
        # for i in range(15):
        #     self.episode_buffer.append([])
        #
        # if self.plot:
        #     self.trajectory_x = []
        #     self.trajectory_y = []

    def update_map(self, map_info):
        # no need in training because of shallow copy
        self.map_info = map_info

    def update_updating_map(self, location):
        """更新智能体周围的局部地图"""
        self.updating_map_info = self.get_updating_map(location)

    def update_location(self, location):
        """
        更新智能体位置并更行移动距离

        参数:
            location: 新位置坐标
        """
        self.location = location
        node = self.node_manager.nodes_dict.find(location.tolist())
        if self.node_manager.nodes_dict.__len__() == 0:
            # print("Warning no point is found")
            pass
        else:
            # 修改
            # node.data.set_visited()
            pass
    def update_frontiers(self):
        """在局部地图中更新边界点集合(探索的候选点)"""
        self.frontier = get_frontier_in_map(self.updating_map_info)
        
    def get_updating_map(self, location):
        """
        获取智能体周围的局部地图

        参数:
            location: 当前世界位置坐标

        返回:
            更新的局部地图信息
        """
        # the map includes all nodes that may be updating
        # 世界坐标系下计算局部地图的边界 左下和右上，以车辆当前位置为中心
        updating_map_origin_x = (location[0] - self.updating_map_size / 2)
        updating_map_origin_y = (location[1] - self.updating_map_size / 2)

        updating_map_top_x = updating_map_origin_x + self.updating_map_size
        updating_map_top_y = updating_map_origin_y + self.updating_map_size
        # 世界坐标系下处理边界情况，确保不超出全局地图范围
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
        # 对齐到网格
        updating_map_origin_x = (updating_map_origin_x // self.cell_size + 1) * self.cell_size
        updating_map_origin_y = (updating_map_origin_y // self.cell_size + 1) * self.cell_size
        updating_map_top_x = (updating_map_top_x // self.cell_size) * self.cell_size
        updating_map_top_y = (updating_map_top_y // self.cell_size) * self.cell_size

        updating_map_origin_x = np.round(updating_map_origin_x, 1)
        updating_map_origin_y = np.round(updating_map_origin_y, 1)
        updating_map_top_x = np.round(updating_map_top_x, 1)
        updating_map_top_y = np.round(updating_map_top_y, 1)
        # 计算局部地图在全局地图中的索引位置
        updating_map_origin = np.array([updating_map_origin_x, updating_map_origin_y])
        updating_map_origin_in_global_map = get_cell_position_from_coords(updating_map_origin, self.map_info)

        updating_map_top = np.array([updating_map_top_x, updating_map_top_y])
        updating_map_top_in_global_map = get_cell_position_from_coords(updating_map_top, self.map_info)
        # 从全局地图中提取局部地图
        updating_map = self.map_info.map[
                    updating_map_origin_in_global_map[1]:updating_map_top_in_global_map[1]+1,
                    updating_map_origin_in_global_map[0]:updating_map_top_in_global_map[0]+1]
        # 创建局部地图信息对象
        updating_map_info = MapInfo(updating_map, updating_map_origin_x, updating_map_origin_y, self.cell_size)

        return updating_map_info

    def update_planning_state(self, map_info, location):
        """
        综合更新地图、位置和图结构
        更新规划状态，包括节点坐标、效用值等
        参数:
            map_info: 地图信息
            location: 当前位置
        返回：
            四叉树中所有节点坐标、所有节点效用值、所有节点的导航标记（有用为1）、占用情况（占用为1，机器人为-1）、邻接矩阵（有邻居为0）、当前索引、邻居索引n*1

        """

        self.update_map(map_info)                   # 更新Agent内部的map_info 栅格数据
        self.update_location(location)              # 更新Agent内部的location(世界坐标系)，并设置节点visited
        self.update_updating_map(self.location)     # 更新Agent内部的局部地图   栅格数据
        self.update_frontiers()                     # 更新Agent的局部地图的边界点集合 世界坐标
        # 更新节点管理器的图结构
        self.node_manager.update_graph(self.location,
                                       self.frontier,
                                       self.updating_map_info,
                                       self.map_info)
        self.node_coords, self.utility, self.goal_distance, self.adjacent_matrix, self.current_index, self.neighbor_indices = \
            self.update_observation()

    def update_observation(self):
        """
        更新观测 获取完整的节点图信息，用于规划和可视化

        参数:
        robot_location: 当前机器人位置
        robot_locations: 所有机器人的位置列表
        占用情况：-1代表自己，1代表其他机器人

        返回:
        所有四叉树中节点坐标、所有节点效用值、所有节点的导航标记（有用为1）、占用情况（占用为1，机器人为-1）、邻接矩阵（有邻居为0）、当前索引、邻居索引n*1
        """
        # 收集所有节点的坐标
        all_node_coords = []
        for node in self.node_manager.nodes_dict.__iter__():    # 遍历四叉树中的所有节点
            all_node_coords.append(node.data.coords)            # 收集节点坐标
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)  # 转换为numpy数组
        utility = []
        # guidepost = [] # 即是否visited
        distance = [] # 与期望点的距离

        n_nodes = all_node_coords.shape[0]      # 节点数量
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)   # 初始化邻接矩阵为全1
        # 使用复数表示坐标，方便后续比较
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        # 构建邻接矩阵和收集效用值
        for i, coords in enumerate(all_node_coords):        # 遍历所有节点坐标
            node = self.node_manager.nodes_dict.find((coords[0], coords[1])).data     # 获取节点
            utility.append(node.utility)                    # 收集节点效用值
            # guidepost.append(node.visited)
            distance.append(np.linalg.norm(node.coords - self.goal_point))
            for neighbor in node.neighbor_edges_set:              # 遍历节点的邻居
                # 在所有节点中查找邻居索引
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)     # 返回二维数组  # [[1] [3]]
                # assert index is not None
                # index = index[0][0]
                # adjacent_matrix[i, index] = 0
                # if index or index == [[0]]:  # 如果找到邻居 单独处理[[0]]因为这个会被视为False
                if index.size > 0:  # TODO 这里貌似会有相同的点出现，应该是fronter的时候产生了相同的点
                    index = index[0][0]  # 获取索引值
                    adjacent_matrix[i, index] = 0  # 在邻接矩阵中标记连接关系（0表示连接）

        utility = np.array(utility)     # 转换效用值为numpy数组
        # guidepost = np.array(guidepost) # 转换guidepost为numpy数组
        distance = np.array(distance)
        # 获取对应的节点索引
        current_index = np.argwhere(node_coords_to_check == self.location[0] + self.location[1] * 1j)[0][0]
        # 找出当前节点的所有邻居索引
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)
        return all_node_coords, utility, distance, adjacent_matrix, current_index, neighbor_indices

    def get_observation(self):
        """
        获取结构化的观察数据，用于智能体决策

        返回:
            填充后的以当前节点为中心的节点元素(360个) 1*360*5 [node_coords, node_utility, node_guidepost, node_occupancy]
            节点元素填充mask 1为填充 1*1*360
            邻接矩阵 1*360*360 1为没有邻居和填充的
            当前节点索引 1*1*1
            当前节点的邻居索引 1*25*1  0为填充的,注意自己也在里面
            当前节点的邻居索引填充mask 1*1*25 1为填充,注意自己也是1


        """
        node_coords = self.node_coords                  # 四叉树所有节点坐标
        node_utility = self.utility.reshape(-1, 1)      # 所有节点效用值
        node_goal_distance = self.goal_distance.reshape(-1, 1)  # 所有节点的导航标记（有用为1）
        current_index = self.current_index              # 当前节点索引
        edge_mask = self.adjacent_matrix                # 邻接矩阵（有邻居为0）
        current_edge = self.neighbor_indices            # 当前节点的邻居（包含自己）（有邻居为0） n*1
        n_node = node_coords.shape[0]                   # 节点数量
        # 计算相对于当前节点的坐标，并进行归一化
        current_node_coords = node_coords[self.current_index]
        # 变成N*2  /96（m）/2
        node_coords = np.concatenate((node_coords[:, 0].reshape(-1, 1) - current_node_coords[0],
                                            node_coords[:, 1].reshape(-1, 1) - current_node_coords[1]),
                                           axis=-1) / UPDATING_MAP_SIZE
        # 归一化效用值 20*3.14 // 1.6 =39
        node_utility = node_utility / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
        # 合并节点特征
        node_inputs = np.concatenate((node_coords, node_utility, node_goal_distance), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
        # 填充节点特征到固定大小  360
        assert node_coords.shape[0] < NODE_PADDING_SIZE, print(node_coords.shape[0], NODE_PADDING_SIZE)
        padding = torch.nn.ZeroPad2d((0, 0, 0, NODE_PADDING_SIZE - n_node))
        node_inputs = padding(node_inputs)
        # 创建节点掩码  是填充的则为1
        node_padding_mask = torch.zeros((1, 1, n_node), dtype=torch.int16).to(self.device)
        node_padding = torch.ones((1, 1, NODE_PADDING_SIZE - n_node), dtype=torch.int16).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)
        # 转换当前索引为张量
        current_index = torch.tensor([current_index]).reshape(1, 1, 1).to(self.device)
        # 转换邻接矩阵为张量并填充
        edge_mask = torch.tensor(edge_mask).unsqueeze(0).to(self.device)
        # 进行右侧和底部填充1
        padding = torch.nn.ConstantPad2d(
            (0, NODE_PADDING_SIZE - n_node, 0, NODE_PADDING_SIZE - n_node), 1)
        edge_mask = padding(edge_mask)
        # 处理当前边缘信息
        # 获取当前节点在自身的邻居列表中的索引  [0][0]返回一个数字  这里修改了与初始不一样，现在不会有自己的index在里面
        # current_in_edge = np.argwhere(current_edge == self.current_index)[0][0]
        current_edge = torch.tensor(current_edge).unsqueeze(0)
        k_size = current_edge.size()[-1]
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 0)
        current_edge = padding(current_edge)
        current_edge = current_edge.unsqueeze(-1)
        # 创建边缘掩码
        edge_padding_mask = torch.zeros((1, 1, k_size), dtype=torch.int16).to(self.device)
        # edge_padding_mask[0, 0, current_in_edge] = 1
        padding = torch.nn.ConstantPad1d((0, K_SIZE - k_size), 1)
        # 最终返回tensor([[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1]]])
        # 其中0代表邻居，前面的1是自己，后面的是填充的
        edge_padding_mask = padding(edge_padding_mask)

        return [node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask]

    def set_goal_point(self, ground_truth_map_info, robot_cell, distance=100):
        """
        从可行区域（值为0的栅格）中随机采样点

        参数：
            grid: 栅格地图（0表示可通行，1表示障碍）
            metadata: 栅格元数据（grid_width, grid_height, grid_resolution）
            num_points: 需要采样的点数
            return_world: 是否返回世界坐标（True），否则返回网格坐标（False）

        返回：
            采样点的坐标列表（格式：[[x1,y1], [x2,y2], ...]）
        """
        ground_truth_map = ground_truth_map_info.map
        # 获取所有可行点的网格坐标（y,x格式）
        free_points_all = np.argwhere(ground_truth_map == 255)
        # 初始化过滤后的点集
        free_points = free_points_all
        # 如果指定了距离约束，过滤符合条件的点
        if distance is not None:
            robot_yx = np.array([robot_cell[1], robot_cell[0]])  # 转换为(y,x)格式
            distances = np.sqrt(np.sum((free_points - robot_yx) ** 2, axis=1))
            free_points = free_points[distances >= distance]
        # 随机选择点
        if len(free_points) == 0:
            print("警告: 没有符合条件的可行点!, 随机生成目标点")
            selected_indices = np.random.choice(len(free_points_all), 1, replace=False)
            selected_points = free_points_all[selected_indices]
        else:
            selected_indices = np.random.choice(len(free_points), 1, replace=False)
            selected_points = free_points[selected_indices]


        self.goal_point = get_coords_from_cell_position(selected_points[:, [1, 0]], ground_truth_map_info)
        self.node_manager.goal_point = self.goal_point


    def select_next_waypoint(self, observation):
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.policy_net(*observation)
        # TODO 修改代码，这里会有pytorch BUG尽管都是-1e8还是会采样到因此只采用有概率的元素进行采样
        # valid_mask = logp > -1e7  # 新增
        # safe_logp = logp.masked_fill(~valid_mask, -float('inf'))    # 新增
        # probs = safe_logp.softmax(dim=-1)  # 新增
        # # action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
        # action_index = torch.multinomial(probs, 1).long().squeeze(1)
        # next_node_index = current_edge[0, action_index.item(), 0].item()
        # next_position = self.node_coords[next_node_index]
        # 新方法
        # 获取有效动作的索引
        try:
            valid_indices = torch.where(logp > -1e7)[1]
            if len(valid_indices) == 0:
                print(f"very error {logp}")
                logger.error(f"very error!!!{logp}")
                edge_num = torch.sum(current_edge > 0)
                valid_indices = torch.tensor(torch.arange(edge_num), device=logp.device)
                logp = torch.ones(1,edge_num)
            # 只计算有效动作的概率
            valid_logp = logp[:, valid_indices]
            valid_probs = valid_logp.softmax(dim=-1)
            # 从有效动作中采样
            sampled_idx = torch.multinomial(valid_probs, 1).long().squeeze(1)
            action_index = valid_indices[sampled_idx]
            next_node_index = current_edge[0, action_index.item(), 0].item()
            next_position = self.node_coords[next_node_index]
        except:
            # 保存为Pickle文件
            error_info = {
                "current_location": self.location,
                "logp": logp,
                "current_edge": current_edge,
                "observation": observation,
                "node_coords": self.node_coords,
                "policy_net_state": self.policy_net.state_dict(),
                "timestamp": datetime.datetime.now(),
                "map_info": self.map_info,
                "updating_map_info": self.updating_map_info,
            }
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assert_error_pro_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(error_info, f)

        # 调试信息准备
        node = self.node_manager.nodes_dict.find((self.location[0], self.location[1]))
        check = np.array(list(node.data.neighbor_edges_set)).reshape(-1, 2)
        # 检查条件
        next_pos_complex = next_position[0] + next_position[1] * 1j
        check_complex = check[:, 0] + check[:, 1] * 1j
        # while next_pos_complex in check_complex:
        condition = next_pos_complex in check_complex
        
        # 当断言失败时保存详细信息
        if not condition:
            # 准备错误信息
            error_info = {
                "next_position": next_position,
                "current_location": self.location,
                "neighbor_edges_set": node.data.neighbor_edges_set,
                "logp": logp,
                "action_index": action_index,
                "current_edge": current_edge,
                "observation": observation,
                "node_coords": self.node_coords,
                "policy_net_state": self.policy_net.state_dict(),
                "timestamp": datetime.datetime.now(),
                "map_info": self.map_info,
                "updating_map_info": self.updating_map_info,
            }
            
            # 保存为Pickle文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assert_error_{timestamp}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(error_info, f)
            
            # 打印提示信息
            print(f"Assertion failed! Full debug info saved to {filename}")
        
        # 原始断言
        assert condition, print(next_position, self.location, node.data.neighbor_edges_set, logp)


        return next_position, action_index

    def plot_env(self):
        plt.switch_backend('agg')

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
        count = 0
        for node, utility in zip(nodes, self.utility):
            # plt.text(node[0], node[1], str(utility), zorder=3)
            # plt.text(node[0],node[1],f"({node[0]},{node[1]}){utility}",  fontsize=5, zorder=3)
            plt.text(node[0],node[1], f"({self.node_coords[count][0]},{self.node_coords[count][1]}){utility}", fontsize=5, zorder=3)
            count += 1
        plt.plot(robot[0], robot[1], 'mo', markersize=16, zorder=5)
        plt.plot(goal_cell[0], goal_cell[1], 'ro', markersize=10, zorder=5)
        for coords in self.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neighbor_coords in node.neighbor_edges_set:
                end = (np.array(neighbor_coords) - coords) / 2 + coords
                plt.plot((np.array([coords[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                               (np.array([coords[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        plt.subplot(1, 3, 3)
        plt.imshow(self.map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(nodes[:, 0], nodes[:, 1], c=self.utility, zorder=2)
        plt.plot(robot[0], robot[1], 'mo', markersize=4, zorder=5)
        plt.plot(goal_cell[0], goal_cell[1], 'ro', markersize=8, zorder=5)
        node = self.node_manager.nodes_dict.find(self.location.tolist()).data
        for neighbor_coords in node.neighbor_edges_set:
            end = (np.array(neighbor_coords) - self.location) + self.location
            plt.plot((np.array([self.location[0], end[0]]) - self.map_info.map_origin_x) / self.cell_size,
                     (np.array([self.location[1], end[1]]) - self.map_info.map_origin_y) / self.cell_size, 'tan', zorder=1)

        # 计算当前节点位置（转换为图像坐标）
        current_node_x = (self.location[0] - self.map_info.map_origin_x) / self.cell_size
        current_node_y = (self.location[1] - self.map_info.map_origin_y) / self.cell_size

        # 在当前位置添加矩形框
        rect_size = UPDATING_MAP_SIZE / CELL_SIZE  # 矩形大小（像素）
        rect = plt.Rectangle((current_node_x - rect_size / 2, current_node_y - rect_size / 2),
                             rect_size, rect_size,
                             linewidth=2, edgecolor='cyan', facecolor='none', zorder=6)
        plt.gca().add_patch(rect)
        # plt.show()


