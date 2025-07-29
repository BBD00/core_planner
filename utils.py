import numpy as np
import imageio
import os
from skimage.morphology import label

import quads
from parameter import *
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
from collections import deque


def get_cell_position_from_coords(coords, map_info, check_negative=True):
    single_cell = False
    if coords.flatten().shape[0] == 2:
        single_cell = True

    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    cell_x = ((coords_x - map_info.map_origin_x) / map_info.cell_size)
    cell_y = ((coords_y - map_info.map_origin_y) / map_info.cell_size)

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)

    if check_negative:
        assert sum(cell_position.flatten() >= 0) == cell_position.flatten().shape[0], print(cell_position, coords, map_info.map_origin_x, map_info.map_origin_y)
    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    """
    获取地图中所有空闲区域的世界坐标

    参数:
    map_info: 地图信息对象

    返回:
    所有空闲区域的世界坐标
    """
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    """
        获取与指定位置相连的空闲区域的二进制地图

        参数:
        location: 指定位置的世界坐标
        map_info: 地图信息对象

        返回:
        二进制地图，表示与指定位置相连的空闲区域 n*n
    """
    # 创建一个二进制地图，FREE区域为1，其他区域为0
    # a binary map for free and connected areas
    # a binary map for free and connected areas
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)  # 默认0是背景，connectivity=2是设置8邻域
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    """
    获取需要更新节点的坐标
    即获取局部地图地图中与传入世界位置相连的可行区域
    参数:
    location: 当前位置的世界坐标
    updating_map_info: 地图信息对象
    check_connectivity: 是否检查连通性

    返回:
    可更新节点的世界坐标和连通的空闲区域地图(如果检查连通性)
    """
    # 计算地图边界的世界坐标
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE
    # 调整边界，使其成为NODE_RESOLUTION的整数倍，方便进行采样得到点云点
    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION
    # 创建均匀分布的坐标网格
    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords)
    # .ravel()将多维数组按行优先顺序
    # x_coords = [1, 2], y_coords = [3, 4]
    # t1 = [[1, 2],  t2 = [[3, 3],
    #        [1, 2]]       [4, 4]]
    # t1.T = [[1, 1],
    #        [2, 2]]
    # t1.T.ravel() = [1, 1, 2, 2] t2.T.ravel() = [3, 4, 3, 4]
    # nodes = [[1, 3],
    #           [1, 4],
    #           [2, 3],
    #           [2, 4]]
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:# 如果需要检查连通性
        # 获取与当前位置相连的空闲区域地图 0,1地图 1是连通
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)    # 实际是整个局部地图都进行
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:   # 如果单元格在连通的空闲区域内
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)    # 只保留连通区域的节点

    return nodes, free_connected_map    # 返回节点坐标和整个地图的连通地图


def get_frontier_in_map(map_info):
    """
    获取地图中的边界区域(已知区域与未知区域的交界)

    参数:
    map_info: 地图信息对象
    voxel_size: 边界单元格大小  1.6

    返回:
    边界区域的坐标集合 N*2
    """
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    # 找出所有空闲单元格的索引 np.where返回索引
    # 先按列优先展开
    # map_info.map = np.array([
    #     [1, 2, 3],  # 第0行
    #     [4, 5, 6]   # 第1行
    # ])
    # 展开后[1, 4, 2, 5, 3, 6]  # 先列后行
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    # 按列展开与前面一致这样可以直接索引
    # cells = [[0 0]    # 坐标(0,0)
    #          [0 1]    # 坐标(0,1)
    #          [0 2]    # 坐标(0,2)
    #          [0 3]    # 坐标(0,3)
    #          [0 4]    # 坐标(0,4)
    #          [1 0]    # 坐标(1,0)
    #          [1 1]    # 坐标(1,1)
    #          ...
    #          [4 4]]   # 坐标(4,4)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords

def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    """
        对边界点进行下采样，减少数据量

        参数:
        data: 边界点坐标
        voxel_size: 下采样体素大小

        返回:
        下采样后的边界点集合
    """
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data

def cluster_frontiers(frontier_set, distance_threshold=2.0, min_points=MIN_CLUSTER_NUM):
    """
    聚合离散的frontier点，忽略小簇

    参数:
        frontier_set: 包含frontier点的集合，每个点是二维坐标 (x, y)
        distance_threshold: 两个点被视为属于同一聚类的最大距离
        min_points: 簇的最小点数阈值（点数小于此值的簇将被忽略）

    返回:
        clustered_frontiers: 聚类后的frontier点集合
    """
    if not frontier_set:
        return set()

    points = list(frontier_set)
    n = len(points)
    visited = [False] * n
    clusters = []
    sq_threshold = distance_threshold ** 2  # 使用平方距离避免开方计算

    # 优化的距离计算函数
    def within_threshold(i, j):
        dx = points[i][0] - points[j][0]
        dy = points[i][1] - points[j][1]
        return dx * dx + dy * dy <= sq_threshold

    for i in range(n):
        if visited[i]:
            continue

        cluster = []
        queue = deque([i])
        visited[i] = True

        while queue:
            idx = queue.popleft()  # O(1)操作
            cluster.append(points[idx])

            for j in range(n):
                if not visited[j] and within_threshold(idx, j):
                    visited[j] = True
                    queue.append(j)

        clusters.append(cluster)

    # 高效计算质心
    centroids = set()
    for cluster in clusters:
        if len(cluster) >= min_points:
            arr = np.array(cluster)
            centroids.add(tuple(arr.mean(axis=0)))

    return centroids

def is_frontier(location, map_info):
    """
    检查指定位置是否为边界点

    参数:
    location: 位置的世界坐标
    map_info: 地图信息对象

    返回:
    布尔值，表示是否为边界点
    """
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
        # 有错
        assert cell[1] - 1 > 0 and cell[0] - 1 > 0 and cell[0] + 2 < map_info.map.shape[1] and cell[1] + 2 < \
               map_info.map.shape[0]
        unknwon = map_info.map[cell[1] - 1:cell[1] + 2, cell[0] - 1: cell[0] + 2] == UNKNOWN
        n = np.sum(unknwon)
        if 1 < n < 8:
            return True
        else:
            return False

def check_collision(start, end, map_info):
    """
    使用Bresenham直线算法检查两点之间是否有障碍物

    参数:
    start: 起点世界坐标
    end: 终点世界坐标
    map_info: 地图信息对象

    返回:
    布尔值，表示是否存在碰撞
    """
    # Bresenham line algorithm checking
    assert start[0] >= map_info.map_origin_x, print(start[0],map_info.map_origin_x)
    assert start[1] >= map_info.map_origin_y, print(start[1],map_info.map_origin_y)
    assert end[0] >= map_info.map_origin_x, print(end[0],map_info.map_origin_x)
    assert end[1] >= map_info.map_origin_y, print(end[1],map_info.map_origin_y)
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1],print(start[0],map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1])
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0],print(start[1],map_info.map_origin_x + map_info.cell_size * map_info.map.shape[0])
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1],print(end[0],map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1])
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0],print(end[1],map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0])
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision


def make_gif(path, n, frame_files, rate, done=None):
    """
    创建GIF动画展示探索过程

    参数:
    path: 保存路径
    n: 标识符
    frame_files: 帧文件列表
    rate: 探索率
    delete_images: 是否删除原始图像
    """
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print(f'{n} gif complete success:{done}\n')

    # Remove files
    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        """
        主要存储传入地图、地图原点、cell_size
        :param map:
        :param map_origin_x:
        :param map_origin_y:
        :param cell_size:
        """
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        """
        主要更新地图、地图原点
        :param map:
        :param map_origin_x:
        :param map_origin_y:
        """
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

def extract_visible_graph_from_map(map_info):
    """
    输入:
        map_info: 包含2D numpy array, 值为 1（障碍物），255（空闲），127（未知）
    输出:
        每个障碍物的可见图（轮廓点 + 无碰撞边）
    """
    # 1. 创建二值障碍物掩码（障碍物=1，其他=0）
    obstacle_mask = np.where(map_info.map == 1, 1, 0).astype(np.uint8)
    # kernel = np.ones((2, 2), np.uint8)
    # eroded = cv2.erode(obstacle_mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    processed_mask = cv2.dilate(obstacle_mask, kernel, iterations=3)
    # 转换为8位图像
    binary_map = np.uint8((processed_mask == 1) * 255)

    # 查找轮廓
    # contours：这是一个列表，列表中的每个元素都是一个轮廓。每个轮廓由一组点构成，其数据类型为 numpy.ndarray，形状为 (N, 1, 2)，其中 N 是轮廓上的点的数量，每个点的坐标为 (x, y)。
    # hierarchy：这是一个 numpy.ndarray，形状为 (1, N, 4)，其中 N 是轮廓的数量。它用于表示轮廓之间的层次关系，每个轮廓对应四个整数值 [Next, Previous, First_Child, Parent]，具体含义如下：
    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    visible_graphs = []
    # 收集所有轮廓的采样点并记录轮廓ID
    all_points = []  # 存储所有点 (x, y)
    point_contour_id = []  # 存储每个点所属的轮廓ID
    contour_sampled_points = []  # 存储每个轮廓的采样点（用于最终输出）

    for idx, contour in enumerate(contours):
        epsilon = 0.005 * cv2.arcLength(contour, True)   # 逼近精度阈值，表示原始曲线与近似多边形之间的最大距离。值越小，近似结果越接近原始曲线。
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # 提取轮廓点
        points = approx[:, 0, :]  # 去掉冗余维度
        # 稀疏采样轮廓点（每隔 n 个点）
        sample_rate = 1  # 可根据需要调整
        sampled_points = points[::sample_rate]
        contour_sampled_points.append(sampled_points.tolist())
        # 将点添加到全局列表
        for point in sampled_points:
            all_points.append(tuple(point))  # 转换为元组以便哈希
            point_contour_id.append(idx)  # 记录点所属的轮廓ID

    # 构建所有点之间的可见性图
    n = len(all_points)
    visibility_matrix = np.zeros((n, n), dtype=bool)  # 初始化可见性矩阵
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 预计算所有点对之间的可见性
    for i in range(n):
        for j in range(i + 1, n):  # 避免重复检查
            p1, p2 = all_points[i], all_points[j]
            line_img = np.zeros_like(map_info.map)
            cv2.line(line_img, p1, p2, 127, 1)
            dilated_line = cv2.dilate(line_img, kernel)
            line_mask = dilated_line.astype(bool)

            # 如果线段不穿过任何障碍物，则标记为可见
            if not np.any(map_info.map[line_mask] == 1):
                visibility_matrix[i, j] = True
                visibility_matrix[j, i] = True  # 对称矩阵

    assert np.array_equal(visibility_matrix, visibility_matrix.T), print(f"edge_error {np.array_equal(visibility_matrix, visibility_matrix.T)}")

    # 为每个轮廓构建可见图
    for idx in range(len(contours)):
        # 获取当前轮廓的所有点索引
        contour_point_indices = [i for i, cid in enumerate(point_contour_id) if cid == idx]
        edges = []

        # 检查当前轮廓的点与所有点（包括其他轮廓）的可见性
        for i in contour_point_indices:
            point_edges = []
            for j in range(len(all_points)):
                if i == j:  # 跳过自身
                    continue
                if visibility_matrix[i, j]:
                    point_edges.append(all_points[j])
            edges.append(point_edges)
        visible_graphs.append({
            'object_id': idx,
            'contour_points': contour_sampled_points[idx],
            'edges': edges
        })

    return visible_graphs

def visualize_contours(grid_map, visible_graphs):
    # 可视化
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_map, cmap='gray')
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for contour in visible_graphs:
        contour_points = contour["contour_points"]
        contour_edges = contour["edges"]
        color = colors[contour['object_id'] % len(colors)]
        for index in range(len(contour_edges)):
            point = contour_points[index]
            edges = contour_edges[index]
            ax.plot(point[0], point[1], 'o', color=color)
            for (x,y) in edges:
                    ax.plot([point[0], x], [point[1], y], '-', color=color, linewidth=1)
    ax.set_title("Visible Graphs for Obstacles")
    plt.grid(True)
    plt.show()

def get_nodes_in_range(updating_map_info, nodes_dict):
    min_x = updating_map_info.map_origin_x + 2
    min_y = updating_map_info.map_origin_y + 2
    max_x = min_x + updating_map_info.cell_size * (updating_map_info.map.shape[1] ) - 2
    max_y = min_y + updating_map_info.cell_size * (updating_map_info.map.shape[0] ) - 2

    # bb = quads.BoundingBox(min_x=center_x - UPDATING_MAP_SIZE // 2 + 2,
    #                        min_y=center_y - UPDATING_MAP_SIZE // 2 + 2,
    #                        max_x=center_x + UPDATING_MAP_SIZE // 2 - 2,
    #                        max_y=center_y + UPDATING_MAP_SIZE // 2 - 2)
    bb = quads.BoundingBox(min_x,
                           min_y,
                           max_x,
                           max_y)
    points_in_range = nodes_dict.within_bb(bb)
    return points_in_range

if __name__ =="__main__":
    # 创建房间示例地图 (100x100)
    grid_map = np.full((100, 100), 255, dtype=np.uint8)  # 默认可通行

    # 创建房间墙壁 (外部矩形)
    grid_map[10:90, 10:15] = 1  # 左墙
    grid_map[10:90, 85:90] = 1  # 右墙
    grid_map[10:15, 10:90] = 1  # 上墙

    # 添加内部障碍物（桌子）
    grid_map[30:60, 30:35] = 1
    grid_map[30:60, 45:50] = 1
    grid_map[30:35, 30:50] = 1

    # 添加未知区域
    grid_map[70:75, 70:85] = 127
    map_info = MapInfo(grid_map,0,0,0)
    map_info = np.load(r"E:\A研究生\code\carla_code\VITR\data.npy", allow_pickle=True)
    visible_graphs = extract_visible_graph_from_map(map_info)

    # 可视化
    visualize_contours(grid_map, visible_graphs)

