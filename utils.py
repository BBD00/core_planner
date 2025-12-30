import numpy as np
import imageio
import os
from skimage.morphology import label
from numba import jit, njit  # 新增导入
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


@njit(cache=True)
def _find_frontier_indices(map_data, unknown_neighbor):
    """Numba 加速的边界点查找"""
    y_len, x_len = map_data.shape
    indices = []
    for y in range(y_len):
        for x in range(x_len):
            if map_data[y, x] == 255:  # FREE
                neighbor_count = unknown_neighbor[y, x]
                if 1 < neighbor_count < 8:
                    indices.append(x * y_len + y)  # 列优先索引
    return np.array(indices, dtype=np.int64)

def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    
    # 使用更高效的卷积计算未知邻居数
    unknown = (map_info.map == UNKNOWN).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    unknown_neighbor = cv2.filter2D(unknown, -1, kernel)
    
    # 使用 Numba 加速的查找
    frontier_indices = _find_frontier_indices(map_info.map, unknown_neighbor)
    
    if frontier_indices.size == 0:
        return set()
    
    # 转换索引为坐标
    frontier_x = frontier_indices // y_len
    frontier_y = frontier_indices % y_len
    frontier_cell = np.column_stack((frontier_x, frontier_y))
    
    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info)
    if FRONTIER_CELL_SIZE != CELL_SIZE:
        return frontier_down_sample(frontier_coords)
    return set(map(tuple, frontier_coords))


def get_stay_count(current_position, past_trajectory_x, past_trajectory_y, window_size=STAY_WINDOW_SIZE, threshold=STAY_DIS_THRESHOLD):
    """
    计算机器人是否在某个地方停留

    参数:
    current_position: 当前位置，格式为 np.array(x,y)
    past_trajectory_x: 历史轨迹的x坐标列表[x1,x2,x3]
    past_trajectory_y: 历史轨迹的y坐标列表[y1,y2,y3]
    window_size: 要比较的历史位置数量，默认为6
    threshold: 判断为"停留"的距离阈值，默认为0.1米

    返回:
    stay_count: 如果当前点与过去多个点很近，则返回1，否则返回0
    """
    # 确保有足够的历史轨迹点
    if len(past_trajectory_x) < window_size:
        return 0
    # 获取最近的window_size个历史位置
    # recent_x = past_trajectory_x[-window_size:]
    # recent_y = past_trajectory_y[-window_size:]
    recent_x = past_trajectory_x
    recent_y = past_trajectory_y
    # 计算当前位置与历史位置的距离
    distances = np.hypot(np.array(recent_x) - current_position[0], np.array(recent_y) - current_position[1])
    close_count = np.sum(distances < threshold)
    return close_count


@njit(cache=True)
def _downsample_core(data, voxel_indices):
    """Numba 加速的下采样核心逻辑"""
    n = data.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    # 手动实现 unique：使用排序 + 去重
    # 将2D索引转换为1D用于排序
    max_y = np.max(voxel_indices[:, 1]) + 1
    keys = voxel_indices[:, 0] * max_y + voxel_indices[:, 1]
    
    # 获取排序索引
    sort_indices = np.argsort(keys)
    sorted_keys = keys[sort_indices]
    
    # 统计唯一值数量
    unique_count = 1
    for i in range(1, n):
        if sorted_keys[i] != sorted_keys[i - 1]:
            unique_count += 1
    
    # 分配结果数组
    result = np.empty((unique_count, 2), dtype=np.float64)
    
    # 填充结果：每个唯一体素取第一个点
    result_idx = 0
    result[result_idx] = data[sort_indices[0]]
    prev_key = sorted_keys[0]
    
    for i in range(1, n):
        if sorted_keys[i] != prev_key:
            result_idx += 1
            result[result_idx] = data[sort_indices[i]]
            prev_key = sorted_keys[i]
    
    return result


def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    """对边界点进行下采样"""
    data = np.asarray(data, dtype=np.float64).reshape(-1, 2)
    if data.shape[0] == 0:
        return set()
    voxel_indices = (data / voxel_size).astype(np.int32)
    result = _downsample_core(data, voxel_indices)
    return set(map(tuple, result))

def cluster_frontiers(frontier_set, distance_threshold=2.0, min_points=MIN_CLUSTER_NUM, max_cluster_size=MAX_CLUSTER_NUM):
    """
    基于网格的快速边界点聚类算法
    
    参数:
        frontier_set: 包含frontier点的集合，每个点是二维坐标 (x, y)
        distance_threshold: 网格大小，用于聚类
        min_points: 簇的最小点数阈值
        max_cluster_size: 最大簇大小，超过此值的簇将被切分
    
    返回:
        clustered_frontiers: 聚类后的frontier点集合
    """
    if not frontier_set:
        return set()

    points = list(frontier_set)
    grid_size = distance_threshold
    
    # 使用字典进行网格聚类 - O(n)
    grid_dict = {}
    for point in points:
        # 计算网格索引
        grid_x = int(point[0] // grid_size)
        grid_y = int(point[1] // grid_size)
        grid_key = (grid_x, grid_y)
        
        if grid_key not in grid_dict:
            grid_dict[grid_key] = []
        grid_dict[grid_key].append(point)
    
    # 合并相邻网格形成簇
    clusters = []
    processed_grids = set()
    
    for grid_key in grid_dict:
        if grid_key in processed_grids:
            continue
            
        # 使用BFS合并相邻网格
        cluster_points = []
        queue = deque([grid_key])
        processed_grids.add(grid_key)
        
        while queue:
            current_grid = queue.popleft()
            cluster_points.extend(grid_dict[current_grid])
            
            # 检查8个相邻网格
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_grid = (current_grid[0] + dx, current_grid[1] + dy)
                    if (neighbor_grid in grid_dict and 
                        neighbor_grid not in processed_grids):
                        processed_grids.add(neighbor_grid)
                        queue.append(neighbor_grid)
        
        if len(cluster_points) >= min_points:
            clusters.append(cluster_points)
    
    # 处理过大的簇 - 简单的空间切分
    final_clusters = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            # 简单的空间切分
            sub_clusters = split_large_cluster_simple(cluster, max_cluster_size)
            final_clusters.extend(sub_clusters)
    
    # 计算质心
    centroids = set()
    for cluster in final_clusters:
        if len(cluster) >= min_points:
            arr = np.array(cluster)
            centroids.add(tuple(arr.mean(axis=0)))
    
    return centroids

def split_large_cluster_simple(cluster_points, max_size):
    """
    简单的大簇切分方法 - 基于空间划分
    
    参数:
        cluster_points: 簇中的点列表
        max_size: 最大簇大小
    
    返回:
        切分后的子簇列表
    """
    if len(cluster_points) <= max_size:
        return [cluster_points]
    
    # 计算边界框
    points_array = np.array(cluster_points)
    min_coords = points_array.min(axis=0)
    max_coords = points_array.max(axis=0)
    
    # 计算需要的分割数
    num_splits = max(2, int(np.ceil(len(cluster_points) / max_size)))
    grid_size = int(np.ceil(np.sqrt(num_splits)))
    
    # 计算每个网格的大小
    x_step = (max_coords[0] - min_coords[0]) / grid_size if max_coords[0] > min_coords[0] else 1.0
    y_step = (max_coords[1] - min_coords[1]) / grid_size if max_coords[1] > min_coords[1] else 1.0
    
    # 如果范围太小，直接平均分割
    if x_step < 0.1 or y_step < 0.1:
        # 简单的线性分割
        chunk_size = max_size
        return [cluster_points[i:i+chunk_size] for i in range(0, len(cluster_points), chunk_size)]
    
    # 空间网格分割
    sub_clusters_dict = {}
    for point in cluster_points:
        grid_x = min(int((point[0] - min_coords[0]) / x_step), grid_size - 1)
        grid_y = min(int((point[1] - min_coords[1]) / y_step), grid_size - 1)
        grid_key = (grid_x, grid_y)
        
        if grid_key not in sub_clusters_dict:
            sub_clusters_dict[grid_key] = []
        sub_clusters_dict[grid_key].append(point)
    
    # 如果某些子簇仍然太大，进一步分割
    final_sub_clusters = []
    for sub_cluster in sub_clusters_dict.values():
        if len(sub_cluster) <= max_size:
            final_sub_clusters.append(sub_cluster)
        else:
            # 递归分割（最多2层递归）
            chunk_size = max_size
            chunks = [sub_cluster[i:i+chunk_size] for i in range(0, len(sub_cluster), chunk_size)]
            final_sub_clusters.extend(chunks)
    
    return final_sub_clusters

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

@njit(cache=True)
def _bresenham_collision_check(x0, y0, x1, y1, map_data, map_height, map_width):
    """
    Numba 加速的 Bresenham 直线碰撞检测核心算法
    
    参数:
    x0, y0: 起点单元格坐标
    x1, y1: 终点单元格坐标
    map_data: 地图数据 (2D numpy array)
    map_height, map_width: 地图尺寸
    
    返回:
    0: 无碰撞 (False)
    1: 障碍物碰撞 (OCCUPIED)
    127: 未知区域 (UNKNOWN)
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map_width and 0 <= y < map_height:
        k = map_data[int(y), int(x)]
        if x == x1 and y == y1:
            break
        if k == 1:  # OCCUPIED
            return 1
        if k == 127:  # UNKNOWN
            return 127
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return 0  # False, 无碰撞


@njit(cache=True)
def _clip_to_boundary(x, y, width, height):
    """
    Numba 加速的边界裁剪
    """
    if x < 0:
        x = 0
    elif x >= width:
        x = width - 1
    if y < 0:
        y = 0
    elif y >= height:
        y = height - 1
    return x, y


@njit(cache=True)
def _world_to_cell(coord_x, coord_y, origin_x, origin_y, cell_size):
    """
    Numba 加速的世界坐标转单元格坐标
    """
    cell_x = int(round((coord_x - origin_x) / cell_size))
    cell_y = int(round((coord_y - origin_y) / cell_size))
    return cell_x, cell_y


def check_collision(start, end, map_info, do_trans=True):
    """
    使用 Numba 加速的 Bresenham 直线算法检查两点之间是否有障碍物

    参数:
    start: 起点世界坐标
    end: 终点世界坐标
    map_info: 地图信息对象
    do_trans: 是否进行坐标转换

    返回:
    False/0: 无碰撞
    OCCUPIED(1): 障碍物碰撞
    UNKNOWN(127): 未知区域
    """
    map_data = map_info.map
    map_height, map_width = map_data.shape
    
    # 坐标转换
    if do_trans:
        start_x, start_y = _world_to_cell(
            float(start[0]), float(start[1]),
            map_info.map_origin_x, map_info.map_origin_y, map_info.cell_size
        )
        end_x, end_y = _world_to_cell(
            float(end[0]), float(end[1]),
            map_info.map_origin_x, map_info.map_origin_y, map_info.cell_size
        )
    else:
        start_x, start_y = int(start[0]), int(start[1])
        end_x, end_y = int(end[0]), int(end[1])
    
    # 检查点是否在地图内
    start_in_map = 0 <= start_x < map_width and 0 <= start_y < map_height
    end_in_map = 0 <= end_x < map_width and 0 <= end_y < map_height
    
    # 如果两点都在地图外，直接返回未知
    if not start_in_map and not end_in_map:
        return UNKNOWN
    
    # 边界裁剪
    if not start_in_map:
        start_x, start_y = _clip_to_boundary(start_x, start_y, map_width, map_height)
    if not end_in_map:
        end_x, end_y = _clip_to_boundary(end_x, end_y, map_width, map_height)
    
    # 调用 Numba 加速的核心算法
    result = _bresenham_collision_check(
        start_x, start_y, end_x, end_y,
        map_data, map_height, map_width
    )
    
    # 转换返回值
    if result == 0:
        return False
    elif result == 1:
        return OCCUPIED
    else:
        return UNKNOWN

@njit(cache=True, parallel=True)
def _batch_collision_check(starts, ends, map_data, origin_x, origin_y, cell_size):
    """
    批量碰撞检测 - 适用于需要检测多条线段的场景
    
    参数:
    starts: 起点数组 (N, 2)
    ends: 终点数组 (N, 2)
    map_data: 地图数据
    origin_x, origin_y: 地图原点
    cell_size: 单元格大小
    
    返回:
    results: 碰撞结果数组 (N,)，0=无碰撞，1=障碍物，127=未知
    """
    n = starts.shape[0]
    results = np.zeros(n, dtype=np.int32)
    map_height, map_width = map_data.shape
    
    for i in range(n):
        # 坐标转换
        x0 = int(round((starts[i, 0] - origin_x) / cell_size))
        y0 = int(round((starts[i, 1] - origin_y) / cell_size))
        x1 = int(round((ends[i, 0] - origin_x) / cell_size))
        y1 = int(round((ends[i, 1] - origin_y) / cell_size))
        
        # 边界检查
        start_in = 0 <= x0 < map_width and 0 <= y0 < map_height
        end_in = 0 <= x1 < map_width and 0 <= y1 < map_height
        
        if not start_in and not end_in:
            results[i] = 127
            continue
        
        # 边界裁剪
        if not start_in:
            x0 = max(0, min(x0, map_width - 1))
            y0 = max(0, min(y0, map_height - 1))
        if not end_in:
            x1 = max(0, min(x1, map_width - 1))
            y1 = max(0, min(y1, map_height - 1))
        
        # Bresenham 算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2
        
        collision = 0
        while 0 <= x < map_width and 0 <= y < map_height:
            k = map_data[y, x]
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = 1
                break
            if k == 127:
                collision = 127
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        results[i] = collision
    
    return results


def batch_check_collision(starts, ends, map_info):
    """
    批量碰撞检测的包装函数
    
    参数:
    starts: 起点数组 (N, 2) 世界坐标
    ends: 终点数组 (N, 2) 世界坐标
    map_info: 地图信息对象
    
    返回:
    results: 碰撞结果数组
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    
    if starts.ndim == 1:
        starts = starts.reshape(1, 2)
        ends = ends.reshape(1, 2)
    
    return _batch_collision_check(
        starts, ends,
        map_info.map,
        map_info.map_origin_x,
        map_info.map_origin_y,
        map_info.cell_size
    )

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
    kernel = np.ones((3, 3), np.uint8)
    processed_mask = cv2.dilate(obstacle_mask, kernel, iterations=3)
    binary_map = np.uint8((processed_mask == 1) * 255)

    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    visible_graphs = []
    
    all_points = []
    point_contour_id = []
    contour_sampled_points = []
    
    original_min_size = 3
    dilated_min_area = (original_min_size + 2 * 3) ** 2
    min_contour_area = dilated_min_area
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    for idx, contour in enumerate(contours):
        epsilon = 3
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx[:, 0, :]
        sample_rate = 1
        sampled_points = points[::sample_rate]
        contour_sampled_points.append(sampled_points.tolist())
        for point in sampled_points:
            all_points.append(tuple(point))
            point_contour_id.append(idx)

    n = len(all_points)
    if n == 0:
        return visible_graphs
    
    # 使用批量碰撞检测优化
    pairs_i, pairs_j = np.triu_indices(n, k=1)

    if len(pairs_i) == 0:
        return visible_graphs
    
    # 构建起点和终点数组
    all_points_array = np.array(all_points, dtype=np.float64)
    starts = all_points_array[pairs_i]
    ends = all_points_array[pairs_j]
    
    # 批量碰撞检测（使用单元格坐标，不需要转换）
    collision_results = _batch_collision_check(
        starts, ends,
        map_info.map,
        0.0, 0.0,  # 单元格坐标，原点为0
        1.0  # cell_size = 1
    )
    
    # 构建可见性矩阵
    visibility_matrix = np.zeros((n, n), dtype=bool)
    for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        if collision_results[idx] == 0:
            visibility_matrix[i, j] = True
            visibility_matrix[j, i] = True

    # 为每个轮廓构建可见图
    for idx in range(len(contours)):
        contour_point_indices = [i for i, cid in enumerate(point_contour_id) if cid == idx]
        edges = []

        for i in contour_point_indices:
            point_edges = []
            for j in range(len(all_points)):
                if i != j and visibility_matrix[i, j]:
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

def get_nodes_id_in_range(updating_map_info, nodes_tree):
    min_x = updating_map_info.map_origin_x + 2
    min_y = updating_map_info.map_origin_y + 2
    max_x = min_x + updating_map_info.cell_size * (updating_map_info.map.shape[1] ) - 2
    max_y = min_y + updating_map_info.cell_size * (updating_map_info.map.shape[0] ) - 2
    bb = (min_x,
          min_y,
          max_x,
          max_y)
    result_ids = list(nodes_tree.intersection(bb))
    return [nid for nid in result_ids]


