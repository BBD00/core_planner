import numpy as np
import imageio
import os
from skimage.morphology import label
from numba import jit, njit
from parameter import *
import matplotlib.pyplot as plt
import cv2
from itertools import combinations
from collections import deque


_FREE = FREE
_OCCUPIED = OCCUPIED
_UNKNOWN = UNKNOWN


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
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


@njit(cache=True)
def _find_frontier_indices(map_data, unknown_neighbor):

    y_len, x_len = map_data.shape
    indices = []
    for y in range(y_len):
        for x in range(x_len):
            if map_data[y, x] == _FREE:
                neighbor_count = unknown_neighbor[y, x]
                if 1 < neighbor_count < 8:
                    indices.append(x * y_len + y)
    return np.array(indices, dtype=np.int64)

def get_frontier_in_map(map_info):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]

    unknown = (map_info.map == UNKNOWN).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    unknown_neighbor = cv2.filter2D(unknown, -1, kernel)

    frontier_indices = _find_frontier_indices(map_info.map, unknown_neighbor)

    if frontier_indices.size == 0:
        return set()

    frontier_x = frontier_indices // y_len
    frontier_y = frontier_indices % y_len
    frontier_cell = np.column_stack((frontier_x, frontier_y))

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info)
    if FRONTIER_CELL_SIZE != CELL_SIZE:
        return frontier_down_sample(frontier_coords)
    return set(map(tuple, frontier_coords))


def get_stay_count(current_position, past_trajectory_x, past_trajectory_y, window_size=STAY_WINDOW_SIZE, threshold=STAY_DIS_THRESHOLD):
    if len(past_trajectory_x) < window_size:
        window_size = len(past_trajectory_x)

    recent_x = past_trajectory_x
    recent_y = past_trajectory_y

    distances = np.hypot(np.array(recent_x) - current_position[0], np.array(recent_y) - current_position[1])
    close_count = np.sum(distances < threshold)
    return close_count


@njit(cache=True)
def _downsample_core(data, voxel_indices):
    n = data.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    max_y = np.max(voxel_indices[:, 1]) + 1
    keys = voxel_indices[:, 0] * max_y + voxel_indices[:, 1]

    sort_indices = np.argsort(keys)
    sorted_keys = keys[sort_indices]

    unique_count = 1
    for i in range(1, n):
        if sorted_keys[i] != sorted_keys[i - 1]:
            unique_count += 1

    result = np.empty((unique_count, 2), dtype=np.float64)

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
    data = np.asarray(data, dtype=np.float64).reshape(-1, 2)
    if data.shape[0] == 0:
        return set()
    voxel_indices = (data / voxel_size).astype(np.int32)
    result = _downsample_core(data, voxel_indices)
    return set(map(tuple, result))

def cluster_frontiers(frontier_set, distance_threshold=2.0, min_points=MIN_CLUSTER_NUM, max_cluster_size=MAX_CLUSTER_NUM):
    if not frontier_set:
        return set()

    points = list(frontier_set)
    grid_size = distance_threshold

    grid_dict = {}
    for point in points:

        grid_x = int(point[0] // grid_size)
        grid_y = int(point[1] // grid_size)
        grid_key = (grid_x, grid_y)

        if grid_key not in grid_dict:
            grid_dict[grid_key] = []
        grid_dict[grid_key].append(point)

    clusters = []
    processed_grids = set()

    for grid_key in grid_dict:
        if grid_key in processed_grids:
            continue

        cluster_points = []
        queue = deque([grid_key])
        processed_grids.add(grid_key)

        while queue:
            current_grid = queue.popleft()
            cluster_points.extend(grid_dict[current_grid])
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

    final_clusters = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            sub_clusters = split_large_cluster_simple(cluster, max_cluster_size)
            final_clusters.extend(sub_clusters)

    centroids = set()
    for cluster in final_clusters:
        if len(cluster) >= min_points:
            arr = np.array(cluster)
            centroids.add(tuple(arr.mean(axis=0)))

    return centroids

def split_large_cluster_simple(cluster_points, max_size):
    if len(cluster_points) <= max_size:
        return [cluster_points]

    points_array = np.array(cluster_points)
    min_coords = points_array.min(axis=0)
    max_coords = points_array.max(axis=0)

    num_splits = max(2, int(np.ceil(len(cluster_points) / max_size)))
    grid_size = int(np.ceil(np.sqrt(num_splits)))

    x_step = (max_coords[0] - min_coords[0]) / grid_size if max_coords[0] > min_coords[0] else 1.0
    y_step = (max_coords[1] - min_coords[1]) / grid_size if max_coords[1] > min_coords[1] else 1.0

    if x_step < 0.1 or y_step < 0.1:
        chunk_size = max_size
        return [cluster_points[i:i+chunk_size] for i in range(0, len(cluster_points), chunk_size)]
    sub_clusters_dict = {}
    for point in cluster_points:
        grid_x = min(int((point[0] - min_coords[0]) / x_step), grid_size - 1)
        grid_y = min(int((point[1] - min_coords[1]) / y_step), grid_size - 1)
        grid_key = (grid_x, grid_y)
        if grid_key not in sub_clusters_dict:
            sub_clusters_dict[grid_key] = []
        sub_clusters_dict[grid_key].append(point)

    final_sub_clusters = []
    for sub_cluster in sub_clusters_dict.values():
        if len(sub_cluster) <= max_size:
            final_sub_clusters.append(sub_cluster)
        else:
            chunk_size = max_size
            chunks = [sub_cluster[i:i+chunk_size] for i in range(0, len(sub_cluster), chunk_size)]
            final_sub_clusters.extend(chunks)

    return final_sub_clusters

def is_frontier(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
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
        if k == _OCCUPIED:
            return _OCCUPIED
        if k == _UNKNOWN:
            return _UNKNOWN
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return 0


@njit(cache=True)
def _clip_to_boundary(x, y, width, height):
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
    cell_x = int(round((coord_x - origin_x) / cell_size))
    cell_y = int(round((coord_y - origin_y) / cell_size))
    return cell_x, cell_y


def check_collision(start, end, map_info, do_trans=True):
    map_data = map_info.map
    map_height, map_width = map_data.shape

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

    start_in_map = 0 <= start_x < map_width and 0 <= start_y < map_height
    end_in_map = 0 <= end_x < map_width and 0 <= end_y < map_height

    if not start_in_map and not end_in_map:
        return _UNKNOWN

    if not start_in_map:
        start_x, start_y = _clip_to_boundary(start_x, start_y, map_width, map_height)
    if not end_in_map:
        end_x, end_y = _clip_to_boundary(end_x, end_y, map_width, map_height)

    result = _bresenham_collision_check(
        start_x, start_y, end_x, end_y,
        map_data, map_height, map_width
    )

    if result == 0:
        return False
    elif result == _OCCUPIED:
        return _OCCUPIED
    else:
        return _UNKNOWN

@njit(cache=True, parallel=True)
def _batch_collision_check(starts, ends, map_data, origin_x, origin_y, cell_size):
    n = starts.shape[0]
    results = np.zeros(n, dtype=np.int32)
    map_height, map_width = map_data.shape

    for i in range(n):
        x0 = int(round((starts[i, 0] - origin_x) / cell_size))
        y0 = int(round((starts[i, 1] - origin_y) / cell_size))
        x1 = int(round((ends[i, 0] - origin_x) / cell_size))
        y1 = int(round((ends[i, 1] - origin_y) / cell_size))

        start_in = 0 <= x0 < map_width and 0 <= y0 < map_height
        end_in = 0 <= x1 < map_width and 0 <= y1 < map_height

        if not start_in and not end_in:
            results[i] = _UNKNOWN
            continue

        if not start_in:
            x0 = max(0, min(x0, map_width - 1))
            y0 = max(0, min(y0, map_height - 1))
        if not end_in:
            x1 = max(0, min(x1, map_width - 1))
            y1 = max(0, min(y1, map_height - 1))

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
            if k == _OCCUPIED:
                collision = _OCCUPIED
                break
            if k == _UNKNOWN:
                collision = _UNKNOWN
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
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print(f'{n} gif complete success:{done}\n')

    for filename in frame_files[:-1]:
        os.remove(filename)


class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y

def extract_visible_graph_from_map(map_info):
    obstacle_mask = np.where(map_info.map == _OCCUPIED, 1, 0).astype(np.uint8)
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

    pairs_i, pairs_j = np.triu_indices(n, k=1)

    if len(pairs_i) == 0:
        return visible_graphs

    all_points_array = np.array(all_points, dtype=np.float64)
    starts = all_points_array[pairs_i]
    ends = all_points_array[pairs_j]

    collision_results = _batch_collision_check(
        starts, ends,
        map_info.map,
        0.0, 0.0,
        1.0
    )

    visibility_matrix = np.zeros((n, n), dtype=bool)
    for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        if collision_results[idx] == 0:
            visibility_matrix[i, j] = True
            visibility_matrix[j, i] = True

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
