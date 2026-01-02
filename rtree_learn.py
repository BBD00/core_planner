# 导入核心模块
from rtree import index
import time
import numpy as np
import sys
sys.setrecursionlimit(50000)
# 1. 创建内存索引（最简单，常用）
idx = index.Index()

# 示例1：插入3个矩形（带id和边界）
idx.insert(0, (0, 0, 0, 0))  # id=0，矩形左下角(0,0)，右上角(2,2)
idx.insert(1, (1, 1, 1, 1))  # id=1，矩形左下角(1,1)，右上角(3,3)
idx.insert(2, (5, 5, 7, 7))  # id=2，矩形左下角(5,5)，右上角(7,7)

# 示例2：插入点（点的边界是x/y的最小/最大值相等）
idx.insert(3, (4, 4, 4, 4))  # id=3，点(4,4)

# 示例3：插入带附加数据的空间对象
idx.insert(4, (0, 5, 2, 7), obj={"name": "公园", "area": 4})  # 附加属性字典

# 范围查询
query_bounds = (0, 0, 0, 0)

# 方式1：仅返回id（默认，简洁）
result_ids = list(idx.intersection(query_bounds))
print("范围查询返回的id列表：", result_ids)  # 输出：[0,1,3]（这3个对象在查询范围内）



# 方式2：返回完整对象（含id、bounds、obj）
result_objs = list(idx.intersection(query_bounds, objects=True))
for item in result_objs:
    print(f"id: {item.id}")
    print(f"边界: {item.bbox}")
    print(f"附加数据: {item.object}")
    print("-" * 20)

# 最近邻查询
# 示例：查询点 (3,3) 的最近2个邻域对象
target_bounds = (3, 3, 3, 3)  # 目标点(3,3)
nearest_ids = list(idx.nearest(target_bounds, num_results=2))
print("近邻查询返回的id列表：", nearest_ids)  # 输出：[1,0]（id=1的矩形离(3,3)最近，其次是id=0）

# 删除
# 示例：删除id=2的对象（插入时bounds是(5,5,7,7)） 必须bounds也一样
idx.delete(2, (5, 5, 7, 7))

# 验证：删除后查询范围(5,5,7,7)，无结果
delete_check = list(idx.intersection((5, 5, 7, 7)))
print("删除id=2后，查询对应范围的结果：", delete_check)  # 输出：[]

# 1. 范围查询：用np数组作为查询窗口
query_bounds_np = np.array([0, 0, 4, 4])  # 查询范围(0,0,4,4)
# 直接传入np数组，返回结果与元组格式完全一致
t1 = time.time()
result_ids = list(idx.intersection(query_bounds_np))
print("time:", time.time() - t1)
print("NumPy数组查询返回的id：", result_ids)  # 输出：[0,1]（对应插入的矩形和点）


class node:
    def __init__(self):
        self.id = 0
        self.a = [123,123,5646,4,51,3]
idx.insert(5, (10,10,10,10), obj=node())
idx.insert(6, (11,11,11,11))
idx.insert(7, (12,11,12,12), obj=node())
idx.insert(8, (12,11,12,12), obj=node())
t1 = time.time()
query_bounds_np = np.array([10, 10, 10, 10]) 
result_ids = list(idx.intersection(query_bounds_np))
print("time:", time.time() - t1)

t1 = time.time()
query_bounds_np = np.array([11,11,11,11]) 
result_ids = list(idx.intersection(query_bounds_np))
print("time:", time.time() - t1)

t1 = time.time()
query_bounds_np = np.array([10,10,10,10]) 
result_ids = idx.intersection(query_bounds_np, objects="raw")
print("time:", time.time() - t1)
print("NumPy数组查询返回的id：", result_ids)
for item in result_ids:
    # item 是 rtree 的 Item 对象
    # item.object 才是你插入的 node 对象
    print(f"rtree id: {item.id}")
    print(f"node.a: {item.a}")


query_bounds_np = np.array([10,10,10,10]) 

print(type(query_bounds_np[0]))

node = {1:(10,15),2:(11,16),3:(12,17)}
edges = list(node.values())
print(edges)
print(type(np.array(edges)[0]))


import quads

# 写一个对比函数，比较两个结构插入查找的时间
def compare_structures():
    # 创建 R树 索引
    idx = index.Index()
    # 创建 QuadTree（中心点在500,500，宽高各1002，覆盖0-1000范围）
    quad_tree = quads.QuadTree((500, 500), 1020, 1020)

    num_points = 10000

    # 生成随机点，避免重复坐标导致 QuadTree 问题
    np.random.seed(42)
    points_x = np.random.uniform(1, 989, num_points)
    points_y = np.random.uniform(1, 989, num_points)

    # ============ 插入测试 ============
    # 测量 R树 插入时间
    start_time = time.time()
    for i in range(num_points):
        rect = (points_x[i], points_y[i], points_x[i] + 1, points_y[i] + 1)
        idx.insert(i, rect)
    rtree_insert_time = time.time() - start_time

    # 测量 QuadTree 插入时间
    start_time = time.time()
    for i in range(num_points):
        quad_tree.insert((points_x[i], points_y[i]), data=i)
    quadtree_insert_time = time.time() - start_time

    # ============ 查询测试 ============
    query_bounds = (400, 400, 600, 600)
    num_queries = 1000

    # 测量 R树 查询时间
    start_time = time.time()
    for _ in range(num_queries):
        rtree_results = list(idx.intersection(query_bounds))
    rtree_query_time = time.time() - start_time

    # 测量 QuadTree 查询时间
    bb = quads.BoundingBox(query_bounds[0], query_bounds[1], query_bounds[2], query_bounds[3])
    start_time = time.time()
    for _ in range(num_queries):
        quadtree_results = quad_tree.within_bb(bb)
    quadtree_query_time = time.time() - start_time

    # ============ 删除测试 ============
    num_deletes = 1000

    # 测量 R树 删除时间
    start_time = time.time()
    for i in range(num_deletes):
        rect = (points_x[i], points_y[i], points_x[i] + 1, points_y[i] + 1)
        idx.delete(i, rect)
    rtree_delete_time = time.time() - start_time

    # 测量 QuadTree 删除时间
    start_time = time.time()
    for i in range(num_deletes):
        point = quads.Point(points_x[i], points_y[i])
        quad_tree.remove(point)
    quadtree_delete_time = time.time() - start_time

    # ============ 修改测试（删除后重新插入） ============
    num_updates = 1000

    # 测量 R树 修改时间（删除旧的 + 插入新的）
    start_time = time.time()
    for i in range(num_deletes, num_deletes + num_updates):
        # 删除旧位置
        old_rect = (points_x[i], points_y[i], points_x[i] + 1, points_y[i] + 1)
        idx.delete(i, old_rect)
        # 插入新位置（偏移10）
        new_rect = (points_x[i] + 10, points_y[i] + 10, points_x[i] + 11, points_y[i] + 11)
        idx.insert(i, new_rect)
    rtree_update_time = time.time() - start_time

    # 测量 QuadTree 修改时间（删除旧的 + 插入新的）
    start_time = time.time()
    for i in range(num_deletes, num_deletes + num_updates):
        # 删除旧位置
        old_point = quads.Point(points_x[i], points_y[i])
        quad_tree.remove(old_point)
        # 插入新位置（偏移10）
        quad_tree.insert((points_x[i] + 10, points_y[i] + 10), data=i)
    quadtree_update_time = time.time() - start_time

    # ============ 输出结果 ============
    print("=" * 50)
    print(f"数据量: {num_points} 个点")
    print("=" * 50)
    print(f"【插入】")
    print(f"  R树: {rtree_insert_time:.6f}秒")
    print(f"  QuadTree: {quadtree_insert_time:.6f}秒")
    print(f"  R树/QuadTree: {rtree_insert_time/quadtree_insert_time:.2f}x")
    print("-" * 50)
    print(f"【查询】({num_queries}次)")
    print(f"  R树: {rtree_query_time:.6f}秒")
    print(f"  QuadTree: {quadtree_query_time:.6f}秒")
    print(f"  QuadTree/R树: {quadtree_query_time/rtree_query_time:.2f}x")
    print(f"  R树结果数: {len(rtree_results)}, QuadTree结果数: {len(quadtree_results)}")
    print("-" * 50)
    print(f"【删除】({num_deletes}次)")
    print(f"  R树: {rtree_delete_time:.6f}秒")
    print(f"  QuadTree: {quadtree_delete_time:.6f}秒")
    print("-" * 50)
    print(f"【修改】({num_updates}次)")
    print(f"  R树: {rtree_update_time:.6f}秒")
    print(f"  QuadTree: {quadtree_update_time:.6f}秒")
    print("=" * 50)

compare_structures()