from worker import Worker
from parameter import *
from model import PolicyNet
import matplotlib
# logging_config.py
from loguru import logger
import numpy as np
import quads


tree = quads.QuadTree((0, 0), 20, 20)
tree.insert((3, 5))
tree.insert((-2, -2))
tree.insert((0, 1))
tree.insert((-3, 3))
tree.insert((-2, 4))


a = np.array([1,1])
b = np.array([0,0])

print(np.linalg.norm(a-b))