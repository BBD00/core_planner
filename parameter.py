# saving path
FOLDER_NAME = 'core'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
map_path = f'/data/kjt_data/code/ARiADNE/maps'

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.4  # meter, your map resolution
NODE_RESOLUTION = 4.0  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 30  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# 车辆移动距距离
MOVE_DISTANCE = 5   # meter
END_MIN_DISTANCE = 2 # meter判断结束的距离

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements 80

# 地图可视点更新容忍值
TOLERANCE = 0.1

# 车辆停滞惩罚阈值
STAGNATION_THRESHOLD = 3
STAY_DIS_THRESHOLD = 3
STAY_WINDOW_SIZE = 20

# 边界聚类最小值
MIN_CLUSTER_NUM = 7
MAX_CLUSTER_NUM = 25

# training parameters
MAX_EPISODE_STEP = 250  #128
REPLAY_SIZE = 20000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 256
LR = 2e-5
GAMMA = 1
NUM_META_AGENT = 32  # how many threads does your CPU have 16

# network parameters
NODE_INPUT_DIM = 5
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 360  # the number of neighboring nodes, fixed  修改现在允许与所有点相连
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs

