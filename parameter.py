# saving path
FOLDER_NAME = 'core_planner'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
map_path = f'./maps'


SUMMARY_WINDOW = 32
LOAD_MODEL = False
SAVE_IMG_GAP = 500

# map and planning resolution
CELL_SIZE = 0.4
NODE_RESOLUTION = 4.0
FRONTIER_CELL_SIZE = 2 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 30
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 2


MOVE_DISTANCE = 5
END_MIN_DISTANCE = 2


UPDATING_MAP_SIZE = 2 * SENSOR_RANGE + 4 * NODE_RESOLUTION


TOLERANCE = 0.1
UPDATE_THRESHOLD = 5


STAGNATION_THRESHOLD = 3
STAY_DIS_THRESHOLD = 3
STAY_WINDOW_SIZE = 20
NOISE_THRESHOLD = 1


MIN_CLUSTER_NUM = 7
MAX_CLUSTER_NUM = 25

# training parameters
MAX_EPISODE_STEP = 250
REPLAY_SIZE = 20000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 256
LR = 2e-5
GAMMA = 0.99
NUM_META_AGENT = 16

# network parameters
NODE_INPUT_DIM = 5
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 512
NODE_PADDING_SIZE = 512  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False
USE_GPU_GLOBAL = True
NUM_GPU = 0

IS_DEBUG = True
IS_TRAIN = True
