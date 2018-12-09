'''
This file keeps some constants used through out the program.
'''
import os

# dataset

# __file__: point to the filename of the current module
# os.path.abspath(): turn that into an absolute path
# os.path.dirname(): remove the last segment of the path
PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = 'MACD_48-12'

NUM_EPOCHS = 200
BATCH_SIZE = 32
HISTORY_LENGTH = 180
HORIZON = 24
TEST_FRAC = 0.2
MEMORY_SIZE = 4000
NUM_CHANNELS = 3

# actions
NUM_ACTIONS = 2
LONG = 1 # buy in
SHORT = -1 # sell out
ACTIONS = [LONG, SHORT]

# Q learning
DISCOUNT_FACTOR = 0.95

LR_DECAY = 0.995
LR_UPDATE_EPOCH = 50

EPSILON_DECAY = 0.9 
EPSILON_UPDATE_STEP = 50




