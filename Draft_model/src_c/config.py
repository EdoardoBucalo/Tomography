# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = 1
LEARNING_RATE = 0.0008654690786896991
BATCH_SIZE = 32
NUM_EPOCHS = 100
OPTIMIZER='adam'
fc_layer_size=64
n_layer=3
activation_function='ReLU'


# Dataset
DATA_DIR = '../data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 0

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'
