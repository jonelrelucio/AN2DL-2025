
# --- Stochasticity -- 
SEED = 100

# --- Model Parameters ---
MODEL_TYPE = 'ResNet50'
PRETRAINED = True
NUM_CLASSES = 10

# --- Training Hyperparameters ---
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 300
OPTIMIZER = 'AdamW'

# --- Data Settings ---
TRAIN_DATA_PATH = 'train_data'
TEST_DATA_PATH = 'test_data'
TRAIN_CACHE_PATH = 'cache/train_images_cache.npy'
TEST_CACHE_PATH = 'cache/test_images_cache.npy'
IMAGE_SIZE = 224
