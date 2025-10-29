# constants.py

DATA_ROOT = "./AD_NC"


IMAGE_SIZE = 224
CLASSES = ['AD', 'NC']
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = [0.1115, 0.1115, 0.1115]
STD  = [0.2186, 0.2186, 0.2186]

NUM_WORKER = 2

# Device settings
DEVICE_MPS = 'mps'
DEVICE_CUDA = 'cuda'
DEVICE_CPU  = 'cpu'

# Data & training
BATCH_SIZE = 512
EPOCHS     = 400
LR         = 1e-3
SWA_LR     = 1e-4
WD         = 0.05


NUM_CLASSES   = 2
DROP_PATH_RATE = 0.1  # for example, maybe 0.1 for tiny model
