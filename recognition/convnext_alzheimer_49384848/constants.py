# constants.py
IMAGE_SIZE = 224
CLASSES = ['AD', 'NC']
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = [0.1115, 0.1115, 0.1115]
STD  = [0.2186, 0.2186, 0.2186]

NUM_WORKER = 8

# Device settings
DEVICE_MPS = 'mps'
DEVICE_CUDA = 'cuda'
DEVICE_CPU  = 'cpu'

# Data & training
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 2e-4
WD         = 3e-4

DATA_ROOT = './recognition/convnext_alzheimer_49384848/AD_NC'

NUM_CLASSES   = 2
DROP_PATH_RATE = 0.25  # for example, maybe 0.1 for tiny model
