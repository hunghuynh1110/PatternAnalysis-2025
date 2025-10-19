# constants.py
IMAGE_SIZE = 224
CLASSES = ['AD', 'NC']
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Device settings
DEVICE_MPS = 'mps'
DEVICE_CUDA = 'cuda'
DEVICE_CPU  = 'cpu'

# Data & training
BATCH_SIZE = 16
EPOCHS     = 5
LR         = 1e-4

DATA_ROOT = './recognition/convnext_alzheimer_49384848/AD_NC'

NUM_CLASSES   = 2
