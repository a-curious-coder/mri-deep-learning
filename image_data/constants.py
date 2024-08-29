""" Settings file """
IMAGE_SIZE = (75, 75)
SLICE_MODE = "average_center"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
RETRAIN = True
CV = 5
CLASSIFICATION = "multiclass" # 'binary' or 'multiclass'
AUGMENTATION = True
MRI_IMAGE_DIR = "../data/mri_images"
TRAIN_DIR = f'../data/dataset/{CLASSIFICATION}/{SLICE_MODE}/train'
TEST_DIR = f'../data/dataset/{CLASSIFICATION}/{SLICE_MODE}/test'
BALANCE_DATA = True # Balance images through custom augmentation class
