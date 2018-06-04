""" This is a wrapper for Waleed Abdulla's M-RCNN model from Matterport."""

import torch
from .config import Config
import modules.ai.mrcnn_model as modellib


class InferenceConfig(Config):

    NAME = "cell"
    NUM_CLASSES = 1 + 4
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100

    RPN_TRAIN_ANCHORS_PER_IMAGE = 512  # 300
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    TRAIN_ROIS_PER_IMAGE = 400  # 256
    RPN_NMS_THRESHOLD = 0.70
    MAX_GT_INSTANCES = 300
    DETECTION_MAX_INSTANCES = 300
    RPN_ANCHOR_SCALES = (4, 16, 64, 128, 256)
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (64, 64)
    LEARNING_RATE = 0.001


class MaskRCNN:
    def __init__(self):
        self._config = InferenceConfig()
        self.model = None

    def load(self, path):
        self.model = modellib.MaskRCNN(model_dir="mrcnn_logs", config=self._config)
        if self._config.GPU_COUNT:
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(path))

    def process(self, image):
        result = self.model.detect([image])[0]
        return result
