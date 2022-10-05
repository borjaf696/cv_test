### Credits to:
# https://github.com/matterport/Mask_RCNN/blob/v2.1/samples/balloon/balloon.py
###
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from os import listdir
from os.path import isfile, join
# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/curves"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from config import Config
import utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_curves.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Default configuration

class CurveConfig(Config):
    # Give the configuration a recognizable name
    NAME = "curve"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class CurveDataset(utils.Dataset):

    def load_curve(self, dataset_dir, subset):
        """Load a subset of the Curve dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("curve", 1, "curve")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        
        images_path = 'images'
        mask_path = 'masks'
        
        dataset_dir = os.path.join(dataset_dir, subset)
        images_dir = os.path.join(dataset_dir, images_path)
        masks_dir = os.path.join(dataset_dir, mask_path)

        onlyfiles = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]

        for i, filename in enumerate(onlyfiles):
            if i > 1000:
                break
            image_path = os.path.join(images_dir, filename)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "curve",
                image_id= filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask_filename=os.path.join(masks_dir, filename),
                type = str(str(filename.split('_')[-1]).split('.')[0]))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        mask_filename = info["mask_filename"]
        mask = skimage.io.imread(mask_filename)
        
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "curve":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)