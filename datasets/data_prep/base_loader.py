# All Rights Reserved.

"""Classes to load KITTI and Cityscapes data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import sys
import shutil
import re

from absl import logging
import numpy as np
import imageio
from PIL import Image
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

from .preprocess import DynamicObjectDetector


# Color ids for masks
colors = np.arange(1, 256, dtype=np.uint8).reshape(-1, 1, 1)

class BaseLoader(object):
    r"""
    Base dataloader. We expect that all loaders inherit this class.
    """

    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 data_format='mono2',
                 mask='none',
                 batch_size=32,
                 threshold=0.5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.data_format=data_format
        self.gen_mask = mask != 'none'
        self.mask = mask
        self.batch_size = batch_size
        self.threshold = threshold

        if self.gen_mask:
            self._initialize_mrcnn_model()

    def _initialize_mrcnn_model(self):
        model = maskrcnn_resnet50_fpn(pretrained=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        self.model = model.to(device)

    def run_mrcnn_model(self, images):
        r"""
        Run Mask-RCNN model on images.
        """
        images = list(images)
        with torch.no_grad():
            results = self.model(images)
        return results

    def generate_mask(self, mrcnn_result, dynamic_map=None):
        r"""
        Generate mask based on the output of Mask-RCNN model.
        """
        # Only consider objects with predicted scores higher than the threshold
        score = mrcnn_result['scores'].detach().cpu().numpy()
        valid = (score > self.threshold).sum()
        masks = (mrcnn_result['masks'] > self.threshold).squeeze(1).detach().cpu().numpy()
        labels = mrcnn_result['labels'].detach().cpu().numpy() 
        if valid > 0:
            masks = masks[:valid] # (N, H, W)
            labels = labels[:valid]
        else:
            masks = np.zeros_like(masks[:1])
            labels = np.zeros_like(labels[:1])
        masks = masks.astype(np.uint8)

        # Throw away the masks that are not pedestrians or vehicles
        masks[labels == 0] *= 0 # __background__
        masks[labels == 5] *= 0 # airplane
        masks[labels == 7] *= 0 # train
        masks[labels > 8] *= 0

        if self.mask == 'instance':
            if masks.shape[0] != 0:
                masks_to_keep = []
                for label in labels:
                    if label in [0, 5, 7] or label > 8:
                        masks_to_keep.append(False)
                    else:
                        masks_to_keep.append(True)

                masks =  masks[masks_to_keep]

            if masks.shape[0] == 0:
                return np.zeros(
                        (1, ) + masks.shape[1:],
                        dtype=np.uint8)
            return masks

        mask_img = np.ones_like(masks, dtype=np.uint8) 
        if self.mask == 'mono':
            mask_img = masks * mask_img
            mask_img = np.sum(mask_img, axis=0)
            mask_img = (mask_img > 0).astype(np.uint8) * 255
            return mask_img
        elif self.mask == 'color':
            for i in range(masks.shape[0]-1):
                masks[i+1:] *= 1 - masks[i]
            # ignore this step when masks is empty 
            if masks.shape[0] != 0:
                # for non-background objects
                # sample colors evenly between 1 and 255
                mask_img = masks * mask_img * colors[
                        np.linspace(0, 254, num= masks.shape[0], dtype=np.uint8)
                        ]
            mask_img = np.sum(mask_img, axis=0)
            return mask_img

    def is_bad_sample(self, target_index):
        r"""
        Check whether this frame fails to satisfy certain criteria.
        This method can be overridden to customize the criteria.
        """
        return False
    
    def is_valid_sample(self, target_index):
        r"""
        Check whether we can find a valid sequence around this frame.
        Every loader should implement its own criterion.
        """
        raise NotImplementedError

    def get_example_with_index(self, target_index):

        add_to_file = False 
        example = None

        #if self.is_bad_sample(target_index):
        #    return add_to_file, example

        # A frame without required adjacent frames
        if not self.is_valid_sample(target_index):
            # For mono2, the current frame is returned but it will not be
            # added to the train or val list
            if self.data_format == 'mono2':
                example = self.load_example(target_index)
                return add_to_file, example
            elif self.data_format == 'struct2depth':
                return add_to_file, example

        add_to_file = True
        example = self.load_example(target_index)

        return add_to_file, example

    def load_image_sequence(self, target_index):
        r"""
        Return a sequence with requested target frame.
        Every loader shoud implement its own method.
        """
        raise NotImplementedError

    def load_example(self, target_index):
        r"""
        Return a sequence with requested target frame.
        Every loader should implement its own method.
        """
        raise NotImplementedError

    def load_image_raw(self, infos):
        r"""
        Load an raw image given its id.
        Every loader should implement its own method.
        """
        raise NotImplementedError

    def load_intrinsics(self, infos):
        r"""
        Load the intrinsic matrix given its id.
        Every loader should implement its own method.
        """
        raise NotImplementedError

    def scale_intrinsics(self, mat, sx, sy, crop_top=0, crop_left=0):
        """Adjust intrinsics after resizing and then cropping
        """
        out = np.copy(mat)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        if crop_top:
            out[1, 2] -= crop_top
        if crop_left:
            out[0, 2] -= crop_left
        return out


def get_resource_path(relative_path):
    return relative_path

def get_seq_start_end(target_index, seq_length, sample_every=1):
    r"""
    Return absolute seq start and end indices for a given target frame.
    """
    if seq_length == 1:
        start_index = end_index = target_index
    else:
        half_offset = int((seq_length - 1) / 2) * sample_every
        end_index = target_index + half_offset
        start_index = end_index - (seq_length - 1) * sample_every
    return start_index, end_index

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]
