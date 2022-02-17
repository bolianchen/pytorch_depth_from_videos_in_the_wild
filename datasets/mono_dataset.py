# Copyright reserved

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

import itertools

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders"""
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg', 
                 not_do_color_aug=False,
                 not_do_flip=False,
                 do_crop=False,
                 crop_bound=[0.0, 1.0],
                 seg_mask='none',
                 boxify=False,
                 MIN_OBJECT_AREA=20,
                 prob_to_mask_objects=0.0):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext

        self.do_crop = do_crop
        self.crop_bound = crop_bound
        self.not_do_color_aug = not_do_color_aug
        self.not_do_flip = not_do_flip

        self.seg_mask = seg_mask
        self.boxify = boxify
        self.MIN_OBJECT_AREA = MIN_OBJECT_AREA
        self.prob_to_mask_objects = prob_to_mask_objects

        # PIL image loader
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # to test if an error occur
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize and augment color images and masks to the required scales

        We create the color_aug object in advance and apply the same 
        augmentation to all images in this item. This ensures that all images 
        input to the pose network receive the same augmentation.
        """
        # list(inputs) is a list composed of the keys of inputs
        for k in list(inputs):
            if "color" in k or "mask" in k:
                n, im, _ = k
                # save images resized to different scales to inputs
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                # convert images to tensors
                inputs[(n, im, i)] = self.to_tensor(f)
                # augment images
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "mask" in k:
                n, im, i = k
                # convert masks to tensors
                if self.seg_mask != 'none':
                    inputs[(n, im, i)] = torch.from_numpy(np.array(f))
                else:
                    inputs[(n, im, i)] = self.to_tensor(f)

        if self.seg_mask != 'none':
            self.process_masks(inputs, self.seg_mask)

            if random.random() < self.prob_to_mask_objects:
                self.mask_objects(inputs)
                inputs["objects_being_masked"] = True
            else:
                inputs["objects_being_masked"] = False

    def process_masks(self, inputs, mask_mode):
        """Convert segmentation masks to binary masks

        Remove masks whose areas are smaller than MIN_OBJECT_AREA
        boxify a local region with the same id if required
        """
        MIN_OBJECT_AREA = self.MIN_OBJECT_AREA

        for scale in range(self.num_scales):

            if mask_mode == 'color':
                object_ids = torch.unique(torch.cat(
                    [inputs['mask', fid, scale] for fid in self.frame_idxs]),
                    sorted=True)
            else:
                object_ids = torch.Tensor([0, 255])

            for fid in self.frame_idxs:
                current_mask = inputs['mask', fid, scale]

                def process_obj_mask(obj_id, mask_mode=mask_mode):
                    """Create a mask for obj_id, skipping the background mask."""
                    if mask_mode == 'color':
                        mask = torch.logical_and(
                                torch.eq(current_mask, obj_id),
                                torch.ne(current_mask, 0)
                                )
                    else:
                        mask = torch.ne(current_mask, 0)

                    # TODO early return when obj_id == 0
                    # Leave out very small masks, that are most often errors.
                    obj_size = torch.sum(mask)
                    if MIN_OBJECT_AREA != 0:
                        mask = torch.logical_and(mask, obj_size > MIN_OBJECT_AREA)
                    if not self.boxify:
                      return mask
                    # Complete the mask to its bounding box.
                    binary_obj_masks_y = torch.any(mask, axis=1, keepdim=True)
                    binary_obj_masks_x = torch.any(mask, axis=0, keepdim=True)
                    return torch.logical_and(binary_obj_masks_y, binary_obj_masks_x)

                object_mask = torch.stack(
                        list(map(process_obj_mask, object_ids))
                        )
                object_mask = torch.any(object_mask, axis=0, keepdim=True)
                inputs['mask', fid, scale] = object_mask.to(torch.float32)

    def mask_objects(self, inputs):
        """Mask objects overlapping with mobile masks"""
        for scale in range(self.num_scales):
            for fid in self.frame_idxs:
                inputs['color_aug', fid, scale] *= (1 - inputs['mask', fid, scale])
                inputs['color', fid, scale] *= (1 - inputs['mask', fid, scale])

    def __len__(self):
        return len(self.filenames)

    def get_color(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def get_mask(self, folder, frame_index, do_flip):
        raise NotImplementedError

    def check_depth(self):
        """Check if there exists ground-truth depth for the dataset"""
        raise NotImplementedError

    def get_depth(self, folder, frame_index, do_flip):
        """Load groud-truth depth if available"""
        raise NotImplementedError
