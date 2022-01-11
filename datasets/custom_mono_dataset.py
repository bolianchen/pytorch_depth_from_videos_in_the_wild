# Copyright reserved

from __future__ import absolute_import, division, print_function

import os
import sys
import random
import skimage.transform
import numpy as np
import PIL.Image as pil

import torch
from torchvision import transforms

from lib.img_processing import image_resize
from .mono_dataset import MonoDataset

class CustomMonoDataset(MonoDataset):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(CustomMonoDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)

        return image_path

    def get_mask_path(self, folder, frame_index):
        f_str = "{:010d}-fseg{}".format(frame_index, self.img_ext)
        mask_path = os.path.join(self.data_path, folder, f_str)

        return mask_path

    def get_intrinsics_path(self, folder, frame_index): 
       
        # set up the intrinsics path for each single image
        intrinsics_path = os.path.join(
                self.data_path,
                folder,
                "{:010d}_cam.txt".format(frame_index))
        # if no intrinsics path available for each single image
        # try to find the common intrinsics path
        if not os.path.exists(intrinsics_path):
            intrinsics_path = os.path.join(self.data_path, 'cam.txt')
            assert os.path.exists(intrinsics_path)

        return intrinsics_path

    def get_image(self, image, do_flip, crop_offset=-3):
        r"""
        Resize (and crop) an image to specified height and width.
        crop_offset is an integer representing how the image will be cropped:
            -3      the image will not be cropped
            -2      the image will be center-cropped
            -1      the image will be cropped by a random offset
            >0      the image will be cropped by this offset
        """
        # If crop_offset is set to -3, do not crop the image. 
        # Resize the image to (self.height, self.width).
        if crop_offset == -3:            
            image, ratio, delta_u, delta_v = image_resize(image, self.height,
                                                        self.width, 0.0, 0.0) 
        # Otherwise resize the image according to self.width, 
        # and then crop the image to self.height according to crop_offset.
        else:
            raw_w, raw_h = image.size
            resize_w = self.width
            resize_h = int(raw_h * resize_w / raw_w)
            image, ratio, delta_u, delta_v = image_resize(image, resize_h,
                                                          resize_w, 0.0, 0.0)
            top = int(self.crop_bound[0] * resize_h)
            if len(self.crop_bound) == 1:
                bottom = top
            elif len(self.crop_bound) == 2:
                bottom = int(self.crop_bound[1] * resize_h) - self.height
            else:
                raise NotImplementedError

            if crop_offset == -1:
                assert bottom >= top, "Not enough height to crop, please set a larger crop_bound range"
                # add one to include the upper limit for sampling
                crop_offset = np.random.randint(top, bottom + 1)
            elif crop_offset == -2:
                crop_offset = int((top+bottom)/2)

            image = np.array(image)
            image = image[crop_offset: crop_offset + self.height]
            image = pil.fromarray(image)
            delta_v += crop_offset

        # if the principal point is not at center,
        # flipping would affect the camera intrinsics but not accounted here
        if do_flip:
            image = image.transpose(pil.FLIP_LEFT_RIGHT)

        return image, ratio, delta_u, delta_v, crop_offset


    def get_color(self, folder, frame_index, do_flip, crop_offset=-3):
        r"""
        Load an RGB image.
        """
        # a PIL image
        color = self.loader(self.get_image_path(folder, frame_index))
        return self.get_image(color, do_flip, crop_offset)

    def get_mask(self, folder, frame_index, do_flip, crop_offset=-3):
        r"""
        Load a binary mask.
        """
        # a PIL image
        mask = self.loader(self.get_mask_path(folder, frame_index))
        return self.get_image(mask, do_flip, crop_offset)

    def load_intrinsics(self, folder, frame_index):

        intrinsics_path = self.get_intrinsics_path(folder, frame_index)

        f = open(intrinsics_path, 'r') 
        arr = np.array([ [float(e) for e in l.split(',')] for l in f.readlines() ])
        K = np.eye(4)
        K[:3,:3] = arr.reshape(3,3)
            
        return np.float32(K)


    def get_repr_intrinsics(self):

        folder, frame_index = self._parse_line(0)
        crop_offset = -2 if self.do_crop else -3

        _, ratio, delta_u, delta_v, _ = self.get_color(
                folder, frame_index, False, crop_offset=-2)

        K = self.load_intrinsics(folder, frame_index)
        K[0, :] *= ratio
        K[1, :] *= ratio
        K[0,2] -= delta_u
        K[1,2] -= delta_v

        return K
       
    def _parse_line(self, index):
        """Decompose a line in train or val list in folder and index"""
        line = self.filenames[index].split()
        folder, frame_index= line[0], int(line[1])
        assert frame_index != 0
        return folder, frame_index


    def __getitem__(self, index):
        """Returns a single training data item as a dictionary.
        Revised from monodepth2 repo

        """
        # an empty dictionary
        inputs = {}


        # whether to do color enhancement and horizontal flip
        do_color_aug = self.is_train and random.random() > 0.5
        do_color_aug = (not self.not_do_color_aug) and do_color_aug
        do_flip = self.is_train and random.random() > 0.5
        do_flip = (not self.not_do_flip) and do_flip

        folder, frame_index = self._parse_line(index)

        # Initialize cropping method
        if self.do_crop:
            # Random crop for training and center crop for validation
            crop_offset = -1 if self.is_train else -2
        else:
            # no cropping
            crop_offset = -3

        # add the images of the original scale to the inputs
        for i in self.frame_idxs:
            # get_color is to load the specified image
            inputs[("color", i, -1)], ratio, delta_u, delta_v, crop_offset = \
                self.get_color(
                    folder, frame_index + i, do_flip, crop_offset
                )
            # get_mask is to load the specified mask
            if self.seg_mask != 'none':
                mask = self.get_mask(
                    folder, frame_index + i, do_flip, crop_offset
                )[0]
                inputs[("mask", i, -1)] = mask.convert('L')

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            
            K = self.load_intrinsics(folder, frame_index)

            # adjust K for the resizing within the get_color function
            K[0, :] *= ratio
            K[1, :] *= ratio
            K[0,2] -= delta_u
            K[1,2] -= delta_v

            # Modify the intrinsic matrix if the image is flipped
            if do_flip:
                K[0,2] = self.width - K[0,2]
            
            # adjust K for images of different scales
            K[0, :] /= (2 ** scale)
            K[1, :] /= (2 ** scale)

            inv_K = np.linalg.pinv(K)

            # add intrinsics to inputs
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # return a transform
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # delete the images of original scale
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            if self.seg_mask != 'none':
                del inputs[("mask", i, -1)]

        return inputs

