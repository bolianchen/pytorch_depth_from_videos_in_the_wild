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

from lib.img_processing import image_resize

from .base_loader import (
    BaseLoader, 
    get_resource_path, 
    get_seq_start_end,
    natural_keys
)

class KittiRaw(BaseLoader):
    r"""
    Base dataloader. We expect that all loaders inherit this class.
    """

    def __init__(self,
                 dataset_dir,
                 split,
                 load_pose=False,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 data_format='mono2',
                 mask='none',
                 batch_size=32,
                 threshold=0.5,
                 detect_dynamic_objects=False,
                 shift_h=0.0,
                 reprojection_info=None):
        super().__init__(dataset_dir, img_height, img_width,
                         seq_length, data_format, mask, batch_size, 
                         threshold, detect_dynamic_objects,
                         reprojection_info=reprojection_info)
        static_frames_file = 'data_prep/kitti/static_frames.txt'
        test_scene_file = 'data_prep/kitti/test_scenes_' + split + '.txt'
        with open(get_resource_path(test_scene_file), 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.load_pose = load_pose
        self.cam_ids = ['02', '03']
        self.date_list = [
            '2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03'
        ]
        self.collect_static_frames(static_frames_file)
        self.collect_train_frames()
        self.shift_h=shift_h

    def collect_static_frames(self, static_frames_file):
        with open(get_resource_path(static_frames_file), 'r') as f:
            frames = f.readlines()
        self.static_frames = []
        for fr in frames:
            if fr == '\n':
                continue
            unused_date, drive, frame_id = fr.split(' ')
            fid = '%.10d' % (np.int(frame_id[:-1]))
            for cam_id in self.cam_ids:
                self.static_frames.append(drive + ' ' + cam_id + ' ' + fid)

    def collect_train_frames(self):
        r"""
        Create a list of training frames.
        """
        all_frames = []
        for date in self.date_list:
            date_dir = os.path.join(self.dataset_dir, date)
            if os.path.isdir(date_dir):
                drive_set = os.listdir(date_dir)
                for dr in drive_set:
                    drive_dir = os.path.join(date_dir, dr)
                    if os.path.isdir(drive_dir):
                        if dr[:-5] in self.test_scenes:
                            continue
                        for cam in self.cam_ids:
                            img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
                            num_frames = len(glob.glob(img_dir + '/*[0-9].png'))
                            for i in range(num_frames):
                                frame_id = '%.10d' % i
                                all_frames.append(dr + ' ' + cam + ' ' + frame_id)
        
        assert len(all_frames)>0, 'no kitti data found in the dataset_dir'

        for s in self.static_frames:
            try:
                all_frames.remove(s)
            except ValueError:
                pass
        
        assert len(all_frames)>0, 'all data are static_frames'

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    
    def is_valid_sample(self, target_index):
        r"""
        Check whether we can find a valid sequence around this frame.
        """
        num_frames = len(self.train_frames)
        target_drive, cam_id, _ = self.train_frames[target_index].split(' ')
        start_index, end_index = get_seq_start_end(target_index, self.seq_length)
        # Check if the indices of the start and end are out of the range
        if start_index < 0 or end_index >= num_frames:
            return False


        start_drive, start_cam_id, start_frame_id= self.train_frames[start_index].split(' ')
        end_drive, end_cam_id, end_frame_id = self.train_frames[end_index].split(' ')

        # check frame continuity
        if self.data_format == 'mono2':
            if (int(end_frame_id) - int(start_frame_id)) != (self.seq_length-1):
                return False

        # Check if the scenes and cam_ids are the same 
        if (target_drive == start_drive and target_drive == end_drive and
            cam_id == start_cam_id and cam_id == end_cam_id):
            return True
        return False

    def get_dynamic_points(self, img, idx):
        # Find the index of reference frame
        target_drive, cam_id, _ = self.train_frames[idx].split(' ')
        ref_idx = idx - 1
        ref_drive, ref_cam_id, _ = self.train_frames[ref_idx].split(' ')
        if ref_idx < 0 or ref_drive != target_drive or ref_cam_id != cam_id:
            pre_idx = idx + 1
        else:
            pre_idx = idx - 1
        
        # Load the reference frame
        pre_drive, pre_cam_id, pre_frame_id = self.train_frames[pre_idx].split(' ')
        infos = {
            'drive': pre_drive,
            'cam_id': pre_cam_id,
            'frame_id': pre_frame_id
        }
        img_pre, _ = self.load_image_raw(infos)

        # Convert images to gray scale and detect dynamic points
        img_gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dynamic_points, outlier_ratio = self.dynamic_object_detector(img_gray_pre, img_gray)

        return dynamic_points, outlier_ratio

    def load_image_sequence(self, target_index):
        r"""
        Return a sequence with requested target frame.
        """
        if self.data_format == 'struct2depth':
            start_index, end_index = get_seq_start_end(
                target_index,
                self.seq_length,
            )
        elif self.data_format == 'mono2':
            start_index = end_index = target_index
        
        image_seq = []
        dynamic_map_seq = []
        target_outlier_ratio = 0.0
        for index in range(start_index, end_index + 1):
            drive, cam_id, frame_id = self.train_frames[index].split(' ')
            infos = {
                'drive': drive,
                'cam_id': cam_id,
                'frame_id': frame_id
            }
            img, intrinsics = self.load_image_raw(infos)

            if self.detect_dynamic_objects:
                try:
                    dynamic_points, outlier_ratio = self.get_dynamic_points(img, index)
                    # Resize the coordinates of dynamic points
                    dynamic_points[:, 0] *= self.img_width / img.shape[1]
                    dynamic_points[:, 1] *= self.img_height / img.shape[0]
                except:
                    dynamic_points, outlier_ratio = None, 0
                if index == target_index:
                    target_outlier_ratio = outlier_ratio

            if index == target_index:
                zoom_y = self.img_height / img.shape[0]
                zoom_x = self.img_width / img.shape[1]

            # Notice the default mode for RGB images is BICUBIC
            img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))
            image_seq.append(img)

            # after resizing
            if self.detect_dynamic_objects:
                dynamic_map = np.zeros_like(img[:, :, 0], dtype=np.uint8)
                if dynamic_points is not None:
                    dynamic_points = dynamic_points.astype('int')                
                    dynamic_map[dynamic_points[:, 1], dynamic_points[:, 0]] = 255
                # Dilate the map for better recall rate
                dynamic_map = cv2.dilate(dynamic_map, np.ones((3, 3), np.uint8), iterations=1)
                dynamic_map_seq.append(dynamic_map)

        return image_seq, zoom_x, zoom_y, intrinsics, dynamic_map_seq, target_outlier_ratio

    def load_pose_sequence(self, target_index):
        r"""
        Returns a sequence of pose vectors for frames around the target frame.
        """
        target_drive, _, target_frame_id = self.train_frames[target_index].split(' ')
        target_pose = self.load_pose_raw(target_drive, target_frame_id)
        start_index, end_index = get_seq_start_end(target_frame_id, self.seq_length)
        pose_seq = []
        for index in range(start_index, end_index + 1):
            if index == target_frame_id:
                continue
            drive, _, frame_id = self.train_frames[index].split(' ')
            pose = self.load_pose_raw(drive, frame_id)
            # From target to index.
            pose = np.dot(np.linalg.inv(pose), target_pose)
            pose_seq.append(pose)
        return pose_seq


    def load_example(self, target_index):
        r"""
        Return a sequence with requested target frame.        
        """
        example = {}
        target_drive, target_cam_id, target_frame_id = (
            self.train_frames[target_index].split(' ')
        )
        infos = {
            'drive': target_drive,
            'cam_id': target_cam_id
        }

        image_seq, zoom_x, zoom_y, intrinsics, dynamic_map_seq, outlier_ratio = (
                self.load_image_sequence(target_index)
                )
        
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = target_drive + '_' + target_cam_id + '/'
        example['file_name'] = target_frame_id
        if self.load_pose:
            pose_seq = self.load_pose_sequence(target_index)
            example['pose_seq'] = pose_seq
        if self.detect_dynamic_objects:
            example['dynamic_map_seq'] = dynamic_map_seq
        example['outlier_ratio'] = outlier_ratio
        return example

    def load_pose_raw(self, drive, frame_id):
        date = drive[:10]
        pose_file = os.path.join(
            self.dataset_dir, date, drive, 'poses', frame_id + '.txt'
        )
        with open(pose_file, 'r') as f:
            pose = f.readline()
        pose = np.array(pose.split(' ')).astype(np.float32).reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape((1, 4))))
        return pose

    def load_image_raw(self, infos):
        r"""
        Load an raw image given its id.
        """
        drive = infos['drive']
        cam_id = infos['cam_id']
        frame_id = infos['frame_id']
        date = drive[:10]
        img_file = os.path.join(
            self.dataset_dir, 
            date, 
            drive, 
            'image_' + cam_id,
            'data', 
            frame_id + '.png'
            )
        img = imageio.imread(img_file)
        source_intrinsics = self.load_intrinsics(infos)
        # do image reprojection
        if self.target_intrinsics is not None:
            img, updated_target_intrinsics, crop_left = self.reproject_img(
                    img, source_intrinsics,
                    self.target_intrinsics,
                    self.target_height, self.target_width)
            allowed_height = int(img.shape[1] * self.img_height/self.img_width)
            crop_top = int(self.shift_h * img.shape[0])
            if (crop_top+allowed_height) > img.shape[0]:
                # TODO: revise crop_top
                img = img[-allowed_height:]
            else:
                img = img[crop_top:crop_top+allowed_height]
            # adjust the v principal point
            intrinsics = self.scale_intrinsics(updated_target_intrinsics, 1.,1.,
                                               crop_top=crop_top,
                                               crop_left=crop_left)
        else:
            intrinsics=source_intrinsics

        return img, intrinsics

    def load_intrinsics(self, infos):
        r"""
        Load the intrinsic matrix given its id.
        """
        drive = infos['drive']
        cam_id = infos['cam_id']
        date = drive[:10]
        calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')
        filedata = self.read_raw_calib_file(calib_file)
        p_rect = np.reshape(filedata['P_rect_' + cam_id], (3, 4))
        intrinsics = p_rect[:3, :3]
        return intrinsics

    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    def read_raw_calib_file(self, filepath):
        r"""
        Read in a calibration file and parse into a dictionary.
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which we don't
                # care about.
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
