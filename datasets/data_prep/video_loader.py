# All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import sys
import shutil
import re
import torch

from absl import logging
import numpy as np
import pandas as pd
import imageio
from PIL import Image
import cv2

from .base_loader import (
    BaseLoader, 
    get_resource_path, 
    get_seq_start_end,
    natural_keys
)
from .preprocess import (
    edge_change_ratio, 
    postprocess, 
    SimpleQueue
)

class Video(BaseLoader):
    r"""
    Make dataloader from any videos in a folder
    """

    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 data_format='mono2',
                 mask='none',
                 batch_size=32,
                 threshold=0.5,
                 sample_every=1,
                 intrinsics=None,
                 trim=[0,0,0,0],
                 crop=[0,0,0,0],
                 del_static_frames=False,
                 augment_strategy='multi',
                 augment_shift_h=0.0,
                 fps=10,
                 video_start=0,
                 video_end=0,
                 img_ext='png'):
        super().__init__(dataset_dir, img_height, img_width, 
                         seq_length, data_format, mask, batch_size, 
                         threshold)
        self.sample_every = sample_every
        self.intrinsics_path = intrinsics
        self.intrinsics = None
        self.trim = trim != [0,0,0,0]
        self.trim_proportion = trim
        self.crop_proportion = crop
        self.del_static_frames = del_static_frames
        self.augment_strategy = augment_strategy
        self.augment_shift_h = augment_shift_h
        self.fps = fps
        self.video_start = pd.Timedelta(video_start).seconds * 1000 # seconds to miliseconds
        self.video_end = pd.Timedelta(video_end).seconds * 1000
        self.img_ext = img_ext

        # Collect frames from videos
        self.throwaway_frames = set()
        self.videos = self.collect_videos()
        self.vid2img()
        self.video_dirs = self.collect_video_dirs()
        self.frames = self.collect_frames()
        self.num_frames = len(self.frames)
        self.num_train = self.num_frames
        logging.info('Total frames collected: %d', self.num_frames)
    
    def collect_videos(self):
        r"""
        Collect absolute paths of the video files for conversion.
        """
        videos = glob.glob(os.path.join(self.dataset_dir, '*.mp4'))
        return videos

    def collect_video_dirs(self):
        r"""
        Return a list of names of all the video directories.
        """
        # Names of all the video directories
        video_dirs = []

        # Iterate through all the file names
        for item in os.listdir(self.dataset_dir):
            # Collect directories named with the videos' names
            if os.path.isdir(os.path.join(self.dataset_dir, item)):
                # Only collect folder names
                video_dirs.append(item)
        return video_dirs

    def vid2img(self):
        r"""
        Convert videos to images without rescaling.
        """
        fps = self.fps
        save_dir = self.dataset_dir
        img_ext = self.img_ext

        for video in self.videos:
            logging.info(f'Converting {video} into images')
            vidcap = cv2.VideoCapture(video)

            # Set current position of the video to start time
            vidcap.set(cv2.CAP_PROP_POS_MSEC, self.video_start)

            # Get the approximate frame rate 
            raw_fps = vidcap.get(cv2.CAP_PROP_FPS)
            if fps:
                # Original fps
                assert raw_fps >= fps, "The specified fps is higher than the raw video"
                # The period of saving images from the video
                period = round(raw_fps/fps)
            else:
                # Save every frame
                period = 1
            
            # The folder to save images of a specific root 
            path = os.path.join(
                save_dir, os.path.basename(video).split('.')[0]
            )
            os.makedirs(path, exist_ok=True)

            # Reset data structures to determine static frames
            # for a new video
            if self.del_static_frames:
                static_frames = []
                img_queue = SimpleQueue(3)
                ecr_queue = SimpleQueue(5)

            count = 0
            while True:
                success, image = vidcap.read()
                # Repeat if video reading has not started
                if vidcap.get(cv2.CAP_PROP_POS_MSEC) == 0.0:
                    success, image = vidcap.read()
                
                # End the conversion if it exceeds end time
                if self.video_end and \
                   vidcap.get(cv2.CAP_PROP_POS_MSEC) > self.video_end:
                    break

                if success:
                    if count % period == 0:
                        save_idx = count // period
                        if self.trim:
                            image = self._trim(image, self.trim_proportion)

                        # Remove static frames
                        if self.del_static_frames:
                            how_is_frame = self.frame_judge(image, img_queue, ecr_queue)
                        else:
                            how_is_frame = 'ok'

                        path_to_save_temp = os.path.join(
                                path,
                                "{:010d}.{}".format(save_idx, img_ext))
                        frame_info = os.path.join(
                                os.path.basename(video).split('.')[0],
                                "{:010d}.{}".format(save_idx, img_ext))

                        if how_is_frame == 'ok':
                            cv2.imwrite(path_to_save_temp, image)
                        elif how_is_frame == 'static':
                            static_frames.append(frame_info)
                        elif how_is_frame == 'bad':
                            self.throwaway_frames.add(frame_info)
                        else:
                            raise NotImplementedError

                    if count % 500 == 0:
                        print(f'{count} raw frames have been processed')
                    count+=1
                else:
                    break
            
            if self.del_static_frames:
                if static_frames: 
                    # If static_frames is not empty,
                    # add static frames to self.throwaway_frames
                    self.discard_static_frames(static_frames)

    def frame_judge(self, image, img_queue, ecr_queue):
        r"""
        Return a string representing the input image quality.
        Returns:
        'ok', 'bad', 'static'
        'ok' means the image is qualified or there is no sufficient info.
        'bad' means the image is broken.
        'static' means the image is taken when the car stops.
        """
        
        img_queue.add(image)

        if img_queue.full():
            prev_img = img_queue.get(3)
            curr_img = img_queue.get(1)
            img_queue.pop()
            ecr = edge_change_ratio(prev_img, curr_img) 

            if ecr >= 0.9 or ecr <= 0.05:
                return 'bad'
            else:
                ecr_queue.add(ecr)

        if ecr_queue.full():
            ECR_avg = ecr_queue.average()
            ecr_queue.pop()
            if ECR_avg < 0.3:
                return 'static'

        return 'ok'

    def discard_static_frames(self, static_frames):
        r"""
        Delete adjacent frames of discontinuous static frames.
        """

        parent_folder = os.path.dirname(static_frames[0])

        def file_name_formatter(index):
            """Return the file name given the index"""
            path = os.path.join(parent_folder,
                                "{:010d}.{}".format(index, self.img_ext))
            return path

        def is_continuous(frame1, frame2):
            idx1 = index_getter(frame1)
            idx2 = index_getter(frame2)

            return abs(idx1 - idx2) == 1

        def remove_frames(target_frame, direction, num_of_frames):

            initial_index = index_getter(target_frame)

            if direction == 'forward':
                for i in range(1, num_of_frames + 1):
                    frame_path = file_name_formatter(initial_index + i)
                    self.throwaway_frames.add(frame_path)

            elif direction == 'backward':
                for i in range(1, num_of_frames + 6):
                    if initial_index - i < 0:
                        break
                    frame_path = file_name_formatter(initial_index - i)
                    self.throwaway_frames.add(frame_path)

        for i, frame in enumerate(static_frames):

            self.throwaway_frames.add(frame)
            
            if i == 0:
                remove_frames(frame, 'backward', 30)
            if i == len(static_frames) -1:
                remove_frames(frame, 'forward', 30)
                continue
            if not is_continuous(frame, static_frames[i+1]):
                remove_frames(frame, 'forward', 30)
                remove_frames(static_frames[i+1], 'backward', 30)
    
    def _trim(self, img, proportion):
        r"""
        Trim an image.
        This Function is to trim off a portion of the input frame. 
        Since there is no following adjustment of intrinsics, it should only be 
        applied when the frame is composed of concatenation of images from 
        different camera.
        """
        left, right, top, bottom = proportion
        h, w, _ = img.shape
        left, right = int(w * left), int(w * (1 - right))
        top, bottom = int(h * top), int(h * (1 - bottom))
        return img[top:bottom, left:right, :]
    
    def collect_frames(self):
        r"""
        Create a list of unique ids for available frames.
        """
        frames = []
        for video_dir in self.video_dirs:
            # Absolute paths
            im_files = glob.glob(os.path.join(self.dataset_dir, video_dir, f'*.{self.img_ext}'))

            # Sort images in a video directory; this sorting works even 
            # when the image indices are not formated to same digits
            im_files = sorted(im_files, key=natural_keys)

            if self.augment_strategy == 'multi':
                # Adding 3 crops of the video.
                frames.extend(['A' + video_dir + '/' + os.path.basename(f) for f in im_files])
                frames.extend(['B' + video_dir + '/' + os.path.basename(f) for f in im_files])
                frames.extend(['C' + video_dir + '/' + os.path.basename(f) for f in im_files])
            elif self.augment_strategy == 'single':
                frames.extend(['S' + video_dir + '/' + os.path.basename(f) for f in im_files])
            elif self.augment_strategy == 'none':
                frames.extend(['N' + video_dir + '/' + os.path.basename(f) for f in im_files])
            else:
                raise NotImplementedError(f'crop {self.crop} not supported')
        return frames

    def is_bad_sample(self, target_index):
        return self.frames[target_index][1:] in self.throwaway_frames

    
    def is_valid_sample(self, target_index):
        r"""
        Check whether we can find a valid sequence around this frame.
        """
        target_video, _ = self.frames[target_index].split('/')

        start_index, end_index = get_seq_start_end(
            target_index, self.seq_length, self.sample_every
        )
        if start_index < 0 or end_index >= self.num_frames:
            return False

        start_video, _ = self.frames[start_index].split('/')
        end_video, _ = self.frames[end_index].split('/')
        if target_video != start_video or target_video != end_video:
            return False

        # Check if all the the collected frames are continuous
        first_frame_local_index = index_getter(self.frames[start_index])
        end_frame_local_index = index_getter(self.frames[end_index])
        if (end_frame_local_index - first_frame_local_index) != self.seq_length-1:
            return False

        # Check if any of the adjacent frames is static
        for idx in range(start_index, end_index+1):
            frame_info = self.frames[idx][1:]
            if frame_info in self.throwaway_frames:
                return False

        return True        
    
    def load_image_sequence(self, target_index):
        r"""
        Return a sequence with requested target frame.
        """
        if self.data_format == 'struct2depth':
            start_index, end_index = get_seq_start_end(
                target_index,
                self.seq_length,
                self.sample_every
            )
        elif self.data_format == 'mono2':
            start_index = end_index = target_index

        image_seq = []
        for idx in range(start_index, end_index + 1, self.sample_every):
            frame_id = self.frames[idx]
            img, crop_top, crop_left = self.load_image_raw({'frame_id': frame_id})

            if self.augment_strategy == 'none':
                size = (self.img_width, int(img.shape[0] * self.img_width / img.shape[1]))
            else:
                size = (self.img_width, self.img_height)
            if idx == target_index:
                zoom_y = size[1] / img.shape[0] # target_h / original_h
                zoom_x = size[0] / img.shape[1] # target_w / original_w

            # Notice the default mode for RGB images rescaling is BICUBIC
            img = np.array(Image.fromarray(img).resize(size))
            image_seq.append(img)

        return image_seq, zoom_x, zoom_y, crop_top, crop_left

    def load_example(self, target_index):
        r"""
        Return a sequence with requested target frame.
        Every loader should implement its own method.
        """
        example = {}
        image_seq, zoom_x, zoom_y, crop_top, crop_left = self.load_image_sequence(target_index)

        target_video, target_filename = self.frames[target_index].split('/')
        if self.augment_strategy == 'multi':
            # Put A, B, C at the end for better shuffling.
            target_video = target_video[1:] + target_video[0]
        elif self.augment_strategy == 'single' or self.augment_strategy == 'none':
            target_video = target_video[1:]
        else:
            raise NotImplementedError(f'crop {self.crop} not supported')

        # First adjust intrinsics due to cropping
        intrinsics = self.load_intrinsics({'crop_top': crop_top, 'crop_left': crop_left})
        # Then adjust intrinsics due to rescaling
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = target_video
        example['file_name'] = target_filename.split('.')[0]
        return example

    def load_image_raw(self, infos):
        r"""
        Load an raw image given its id.
        Every loader should implement its own method.
        """
        frame_id = infos['frame_id']
        crop_type = frame_id[0]
        img_file = os.path.join(self.dataset_dir, frame_id[1:])
        img = imageio.imread(img_file)

        # Image shape (H, W, C)
        crop_left, crop_right, crop_top, crop_bottom = self.crop_proportion
        crop_left = int(img.shape[1] * crop_left)
        crop_right = int(img.shape[1] * (1 - crop_right))
        crop_top = int(img.shape[0] * crop_top)
        crop_bottom = int(img.shape[0] * (1 - crop_bottom))

        # Crop the image
        img = img[crop_top:crop_bottom, crop_left:crop_right, :]

        allowed_height = int(img.shape[1] * self.img_height / self.img_width)
        # Starting height for the middle crop.
        mid_crop_top = int(img.shape[0] / 2 - allowed_height / 2)
        # How much to go up or down to get the other two crops.
        # Hard coded as one-third of the top cropping
        # crop_top will be used to adjust the princial point y in the intrinsics
        # due to the cropping
        height_var = int(mid_crop_top / 3)
        if crop_type == 'A':
            crop_top = mid_crop_top - height_var
        elif crop_type == 'B':
            crop_top = mid_crop_top
        elif crop_type == 'C':
            crop_top = mid_crop_top + height_var
        elif crop_type == 'S':
            crop_top = int(self.augment_shift_h * img.shape[0])
        elif crop_type == 'N':
            crop_top = 0
        else:
            raise ValueError('Unknown crop_type: %s' % crop_type)

        if crop_type == 'N':
            crop_bottom = img.shape[0]
        else:
            crop_bottom = crop_top + allowed_height + 1

        return img[crop_top:crop_bottom, :, :], crop_top, crop_left

    def load_intrinsics(self, infos):
        r"""
        Load the intrinsic matrix given its id.
        Every loader should implement its own method.
        """
        # Load the intrinsic matrix
        if self.intrinsics is None:
            if self.intrinsics_path:
                self.intrinsics = np.loadtxt(self.intrinsics_path).reshape(3, 3)
            else:
                # Default intrinsics
                # https://www.wired.com/2013/05/calculating-the-angular-view-of-an-iphone/
                # https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
                # https://stackoverflow.com/questions/39992968/how-to-calculate-field-of-view-of-the-camera-from-camera-intrinsic-matrix
                # # iPhone: These numbers are for images with resolution 720 x 1280.
                # Assuming FOV = 50.9 => fx = (1280 // 2) / math.tan(fov / 2) = 1344.8
                self.intrinsics = np.array(
                    [[ 1344.8,      0.0,  640.0],
                    [     0.0,   1344.8,  360.0],
                    [     0.0,      0.0,    1.0]]
                )

        crop_top = infos['crop_top']
        crop_left = infos['crop_left']
        intrinsics = self.intrinsics[:]
        intrinsics[0, 2] -= crop_left
        intrinsics[1, 2] -= crop_top

        return intrinsics

    def delete_temp_images(self):
        r"""
        Delete the intially converted images from videos.
        """
        for video_dir in self.video_dirs:
            try:
                shutil.rmtree(os.path.join(self.dataset_dir, video_dir))
            except:
                assert False, f'error occurred while deleting {video_dir}'

def index_getter(frame):
    r"""
    Return frame index in int format.
    """
    index = os.path.basename(frame).split('.')[0]
    return int(index)
