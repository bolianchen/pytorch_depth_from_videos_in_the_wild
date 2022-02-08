# All rights reserved.

import os
import argparse
import yaml
import pandas as pd

# TODO: add comments
DATASETS = ['video', 'kitti_raw_eigen', 'kitti_raw_stereo']

class DataGenOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                description='Options for training data generation')
        self.parser.add_argument('--dataset_name',
                                 type=str, choices=DATASETS,
                                 default='video',
                                 help='what raw dataset to convert'
                                      'video: videos in mp4 format')
        self.parser.add_argument('--dataset_dir',
                                 type=str, default='./raw_data',
                                 help='location of the folder containing the '
                                      'raw dataset')
        self.parser.add_argument('--save_dir',
                                 type=str, default='./generated_data',
                                 help='location to save the generated '
                                      'training data.')
        self.parser.add_argument('--save_img_ext',
                                 type=str, choices=['png', 'jpg'],
                                 default='png',
                                 help='what image format to save')
        self.parser.add_argument('--seq_length',
                                 type=int, default=3,
                                 help='number of images of each training '
                                      'sequence')
        self.parser.add_argument('--img_height',
                                 type=int, default=128,
                                 help='height of the generated images')
        self.parser.add_argument('--img_width',
                                 type=int, default=416,
                                 help='width of the generated images')
        self.parser.add_argument('--data_format',
                                 type=str, choices=['mono2', 'struct2depth'],
                                 default='mono2',
                                 help='mono2: a single generated image is '
                                      'converted from a single raw image'
                                      'struct2depth: a single generated image'
                                      ' is a concatenation of raw images in a '
                                      'training sequence')
        self.parser.add_argument('--del_static_frames',
                                 action='store_true',
                                 help='remove frames when the camera is '
                                      'judged as not moving with a heuristic '
                                      'algorithm implemented by us')
        self.parser.add_argument('--intrinsics',
                                 type=str, default=None,
                                 help='a document containing 9 entries '
                                      'of the flattened target intrinsics')
        self.parser.add_argument('--trim',
                                 nargs=4,
                                 type=float, default=[0.0, 0.0, 0.0, 0.0],
                                 help='romove the [left, right, top, bottom] '
                                      'part of each frame by this proportion'
                                      '; this operation WILL NOT induce '
                                      'intrinsics adjustment')
        self.parser.add_argument('--crop',
                                 nargs=4,
                                 type=float, default=[0.0, 0.0, 0.0, 0.0],
                                 help='romove the [left, right, top, bottom] '
                                      'part of each frame by this proportion'
                                      '; this operation WILL induce '
                                      'intrinsics adjustment')
        self.parser.add_argument('--augment_strategy',
                                 type=str, choices=['none', 'single', 'multi'],
                                 default='single',
                                 help='multi: augment data with 3 pre-defined '
                                      'cropping; '
                                      'single: crop images according to '
                                      'shift_h '
                                      'none: no cropping, for random cropping '
                                      'during the training')
        self.parser.add_argument('--augment_shift_h',
                                 type=float, default=0.0,
                                 help='what proportion from the top to crop '
                                      'a frame. this only applies when augment'
                                      '_strategy is set to single')
        self.parser.add_argument('--video_start',
                                 type=str,
                                 default='00:00:00',
                                 help='set a start time for the video '
                                      'conversion; the format should be '
                                      'hh:mm:ss')
        self.parser.add_argument('--video_end',
                                 type=str,
                                 default='00:00:00',
                                 help='set an end time for the video '
                                      'conversion; the format should be '
                                      'hh:mm:ss; if set to 00:00:00, convert '
                                      'the video till the end')
        self.parser.add_argument('--fps',
                                 type=int,
                                 default=10,
                                 help='frames per second to sample from a '
                                      ' video to do the conversion')
        self.parser.add_argument('--delete_temp',
                                 action='store_false',
                                 help='remove temporary images during '
                                      'conversion')
        self.parser.add_argument('--num_threads',
                                 type=int,
                                 help='number of worker threads. the default '
                                      ' is the CPU cores.')
        self.parser.add_argument('--batch_size',
                                 type=int, default=4,
                                 help='batch size to run Mask-RCNN model')
        self.parser.add_argument('--threshold',
                                 type=float, default=0.5,
                                 help='score threshold for Mask-RCNN model')
        self.parser.add_argument('--mask',
                                 type=str, choices=['none', 'mono', 'color'],
                                 default='mono',
                                 help='what segmentation masks to generate '
                                      'none: do not generate masks '
                                      'mono: generate binary masks '
                                      'color: pixel values vary on masks by '
                                      'object instances')
        self.parser.add_argument('--single_process',
                                 action='store_true',
                                 help='only use a single gpu process '
                                      'this option is mainly for debugging')
        self.parser.add_argument('--to_yaml',
                                 action='store_true',
                                 help='save the options to a yaml file')

    def parse(self):
        """
        Parse arguments from both command line and YAML configuration.
        The order of looking for the value for an argument is as follows:
        command line -> YAML configuration (if provided) -> default value.
        """
        conf_parser = argparse.ArgumentParser(add_help=False)
        conf_parser.add_argument('--config',
                                 type=str, default=None,
                                 help='the path to load YAML configuration; '
                                      'options set in this file may be '
                                      'overridden by command-line arguments')
        conf_arg, remaining_args = conf_parser.parse_known_args()

        if conf_arg.config:
             with open(conf_arg.config, 'r') as f:
               config = yaml.load(f, yaml.FullLoader)
             self.parser.set_defaults(**config)
     
        return self.parser.parse_args(remaining_args)