# All rights reserved.

import os
import argparse

# TODO: add comments
DATASETS = ['video', 'kitti_raw_eigen', 'kitti_raw_stereo']

class DataGenOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                description='Options for training data generation')
        self.parser.add_argument('--dataset_name',
                                 type=str, choices=DATASETS, required=True,
                                 help='what raw dataset to convert'
                                      'video: videos in mp4 format')
        self.parser.add_argument('--dataset_dir',
                                 type=str, required=True,
                                 help='location of the folder containing the '
                                      'raw dataset')
        self.parser.add_argument('--save_dir',
                                 type=str, required=True,
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
        # TODO: refactor cut and crop related codes
        # refactor to [x1, x2, y1, y2] format in the future
        # cut does not induce intrinsics adjustment but crop does
        self.parser.add_argument('--cut',
                                 action='store_true',
                                 help='for video dataset, whether to pre-crop '
                                      'each frame')
        self.parser.add_argument('--crop_left',
                                 type=int, default=0,
                                 help='how many pixels to crop from the left')
        self.parser.add_argument('--crop_right',
                                 type=float, default=0.0,
                                 help='what proportion from the right to crop')
        self.parser.add_argument('--shift_h',
                                 type=float, default=0.0,
                                 help='what proportion from the top to crop')
        self.parser.add_argument('--crop_bottom',
                                 type=float, default=0.0,
                                 help='what proportion from the bottom to crop')
        self.parser.add_argument('--target_intrinsics',
                                 type=str,
                                 help='a document containing 9 entries '
                                      'of the flattened target intrinsics')
        self.parser.add_argument('--target_height',
                                 type=int,
                                 help=' height of the canvas for reprojection')
        self.parser.add_argument('--target_width',
                                 type=int,
                                 help=' height of the canvas for reprojection')
        self.parser.add_argument('--crop',
                                 type=str, choices=['none', 'single', 'multi'],
                                 default='single',
                                 help='multi: augment data with 3 pre-defined'
                                      'cropping; '
                                      'single: crop images according to '
                                      'shift_h '
                                      'none: no cropping, for random cropping '
                                      'during the training')
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
                                      'none: do not generate masks'
                                      'mono: generate binary masks '
                                      'color: pixel values vary on masks by '
                                      'object instances')
        self.parser.add_argument('--single_process',
                                 action='store_true',
                                 help='only use a single gpu process '
                                      'this option is mainly for debugging')

    def parse(self):
        return self.parser.parse_args()
