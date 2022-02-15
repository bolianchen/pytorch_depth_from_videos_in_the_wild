from __future__ import absolute_import, division, print_function

import os
import argparse
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class InferOptions:
    """Options to initialize models and image preprocessor
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # DEPTH ESTIMATION
        self.parser.add_argument("--input_path", type=str, required=True,
                                help="path of a video or a folder of image")
        self.parser.add_argument("--output_dir", type=str,
                                default="./", help="dir to save output image")

        # MODEL INITIALIZATION
        self.parser.add_argument('--model_path', type=str,
                                help='relative or absolute path to the model directory, '
                                    ' either this or model_name should be given')

        # TODO: Refactor as gen_data.py
        #  IMAGE PROCESSING for EVALUATIONS
        self.parser.add_argument("--cut", type=str, default='',
                                 choices=['', 'upleft', 'upright', 'downleft',
                                          'downright'],
                                 help="to cut videos from 4 fused cameras")
        self.parser.add_argument("--cut_h", type=int, default=720,
                                 help="height to cut")
        self.parser.add_argument("--cut_w", type=int, default=1280,
                                 help="width to cut")
        self.parser.add_argument("--shift_h",
                                 type=float,
                                 default=0.0,
                                 help="whether to align the resolution with KITTI")
        self.parser.add_argument("--shift_w",
                                type=float,
                                default=0.0,
                                help="whether to align the resolution with KITTI")

    def parse(self):
        return self.parser.parse_known_args()
