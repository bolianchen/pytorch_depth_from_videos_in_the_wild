from __future__ import absolute_import, division, print_function

import os
import argparse
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EvalBaseOptions:
    """Options to initialize models and image preprocessor
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--method', type=str, default='',
                                 choices = ['', 'monodepth2', 'manydepth',
                                            'wild', 'summer2021'],
                                 help = 'this argument is only needed when'
                                        ' opt.json is not available. '
                                        ' For instance, when the officially'
                                        ' released models are applied')
        # MODEL INITIALIZATION
        self.parser.add_argument('--custom_model_path', type=str,
                                help='relative or absolute path to the model directory, '
                                    ' either this or model_name should be given')
        self.parser.add_argument('--model_name', type=str,
                                choices=[
                                    "mono_640x192",
                                    "stereo_640x192",
                                    "mono+stereo_640x192",
                                    "mono_no_pt_640x192",
                                    "stereo_no_pt_640x192",
                                    "mono+stereo_no_pt_640x192",
                                    "mono_1024x320",
                                    "stereo_1024x320",
                                    "mono+stereo_1024x320",
                                    "mono_model"],
                                help='name of a pretrained model to use, '
                                    'either this or custom_model_path should be given')
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
