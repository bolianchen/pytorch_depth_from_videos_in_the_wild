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
                                default="./output", help="dir to save output image")

        # MODEL INITIALIZATION
        self.parser.add_argument('--model_path', type=str,
                                help='relative or absolute path to the model directory, '
                                    ' either this or model_name should be given')

        # TODO: Refactor as gen_data.py
        #  IMAGE PROCESSING for EVALUATIONS
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

    def parse(self):
        return self.parser.parse_known_args()
