import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.data_iterators import ImageReader, VideoReader
from lib.img_processing import ImageProcessor, concat_depth_img
from lib.utils import get_model_opt

from evaluators import EVALUATORS
from options import InferOptions

class DepthInference:
    def __init__(self, args):
        self.args = args
        self._init_evaluator()
        self._init_img_processor()
        self._init_data_reader()

    def _init_evaluator(self):
        model_opt = get_model_opt(self.args.model_path)
        self.evaluator = EVALUATORS[model_opt.method](model_opt)

    def _init_img_processor(self):
        model_h, model_w = self.evaluator.get_training_res()
        self.img_processor = ImageProcessor(
                self.args, model_h, model_w)

    def _init_data_reader(self):
        """Make an image iterator
        """
        fp = self.args.input_path

        if fp.endswith('mp4'):
            self.data_reader = VideoReader(fp)

        else:
            self.data_reader = ImageReader(fp)

    def infer(self):
        frame_results = {}
        for img in self.data_reader:
            img_ori = ImageProcessor.cut_img(
                    img, self.args.cut, cut_h=self.args.cut_h,
                    cut_w=self.args.cut_w)
            img = self.img_processor.process(img_ori)
            disp_colormap, _ = self.evaluator.estimate_depth(img)
            disp_img = concat_depth_img(disp_colormap, img_ori, self.args.shift_h)
            disp_img_bgr = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('disp_img', disp_img_bgr)
            cv2.waitKey(1)


if __name__ == '__main__':
    
    depth_estimator = DepthInference(InferOptions().parse()[0])
    depth_estimator.infer()

