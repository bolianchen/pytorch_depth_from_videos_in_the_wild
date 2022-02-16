import os
import numpy as np
import torch
from torchvision import transforms

from trainers import WildTrainer
from options import WildOptions

from .base_evaluator import BaseEvaluator

class WildEvaluator(BaseEvaluator):
    def __init__(self, opt):
        self.opt = opt
        if not hasattr(self, 'num_pose_frames'):
            self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)
        self.opt.models_to_load = ['encoder', 'depth', 'pose', 'motion']
        super().__init__(self.opt)

    def _init_depth_net(self):
        WildTrainer._init_depth_net(self)
    
    def _init_pose_net(self):
        WildTrainer._init_pose_net(self)

    def _load_models(self):
        WildTrainer.load_model(self)

    def estimate_depth(self, img):
        with torch.no_grad():
            img = transforms.ToTensor()(np.array(img)).unsqueeze(0)
            features = self.models['encoder'](img.to(self.device))
            outputs = self.models['depth'](features)
        depth = outputs[("depth", 0)]
        disp = 1/depth
        disp_colormap = self._color_disp(disp)

        return disp_colormap, depth
