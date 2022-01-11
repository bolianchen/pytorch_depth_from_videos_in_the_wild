# All rights reserved.

from __future__ import absolute_import, division, print_function

import os
import argparse
from .base_options import BaseOptions

class WildOptions(BaseOptions):
    def __init__(self):
        
        super(WildOptions, self).__init__()
        self.parser.description = 'Options for Depth from the Videos in the Wild' 

        # MODEL ARCHITECTURES
        ## DEPTH ENCODER
        self.parser.add_argument('--not_use_layernorm',
                                 action='store_true',
                                 help='whether to use layer normalization'
                                      'in depth network')
        self.parser.add_argument('--layernorm_noise_rampup_steps',
                                 type=int,
                                 default=10000,
                                 help='rate to rampup the layernorm noise')
        self.parser.add_argument('--use_norm_in_downsample',
                                 action='store_true',
                                 help='whether to use norm_layer'
                                      'in downsample')
        self.parser.add_argument('--use_mono2_depth_decoder',
                                 action='store_true',
                                 help='set true to use the monodepth2 depth '
                                      'decoder architecture')

        ## MOTIONFIELDNET
        self.parser.add_argument('--learn_intrinsics',
                                 action='store_true',
                                 help='set true to learn camera intrinsics')
        self.parser.add_argument('--foreground_dilation',
                                 type=int,
                                 default=8,
                                 help='dilation of the foreground masks')

        # MODEL INITIALIZATION
        self.parser.add_argument('--models_to_init',
                                 nargs='+',
                                 type=str,
                                 default=['depth', 'pose', 'motion',
                                          'intrinsics_head'],
                                 help='models to initialize parameters '
                                      'according to init_mode. Exclude encoder'
                                      ' because it usually loads imagenet '
                                      'pretrained checkpoint. '
                                      'Selectable from encoder, depth, pose, '
                                      'motion, scaler, and intrinsics_head')
        self.parser.add_argument('--init_encoder_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='methods to initialize model '
                                      'parameters')
        self.parser.add_argument('--init_encoder_params',
                                 nargs='+',
                                 type=float,
                                 default=[0.5],
                                 help='')
        self.parser.add_argument('--init_depth_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='methods to initialize model '
                                      'parameters')
        self.parser.add_argument('--init_depth_params',
                                 nargs='+',
                                 type=float,
                                 default=[0.5],
                                 help='')
        self.parser.add_argument('--init_pose_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='methods to initialize model '
                                      'parameters')
        self.parser.add_argument('--init_pose_params',
                                 nargs='+',
                                 type=float,
                                 default=[0.5],
                                 help='')
        self.parser.add_argument('--init_motion_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='methods to initialize model '
                                      'parameters')
        self.parser.add_argument('--init_motion_params',
                                 nargs='+',
                                 type=float,
                                 default=[0.5],
                                 help='')
        self.parser.add_argument('--init_intrinsics_head_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='methods to initialize model '
                                      'parameters')
        self.parser.add_argument('--init_intrinsics_head_params',
                                 nargs='+',
                                 type=float,
                                 default=[0.5],
                                 help='')

        # LOAD TRAINED MODELS
        self.parser.add_argument('--models_to_load',
                                 nargs='+',
                                 type=str,
                                 default=[],
                                 help='networks to load from the folder '
                                      'defined by the load_weights_folder '
                                      ' that is in the base_options.py'
                                      'Selectable from encoder, depth, pose, '
                                      'motion, scaler, and intrinsics_head')
        self.parser.add_argument('--models_to_freeze',
                                 nargs='+',
                                 type=str,
                                 default=[],
                                 help='models to freeze')

        # LOSS COMPUTATIONS
        self.parser.add_argument('--losses_to_use',
                                 nargs='+',
                                 type=str,
                                 default=['reconstr_loss',
                                          'depth_consistency_loss',
                                          'ssim_loss', 'rot_loss',
                                          'trans_loss', 'smooth_loss',
                                          'motion_smooth_loss'],
                                 help='losses used for training')
        self.parser.add_argument('--reconstr_loss_weight',
                                 type=float,
                                 default = 0.85,
                                 help='weighting for frame reconstruction loss')
        self.parser.add_argument('--ssim_loss_weight',
                                 type=float,
                                 default = 1.5,
                                 help='weighting for SSIM loss')
        self.parser.add_argument('--smooth_loss_weight',
                                 type=float,
                                 default = 1e-2,
                                 help='weighting for depth smooth loss')
        self.parser.add_argument('--motion_smooth_loss_weight',
                                 type=float,
                                 default = 1e-3,
                                 help='weighting for motion smooth loss')
        self.parser.add_argument('--depth_consistency_loss_weight',
                                 type=float,
                                 default = 1e-2,
                                 help='weighting for depth_consistency_loss')
        self.parser.add_argument('--rot_loss_weight',
                                 type=float,
                                 default = 1e-3,
                                 help='weighting for rotation consistency loss')
        self.parser.add_argument('--trans_loss_weight',
                                 type=float,
                                 default = 1e-2,
                                 help='weighting for translation consistency loss')

        # OPTIMIZATION
        ## Regularization
        self.parser.add_argument('--encoder_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='')
        self.parser.add_argument('--depth_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='')
        self.parser.add_argument('--pose_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='')
        self.parser.add_argument('--pose_other_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='conv layers in background_motion_conv '
                                      ', rot_scale and trans_scale')
        self.parser.add_argument('--motion_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='')
        self.parser.add_argument('--intrinsics_head_weight_decay',
                                 type=float,
                                 default = 0,
                                 help='')

        # LOGGING
        self.parser.add_argument('--log_mobile_mask',
                                 type=str,
                                 default='none',
                                 choices = ['none', 'normal', 'dilated'],
                                 help='whether to save possibly mobile '
                                      'masks to tensorboard')
        self.parser.add_argument('--log_trans',
                                 type=str,
                                 default='none',
                                 choices = ['none', 'whole', 'masked'],
                                 help='whether to save residual translation')

    def parse(self):
        self.options, unknown_args = super().parse()
        try:
            idx = self.options.models_to_init.index('intrinsics_head')
            if not self.options.learn_intrinsics:
                self.options.models_to_init.pop(idx)
        except:
            pass
        return self.options, unknown_args
