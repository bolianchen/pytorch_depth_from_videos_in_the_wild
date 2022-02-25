# All rights reserved.

from __future__ import absolute_import, division, print_function

import os
import argparse
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Basic Options for trainers')

        # METHODOLOGY
        self.parser.add_argument('--method',
                                 type=str,
                                 default= 'wild',
                                 help='depth estimation methodology to use '
                                      'actually, this argument is parsed and '
                                      'used in train.py. The one here is to '
                                      'be saved in opt.json')

        # PATHS
        self.parser.add_argument('--data_path',
                                 nargs='+',
                                 type=str,
                                 help='absolute or relative path to the '
                                      'project folders') 
        self.parser.add_argument('--log_dir',
                                 type=str,
                                 default=os.path.join(project_dir, 'models'),
                                 help='log directory, the root folder to save '
                                      'the trained models')
        self.parser.add_argument('--images_to_predict_depth',
                                 type=str,
                                 default='',
                                 help='path to the folder containing the '
                                      'images only for depth prediction')
        self.parser.add_argument('--model_name',
                                 type=str,
                                 default='',
                                 help='the folder name to save the model ')
        self.parser.add_argument('--overwrite_old',
                                 action='store_true',
                                 help='if set true, overwrite the existing '
                                      'files in the folder to save models')

        # MODEL ARCHITECTURES
        ## RESNET-based DEPTH ENCODER
        self.parser.add_argument('--num_layers',
                                 type=int,
                                 help='number of resnet layers',
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])

        # MODEL INITIALIZATION
        # options here are fallback for the networks whose
        # intialization not defined in a method-specific options.py
        self.parser.add_argument('--weights_init',
                                 type=str,
                                 default='pretrained',
                                 choices=['pretrained', 'scratch'],
                                 help='train resnet-based encoders '
                                      ' from scratch or use pretrained weights'
                                )
        self.parser.add_argument('--init_mode',
                                 type=str,
                                 choices=['xavier_normal_', 
                                          'xavier_uniform_',
                                          'kaiming_uniform_',
                                          'kaiming_normal_',
                                          'normal_'],
                                 default = 'xavier_uniform_',
                                 help='the initializing method for the '
                                      'networks added to the model_to_init '
                                      'argument, that is in the options.py '
                                      'specific to each method.')
        self.parser.add_argument('--init_params',
                                 nargs='+',
                                 type=float,
                                 default=[1.0],
                                 help='parameters fed to the initializing '
                                      'method, default is gain for '
                                      'xavier_normal_ or kaiming_uniform')

        # LOAD TRAINED MODELS and FREEZE MODELS
        self.parser.add_argument('--load_weights_folder',
                                 type=str,
                                 help='path of the model to load')

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

        # DATASET PREPROCESSING
        self.parser.add_argument('--subset_ratio', # mixed datasets
                                 nargs='+',
                                 type=float,
                                 default=[1.0],
                                 help='random sample a subset from the used '
                                      ' dataset') 
        self.parser.add_argument('--png',
                                 action='store_true',
                                 help='set True if the data are in png format'
                                      ' False for jpg')
        self.parser.add_argument('--height',
                                 type=int,
                                 default=128,
                                 help='input image height, please set a '
                                      'multiple of 32')
        self.parser.add_argument('--width',
                                 type=int,
                                 default=416,
                                 help='input image width, please set a '
                                      'multiple of 32')
        self.parser.add_argument('--scales',
                                 nargs='+',
                                 type=int,
                                 default=[0],
                                 help='scales used in the loss, this affects '
                                      'both dataloader and some methods to '
                                      'compute losses')
        self.parser.add_argument('--frame_ids',
                                 nargs='+',
                                 type=int,
                                 help='frames to load',
                                 default=[0, -1, 1])

        ## POSSIBLY MOBILE MASKS
        MASK = ['none', 'mono', 'color']
        self.parser.add_argument('--seg_mask',
                                 type=str,
                                 choices=MASK,
                                 default='mono',
                                 help='whether to use segmetation mask')
        self.parser.add_argument('--MIN_OBJECT_AREA',
                                 type=int,
                                 default=20,
                                 help='size threshold to discard mobile masks'
                                      ' set as 0 to disable the size screening'
                                 )
        self.parser.add_argument('--boxify',
                                 action='store_true',
                                 help='reshape masks to bounding boxes')

        ## REMOVE MASKED OBJECTS
        self.parser.add_argument('--prob_to_mask_objects',
                                 nargs='+',
                                 type=float,
                                 default=[0.0],
                                 help='probability to remove objects '
                                      'overlapping with mobile masks.'
                                      ' set 0.0 to disable, set 1.0 to'
                                      ' objects with 100%')

        ## DATA AUGMENTATION
        self.parser.add_argument('--not_do_color_aug',
                                 help='whether to do color augmentation',
                                 action='store_true')
        self.parser.add_argument('--not_do_flip',
                                 help='whether to flip image horizontally ',
                                 action='store_true')
        self.parser.add_argument('--do_crop',
                                 help='whether to crop image height',
                                 action='store_true')
        self.parser.add_argument('--crop_bound', # mixed datasets
                                 type=float, nargs='+',
                                 help='for example, crop_bound=[0.0, 0.8]'
                                      ' means the bottom 20% of the image will'
                                      ' never be cropped. If only one value is'
                                      ' given, only the top will be cropped'
                                      ' according to the ratio',
                                 default=[0.0, 1.0])

        # OPTIMIZATION
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 help='batch size',
                                 default=12)
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 help='number of dataloader workers',
                                 default=12)
        self.parser.add_argument('--learning_rate',
                                 type=float,
                                 help='learning rate',
                                 default=1e-4)
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 help='number of epochs',
                                 default=20)
        self.parser.add_argument('--warmup_epochs',
                                 type=int,
                                 default=0,
                                 help='if nonzero, a warmup mechanism would'
                                      'enabled')
        self.parser.add_argument('--scheduler_step_size',
                                 type=int,
                                 help='step size of the scheduler',
                                 default=15)
        self.parser.add_argument('--scheduler_gamma',
                                 type=float,
                                 help='factor of learning rate decay',
                                 default=0.1)
        self.parser.add_argument('--gradient_clip_norm',
                                 type=float,
                                 default = 10,
                                 help='set 0 to disable it')
        self.parser.add_argument('--early_stop_patience',
                                 type=int,
                                 help='if set positive, early stop training when the model does not update in setting patience',
                                 default=0)

        ## Regularization - for the common networks across methods
        self.parser.add_argument('--weight_decay',
                                 type=float,
                                 default = 0,
                                 help='for networks whose corresponding '
                                      ' weight decay values are not defined '
                                      'in the model-specific options.py')

        # LOSS COMPUTATIONS options
        self.parser.add_argument('--no_ssim',
                                 action='store_true',
                                 help='if set, disables ssim in the loss')
        self.parser.add_argument('--weighted_ssim',
                                 action = 'store_true',
                                 help='if true, use weighted ssim instead of '
                                      'ssim loss')
        self.parser.add_argument('--use_weighted_l1',
                                 action = 'store_true',
                                 help='use weighted_l1 error which would not'
                                      ' include the masked out regions into '
                                      'the denominator')

        # SYSTEM options
        self.parser.add_argument('--no_cuda',
                                 help='if set disables CUDA',
                                 action='store_true')
        self.parser.add_argument('--seed',
                                 type=int,
                                 help='seed',
                                 default=9487)

        # LOGGING options
        self.parser.add_argument('--log_frequency',
                                 type=int,
                                 default=0,
                                 help='number of batches between '
                                      'each tensorboard log')
        self.parser.add_argument('--save_frequency',
                                 type=int,
                                 default=0,
                                 help='number of epochs between each save '
                                      'only save the best checkpoint if set 0')
        self.parser.add_argument('--log_multiframe',
                                 action='store_true',
                                 help='save visual results of all the frames')
        self.parser.add_argument('--log_multiscale',
                                 action='store_true',
                                 help='save visual results of all the scales')
        self.parser.add_argument('--log_depth',
                                 action='store_true',
                                 help='whether to save depths to tensorboard')

        self.parser.add_argument('--log_lr',
                                 action='store_true',
                                 help='whether to save depths to tensorboard')
        
    def parse(self):
        self.options, unknown_args = self.parser.parse_known_args()
        self.options.project_dir = project_dir
        if len(self.options.data_path) == 1:
            self.options.data_path = os.path.abspath(
                    os.path.expanduser(self.options.data_path[0])
                    )
        else:
            self.options.data_path = [os.path.abspath(
                os.path.expanduser(dp)) for dp in self.options.data_path]

        # any options expanded as list for the mixed dataset application
        # except for the data_path option
        other_opts_for_mixed_dataset = ['subset_ratio', 'prob_to_mask_objects'] 
        for opt in other_opts_for_mixed_dataset:
            if len(eval(f'self.options.{opt}')) == 1:
                exec(f'self.options.{opt}=self.options.{opt}[0]')

        if self.options.model_name == '':
            curr_t = datetime.now().strftime('%y-%b-%d-%H-%M-%S')
            self.options.model_name = curr_t
        return self.options, unknown_args
