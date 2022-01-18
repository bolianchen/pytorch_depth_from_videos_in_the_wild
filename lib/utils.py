# Copyright reserved.

from __future__ import absolute_import, division, print_function
import os
import sys
import time
import json
import shutil
import hashlib
import zipfile
from six.moves import urllib
import random
import math
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from matplotlib import pyplot as plt

from collections import Counter

from options import WildOptions

opts = {'wild': WildOptions}

# the first several functions: readlines, sec_to_hm, sec_to_hm_str
# are borrowed from the official Monodepth2 repository:
# https://github.com/nianticlabs/monodepth2/blob/master/utils.py  
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def same_seeds(seed):
    """Setup the random seed for modules with randomness
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_model_opt(model_path, method=None):
    """ Get the options to initialize a model evaluator
    """
    opt_path = os.path.dirname(model_path.rstrip('/')) + '/opt.json'
    if os.path.isfile(opt_path):
        # load the options used while training the model
        opt = json.load(open(opt_path))
        opt = dotdict(opt)
        opt.load_weights_folder = model_path
    else:
        # for models without the opt.json, then check the 'method' argument
        # if it is not specified, try monodepth2
        # when using the model released by the official repo
        # apply monodepth2 default options 
        if method is None or method == '':
            method = 'monodepth2'
        opt, _ = opts[method]().parse()

        # try to load checkpoints from the folder
        try:
            ckpts = list(
                filter(lambda s: s.find('.pth')!=-1, os.listdir(model_path))
                )
            assert len(ckpts) >= 4, 'not a valid checkpoint path'
            opt.load_weights_folder = model_path

        # if the above trial fails, load monodepth2 officially released models
        except:
            download_model_if_doesnt_exist(model_path)
            #TODO: refine the line below to enhance generalarity
            opt.load_weights_folder = os.path.join('models', model_path)

    if opt.method is None:
        opt.method = method

    return opt

def args_validity_check(*unknown_args):
    """ Raise an Error if any args are unrecognized by all the parsers

    unknown_args: unknown_args1, unknown_args2, ...
    """
    if len(unknown_args) == 1 and len(unknown_args[0]) > 0:
        return False

    base_unknown_args = unknown_args[0]

    for arg in base_unknown_args:
        # assume each arg contains "--"
        if '--' not in arg:
            continue
        # if an argument contained by all members in the unknown_args
        # raise error for the first detected unrecognized argument
        contained = [(arg in ua) for ua in unknown_args[1:]]
        assert not all(contained), (
        f"unrecognized argument {arg} by all parsers")
