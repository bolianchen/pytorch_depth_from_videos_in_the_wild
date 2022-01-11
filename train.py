# Copyright reserved.

from __future__ import absolute_import, division, print_function

import argparse
from options import WildOptions
from trainers import WildTrainer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.utils import args_validity_check

method_zoo = { 'wild': (WildOptions, WildTrainer) }

def main_worker(rank, world_size, method, gpus, dist_backend, unknown_args1):
    options = method_zoo[method][0]()
    opts, unknown_args2 = options.parse()
    args_validity_check(unknown_args1, unknown_args2)
    opts.rank = rank
    opts.world_size = world_size
    opts.gpus = gpus
    opts.dist_backend = dist_backend 
    trainer = method_zoo[method][1](opts)
    trainer.train()

if __name__ == "__main__":
    # Select METHODOLOGY
    method_initializer = argparse.ArgumentParser(description="Method Initializer")
    method_initializer.add_argument("--method",
                             type=str,
                             required = True,
                             choices = ["wild"],
                             help="depth estimation methodology to use")
    # GPU options
    method_initializer.add_argument("--gpus_to_use",
                             nargs="+",
                             type=int,
                             default=[0],
                             help="gpu(s) used for training")
    method_initializer.add_argument("--dist_backend",
                             type=str,
                             default='nccl',
                             choices = ['nccl', 'gloo'],
                             help="torch distributed built-in backends")

    args, unknown_args1 = method_initializer.parse_known_args()
    method = args.method

    world_size = len(args.gpus_to_use)
    gpus = args.gpus_to_use
    dist_backend = args.dist_backend

    if world_size > 1: # multi-gpu training
        mp.spawn(main_worker,
                 args=(world_size, method, gpus, dist_backend, unknown_args1),
                 nprocs=world_size,
                 join=True)
    else:
        main_worker(0, world_size, method, gpus, dist_backend, unknown_args1)

