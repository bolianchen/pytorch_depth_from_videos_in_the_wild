# Copyright  reserved.
# FUNCTIONS and CLASSES specific for DEPTH FROM VIDEOS IN THE WILD
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

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

def rot_from_euler(vec, expand=True):
    """Convert an Euler angle representation to a rotation matrix.

    refer to the implementation in tensorflow-graphics
    """
    ca = torch.cos(vec)
    sa = torch.sin(vec)

    sx, sy, sz = torch.unbind(sa, axis=-1)
    cx, cy, cz = torch.unbind(ca, axis=-1)
    
    if expand:
        rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)
    else:
        rot = torch.zeros((vec.shape[0], 3, 3)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(cy*cz)
    rot[:, 0, 1] = torch.squeeze((sx * sy * cz) - (cx * sz))
    rot[:, 0, 2] = torch.squeeze((cx * sy * cz) + (sx * sz))
    rot[:, 1, 0] = torch.squeeze(cy * sz)
    rot[:, 1, 1] = torch.squeeze((sx * sy * sz) + (cx * cz))
    rot[:, 1, 2] = torch.squeeze((cx * sy * sz) - (sx * cz))
    rot[:, 2, 0] = torch.squeeze(-sy)
    rot[:, 2, 1] = torch.squeeze(sx * cy)
    rot[:, 2, 2] = torch.squeeze(cx * cy)
    if expand:
        rot[:, 3, 3] = 1

    return rot

def matrix_from_angles(axisangle):
    R = rot_from_euler(-axisangle, expand=False)
    return R.transpose(-2,-1)

def compute_projected_rotation(axisangle, K, inv_K):
    """
    """
    R = rot_from_euler(-axisangle)
    R = R.transpose(-2,-1)
    return torch.einsum('bij,bjk,bkl->bil', K, R, inv_K)

def compute_projected_translation(translation, K):
    """
    """
    return torch.einsum('bij,bjhw->bihw', K[:,:3,:3], translation)

def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
  r2r1 = torch.matmul(rot_mat2, rot_mat1)
  r2t1 = torch.matmul(rot_mat2, trans_vec1.unsqueeze(-1))
  r2t1 = r2t1.squeeze(-1)
  return r2r1, r2t1 + trans_vec2

def compute_projected_pixcoords(proj_rot, proj_trans, depth):
    """
    equivalent to the transform_depth_map._using_motion_vector function
    """
    batch_size = depth.shape[0]
    height, width = depth.shape[-2:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    meshgrid.append(np.ones((height, width)))
    id_coords = np.stack(meshgrid, axis=0).squeeze().astype(np.float32)
    id_coords = torch.from_numpy(id_coords).to(depth.device)
            
    # (bs, 3, h, w)
    proj_coords = torch.einsum('bij,jhw,bhw->bihw',
                               proj_rot[:,:3,:3],
                               id_coords, depth[:,0,:,:])
    proj_coords = proj_coords + proj_trans
    # each has shape (bs, h, w)
    x, y, z = torch.unbind(proj_coords, axis = 1)
    pixel_x, pixel_y = x/z, y/z
    coords_not_underflow = torch.logical_and(pixel_x>=0.0, pixel_y>=0.0)
    coords_not_overflow = torch.logical_and(pixel_x<=width-1,
                                            pixel_y<=height-1)
    z_positive = z > 0

    coords_not_nan = torch.logical_not(
            torch.logical_or(torch.isnan(x), torch.isnan(y))
            )

    not_nan_mask = coords_not_nan.float()
    pixel_x *= not_nan_mask
    pixel_y *= not_nan_mask
    mask = (coords_not_underflow & coords_not_overflow & 
            coords_not_nan & z_positive)
    
    # clamp
    pixel_x = pixel_x.clamp(min=0.0, max=width-1)
    pixel_y = pixel_y.clamp(min=0.0, max=height-1)
    # normalize
    pixel_x /= width - 1
    pixel_y /= height - 1
    pixel_x = (pixel_x - 0.5) * 2
    pixel_y = (pixel_y - 0.5) * 2

    pix_coords = torch.cat(
            [pixel_x.unsqueeze(-1), pixel_y.unsqueeze(-1)], dim =3)

    return pix_coords, z.unsqueeze(1), mask.unsqueeze(1)

def get_motion_smooth_loss(motion_map):
    """Compute the smoothness of rotations and translations"""
    norm = 3.0 * torch.mean(torch.square(motion_map),
                            dim=[1, 2, 3], keepdim=True)
    motion_map = torch.div(motion_map, torch.sqrt(norm + 1e-12))
    return _motion_smooth_helper(motion_map)

def _motion_smooth_helper(motion_map):
    """Calculates L1 (total variation) smoothness loss of a tensor.
    Args:
      motion_map: A tensor to be smoothed, of shape [B, C, H, W].
    Returns:
      A scalar tensor, the total variation loss.
    """
    motion_map_dx = motion_map - torch.roll(motion_map, 1, 2)
    motion_map_dy = motion_map - torch.roll(motion_map, 1, 3)
    sm_loss = torch.sqrt(1e-24 + torch.square(motion_map_dx) + torch.square(motion_map_dy))
    return sm_loss.mean()

class WeightedAvgPool3x3(nn.Module):
    def __init__(self, weight_epsilon):
      super(WeightedAvgPool3x3, self).__init__()
      # kernel_size = 3; stride=1
      self.avg_pool = nn.AvgPool2d(3,stride=1)
      self.weight_epsilon = weight_epsilon
    def forward(self, x, weight):
      average_pooled_weight = self.avg_pool(weight)
      weight_plus_epsilon = weight + self.weight_epsilon
      inverse_average_pooled_weight = 1.0/(average_pooled_weight + self.weight_epsilon)
      weight_avg = self.avg_pool(x * weight_plus_epsilon)
      return weight_avg * inverse_average_pooled_weight

class WeightedSSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, c1=0.01**2, c2=0.03**2, weight_epsilon=0.01):
        super(WeightedSSIM, self).__init__()
        self.weight_pool = nn.AvgPool2d(3,1)
        self.weighted_avg_pool3x3 = WeightedAvgPool3x3(weight_epsilon)
        self.C1 = c1
        self.C2 = c2
        self.weight_epsilon = weight_epsilon

    def forward(self, x, y, weight):
        mu_x = self.weighted_avg_pool3x3(x, weight)
        mu_y = self.weighted_avg_pool3x3(y, weight)
        sigma_x = self.weighted_avg_pool3x3(x**2, weight) - mu_x**2
        sigma_y = self.weighted_avg_pool3x3(y**2, weight) - mu_y**2
        sigma_xy = self.weighted_avg_pool3x3(x*y, weight) - mu_x*mu_y
        average_pooled_weight = self.weight_pool(weight)
       
        if self.C1 == float('inf'):
            ssim_n = (2 * sigma_xy + self.C2)
            ssim_d = (sigma_x + sigma_y + self.C2)
        elif self.C2 == float('inf'):
            ssim_n = 2 * mu_x * mu_y + self.C1
            ssim_d = mu_x ** 2 + mu_y ** 2 + self.C1
        else:
            ssim_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
            ssim_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - ssim_n / ssim_d) / 2, 0, 1), average_pooled_weight

def make_randomized_layernorm(noise_rampup_steps=10000):
    """Return a RandomizedLayerNorm class
    it is customized by the specifed noise_rampup_steps
    """

    class RandomizedLayerNorm(nn.Module):
        def __init__(self, num_features, affine=True):
            super(RandomizedLayerNorm, self).__init__()
            self.beta = torch.nn.Parameter(torch.zeros(num_features), requires_grad=affine)
            self.gamma = torch.nn.Parameter(torch.ones(num_features), requires_grad=affine)

            ## the difference between 1.0 and the next smallest machine
            ## representable float
            self.epsilon = torch.finfo(torch.float32).eps
            self.step = 0

        def __truncated_normal(self, shape=(), mean=0.0, stddev=1.0):
            """
            # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
            refactor loicsacre's solution
            easy to understand, in gpu
            likely to be inefficient
            """
            t = torch.nn.init.normal_(torch.ones(shape, device=self.beta.device),
                                      mean=mean, std=stddev)
            while True:
                cond = torch.logical_or(t < mean - 2*stddev, t > mean + 2*stddev)
                if not torch.sum(cond):
                    break
                t = torch.where(cond,
                        torch.nn.init.normal_(torch.ones(shape, device=self.beta.device),
                        mean=mean, std=stddev), t)
            return t

        def _truncated_normal(self, shape=(), mean=0.0, stddev=1.0):
            """
            # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/20
            refactor heiner's solution
            TODO: check the correctness of the function: remap uniform to standard normal
            """
            uniform = torch.rand(shape, device=self.beta.device)

            def parameterized_truncated_normal(uniform, mean, stddev, lower_level=-2, upper_level=2):

                # stardard normal
                normal = torch.distributions.normal.Normal(0, 1)

                lower_normal_cdf = normal.cdf(lower_level)
                upper_normal_cdf = normal.cdf(upper_level)
                
                p = lower_normal_cdf + (upper_normal_cdf - lower_normal_cdf) * uniform

                ## clamp the values out of the range to the edge values
                v = torch.clamp(2 * p - 1, -1.0 + self.epsilon, 1.0 - self.epsilon)
                x = mean + stddev * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(v)

                return x

            return parameterized_truncated_normal(uniform, mean, stddev)

        def forward(self, x):
            mean = x.mean((2,3), keepdim=True)
            variance = torch.square(x - mean).mean((2,3), keepdim=True)
            if noise_rampup_steps <= 0:
                stddev = 0.5
            else:
                stddev = 0.5 * pow(min(self.step/noise_rampup_steps, 1.0), 2)
            if self.training:
                mean = torch.mul(
                        mean,
                        1.0 + self._truncated_normal(mean.shape, stddev=stddev)
                        )
                variance = torch.mul(
                        variance,
                        1.0 + self._truncated_normal(variance.shape, stddev=stddev)
                        )
            outputs = (self.gamma.view(1,-1,1,1)
                       * torch.div(x - mean, torch.sqrt(variance) + 1e-3)
                       + self.beta.view(1,-1,1,1))
            self.step += 1
            return outputs

    return RandomizedLayerNorm

def select_weight_initializer(mode, *args, same_for_bias=False):
    """Return a weight initializer
    """

    def initialize_weights(m):

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # initialize the weights
            eval(f'nn.init.{mode}')(m.weight.data, *args)
            if m.bias is not None:
                # whether to apply the same initialization to biases
                if same_for_bias:
                    eval(f'nn.init.{mode}')(m.bias.data, *args)
                else:
                    nn.init.constant_(m.bias.data, 0)

        # the same with the default initialization
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    return initialize_weights

def weighted_average(x, weights, epsilon=1.0):
    """Compute weighted average of x by weights
    Args:
        epsilon is added to denominator to prevent overflow
    """
    weighted_sum = torch.sum(torch.mul(x, weights), dim=(2,3), keepdim=True)
    sum_of_weights = torch.sum(weights, axis=(2,3), keepdim=True)
    return torch.div(weighted_sum, sum_of_weights + epsilon)

def l1_error(resampled_target, source, mask=None):
    """Compute l1 error"""
    assert resampled_target.shape == source.shape

    if mask is None:
        error = torch.abs(resampled_target-source)
    else:
        assert resampled_target[:,0:1,:,:].shape == mask.shape
        error = torch.abs(resampled_target-source) * mask
    return torch.mean(error)

def weighted_l1_error(resampled_target, source, mask=None, epsilon=1.0):
    """Compute weighted l1 error"""
    assert resampled_target.shape == source.shape
    if mask is None:
        error = torch.abs(resampled_target-source)
        return torch.mean(error)
    else:
        assert resampled_target[:,0:1,:,:].shape == mask.shape
        error = torch.abs(resampled_target-source)
        weighted_error_sum = torch.sum(
                torch.mul(error, mask), axis=(2,3), keepdim=True)
        sum_of_weights = torch.sum(mask, axis=(2,3), keepdim=True)
        result = torch.div(weighted_error_sum,
                         sum_of_weights + epsilon).mean()
        return result

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

# Borrowed and Revised from:
# https://github.com/ildoonet/pytorch-gradual-warmup-lr
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.

    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_iters: target learning rate is reached at total_iters, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iters, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_iters = total_iters
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """Get lr to update the wrapped optimizer if not self.finished"""
        if self.last_epoch >= self.total_iters:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                            base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.base_lrs
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * ((self.last_epoch+1) / self.total_iters)
                    for base_lr in self.base_lrs]
        else: # not checked
            return [base_lr * ((self.multiplier-1) * (self.last_epoch+1)
                / self.total_iters + 1) for base_lr in self.base_lrs]

    # not checked
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_iters:
            warmup_lr = [base_lr * ((self.multiplier - 1) * self.last_epoch
                / self.total_iters + 1) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_iters)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_iters)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
