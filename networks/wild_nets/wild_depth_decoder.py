# All rights reserved.
from __future__ import absolute_import, division, print_function

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn

import collections
from collections import OrderedDict

from networks.building_blocks import upsample, Conv3x3

class WildDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 use_skips=True, use_mono2_arch=True):
        super(WildDepthDecoder, self).__init__()

        if use_mono2_arch:
            from networks.building_blocks import ConvBlock as UpConv0
            from networks.building_blocks import ConvBlock as UpConv1
            self.upsample_mode = 'nearest'
        else: # apply wild architecture
            from networks.building_blocks import TransposeConv3x3 as UpConv0
            from networks.building_blocks import WildConvBlock as UpConv1
            self.upsample_mode = 'identity'

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.depth_decoder = nn.ModuleDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.depth_decoder[f'upconv_{i}_0'] = UpConv0(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.depth_decoder[f'upconv_{i}_1'] = UpConv1(num_ch_in, num_ch_out)

        for s in self.scales:
            self.depth_decoder[f'depthconv_{s}'] = nn.Sequential(
                    Conv3x3(self.num_ch_dec[s], self.num_output_channels),
                    nn.Softplus()
                    )

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.depth_decoder[f'upconv_{i}_0'](x)[:,:,:-1,:-1]
            x = [upsample(x, mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.depth_decoder[f'upconv_{i}_1'](x)
            if i in self.scales:
                self.outputs[("depth", i)] = self.depth_decoder[f'depthconv_{i}'](x)

        return self.outputs
