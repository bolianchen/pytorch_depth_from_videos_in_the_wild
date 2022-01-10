# All rights reserved.
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class RotTransScaler(nn.Module):
    def __init__(self):
        super(RotTransScaler, self).__init__()
        self.rot_scale = torch.nn.Parameter(torch.tensor(0.01), requires_grad=True)
        self.trans_scale = torch.nn.Parameter(torch.tensor(0.01), requires_grad=True)
    def forward(self, x, rot_or_trans):
        if rot_or_trans == 'rot':
            return x * self.rot_scale
        elif rot_or_trans == 'trans':
            return x * self.trans_scale

class PosePredictionNet(nn.Module):
    def __init__(self, input_dims, num_input_images=2):
        super(PosePredictionNet, self).__init__()
        self.input_dims = input_dims
        self.num_input_images = num_input_images
        self._init_bottleneck()
        self.background_motion_conv = nn.Conv2d(1024, 6, 1, bias=False)

        # TODO: learning intrinsics
        # TODO: model parameter initialization

    def _init_bottleneck(self, padding='same'):

        self.btk_kernel_size = 3
        self.btk_stride = 2
        assert self.btk_kernel_size%2 != 0
        self.bottleneck_dims = []
        self.pads = nn.ModuleDict()
        h, w = self.input_dims
        # hard coded for kernel size = 3 and stride = 2
        # TODO: make the codes general, minor
        for i in range(7):
            if padding == 'valid':
                self.pads[f'pad{i+1}'] = nn.Identity()
                h = (h-self.btk_kernel_size)//self.btk_stride + 1
                w = (w-self.btk_kernel_size)//self.btk_stride + 1
                self.bottleneck_dims.append((h,w))
            elif padding == 'same':
                self.pads[f'pad{i+1}'] = nn.ConstantPad2d((w%2,1, h%2, 1), 0)
                pad = 2*(self.btk_kernel_size//2)
                h = (h-self.btk_kernel_size+pad)//self.btk_stride + 1
                w = (w-self.btk_kernel_size+pad)//self.btk_stride + 1
                self.bottleneck_dims.append((h,w))

        # conv1 ~ conv7
        # convolutions + relus, no normalizations
        self.conv1 = nn.Conv2d(self.num_input_images * 3, 16,
                               self.btk_kernel_size, stride = self.btk_stride)
        self.conv2 = nn.Conv2d(16, 32, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.conv3 = nn.Conv2d(32, 64, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.conv4 = nn.Conv2d(64, 128, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.conv5 = nn.Conv2d(128, 256, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.conv6 = nn.Conv2d(256, 512, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.conv7 = nn.Conv2d(512, 1024, self.btk_kernel_size,
                               stride = self.btk_stride)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def _init_intrinsics_head(self):
        pass

    def forward(self, input_images):
        # TODO: two modes
        self.features = []
        self.features.append(input_images)
        self.features.append(
                self.relu(self.conv1(
                    self.pads['pad1'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv2(
                    self.pads['pad2'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv3(
                    self.pads['pad3'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv4(
                    self.pads['pad4'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv5(
                    self.pads['pad5'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv6(
                    self.pads['pad6'](self.features[-1])
                    ))
                )
        self.features.append(
                self.relu(self.conv7(
                    self.pads['pad7'](self.features[-1])
                    ))
                )
        bottleneck = self.global_pooling(self.features[-1])

        background_motion = self.background_motion_conv(bottleneck)
        rotation = background_motion[:,:3, 0, 0]
        translation = background_motion[:,3:, :, :]

        return rotation, translation, self.features

