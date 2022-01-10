# All rights reserved.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv1x1
from torch.hub import load_state_dict_from_url


class WildDepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1,
                 norm_layer=None, use_norm_in_downsample=True):
        super(WildDepthEncoder, self).__init__()

        self.use_norm_in_downsample = use_norm_in_downsample

        # channels info will be passed to decoder
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.encoder = self._build_encoder(num_layers, pretrained = pretrained,
                                           norm_layer=norm_layer)
        # remove the unused avgpool and fc layers
        self.encoder.avgpool = nn.Sequential()
        self.encoder.fc = nn.Sequential()
        #self.encoder = torch.nn.Sequential(
        #        *(list(self.encoder.children())[:-2])
        #        )
                       
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def _build_encoder(self, num_layers, pretrained=False, progress=True, **kwargs):
        """ Build a Resent Encoder
        Refactor the resent and _resnet functions from
        the torchvision.models.resnet module
        """
        # information to build pretrained resnet-based encoders
        model_urls = models.resnet.model_urls
        model_ingredients = {
            'resnet18': (BasicBlock, [2, 2, 2, 2]),
            'resnet34': (BasicBlock, [3, 4, 6, 3]),
            'resnet50': (Bottleneck, [3, 4, 6, 3])
            # more can be added
        }
     
        model_name = 'resnet' + str(num_layers)
        assert model_name in model_ingredients, "{} is not a valid number of resnet layers"
        ingredients = model_ingredients[model_name]
        block, layers = ingredients[0], ingredients[1]
        
        def _resnet(arch, block, layers, pretrained, progress, **kwargs):
            model = self._wild_resnet()(block, layers, **kwargs)
            if pretrained:
                state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
                # ignore norm_layer related weights while using layernorm
                model.load_state_dict(state_dict, strict=False)
            return model

        return _resnet(model_name, block, layers, pretrained, progress,
                       **kwargs)

    def _wild_resnet(self):
        """Return a Customized ResNet for Wild"""

        use_norm_in_downsample = self.use_norm_in_downsample

        class WildResNet(ResNet):
            """Replace a portion of batchnorm with layernorm"""

            def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation 
                if dilate:                                                              
                    self.dilation *= stride 
                    stride = 1                                                          
                if stride != 1 or self.inplanes != planes * block.expansion:            
                    if use_norm_in_downsample:
                        downsample = nn.Sequential(
                                conv1x1(self.inplanes, planes * block.expansion, stride),
                                norm_layer(planes * block.expansion),
                                ) 
                    else:
                        downsample = nn.Sequential(
                                conv1x1(self.inplanes, planes * block.expansion, stride),
                                )

                layers = [] 
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                                    self.base_width, previous_dilation, norm_layer))                                
                self.inplanes = planes * block.expansion 
                # single iteration for resnet18 
                for _ in range(1, blocks): 
                    # only replace batchnorm at the last non-downsampling residual block
                    if planes == 512:
                        layers.append(block(self.inplanes, planes, groups=self.groups, 
                                            base_width=self.base_width, dilation=self.dilation, 
                                            norm_layer=norm_layer)) 
                    # use batchnorm otherwise
                    else:
                        layers.append(block(self.inplanes, planes, groups=self.groups, 
                                            base_width=self.base_width, dilation=self.dilation, 
                                            norm_layer=None))                                                           
                return nn.Sequential(*layers)       

        return WildResNet


    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
