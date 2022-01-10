# All rights reserved.
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class RefinementLayer(nn.Module):
    def __init__(self, num_channel, dims, num_motion_fields=3):
        super(RefinementLayer, self).__init__()
        self.num_channel = num_channel
        self.num_mid_channel = max(4, self.num_channel)
        self.dims = dims
        self.num_motion_fields = num_motion_fields
        # same padding by hard coded for now
        self.conv1 = nn.Conv2d(self.num_motion_fields + self.num_channel,
                               self.num_mid_channel, 3, padding=1)
        self.conv2_1 = nn.Conv2d(self.num_motion_fields + self.num_channel,
                                 self.num_mid_channel, 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.num_mid_channel, self.num_mid_channel, 3,
                                 padding=1)
        self.conv3 = nn.Conv2d(self.num_mid_channel*2,
                               self.num_motion_fields, 1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, motion_field, feature):
        # Pytorch does not support the half_pixel_center argument
        # it induces difference in the network output compared
        # to the official tensorflow repo
        upsampled_motion_field = nn.functional.interpolate(
                motion_field, self.dims, mode='bilinear')
        x = torch.cat((upsampled_motion_field, feature), dim=1)
        output1 = self.relu(self.conv1(x))
        output2 = self.relu(self.conv2_1(x))
        output2 = self.relu(self.conv2_2(output2))
        output = torch.cat((output1, output2), dim=1)
        output = upsampled_motion_field + self.conv3(output)
        return output

class MotionPredictionNet(nn.Module):
    def __init__(self, input_dims, num_input_images, bottleneck_dims):
        super(MotionPredictionNet, self).__init__()
        self.input_dims = input_dims
        self.num_input_images = num_input_images
        self.bottleneck_dims = bottleneck_dims
        self._init_motion_field_net()

    def _init_motion_field_net(self):
        self.num_ch_bottleneck = [16, 32, 64, 128, 256, 512, 1024]
        self.refinement1 = self._make_refinement_layer(
                self.num_ch_bottleneck[-1],
                self.bottleneck_dims[-1])
        self.refinement2 = self._make_refinement_layer(
                self.num_ch_bottleneck[-2],
                self.bottleneck_dims[-2])
        self.refinement3 = self._make_refinement_layer(
                self.num_ch_bottleneck[-3],
                self.bottleneck_dims[-3])
        self.refinement4 = self._make_refinement_layer(
                self.num_ch_bottleneck[-4],
                self.bottleneck_dims[-4])
        self.refinement5 = self._make_refinement_layer(
                self.num_ch_bottleneck[-5],
                self.bottleneck_dims[-5])
        self.refinement6 = self._make_refinement_layer(
                self.num_ch_bottleneck[-6],
                self.bottleneck_dims[-6])
        self.refinement7 = self._make_refinement_layer(
                self.num_ch_bottleneck[-7],
                self.bottleneck_dims[-7])
        self.final_refinement = self._make_refinement_layer(
                self.num_input_images*3, self.input_dims)
 
    def _make_refinement_layer(self, ch, dims):
        return RefinementLayer(ch, dims)

    def forward(self, translation, features):
        # TODO: check the effect of apply trans_scale on translation

        residual_translation = self.refinement1(translation, features[7])
        residual_translation = self.refinement2(
                residual_translation, features[6])
        residual_translation = self.refinement3(
                residual_translation, features[5])
        residual_translation = self.refinement4(
                residual_translation, features[4])
        residual_translation = self.refinement5(
                residual_translation, features[3])
        residual_translation = self.refinement6(
                residual_translation, features[2])
        residual_translation = self.refinement7(
                residual_translation, features[1])
        residual_translation = self.final_refinement(
                residual_translation, features[0]
                )
        return residual_translation

