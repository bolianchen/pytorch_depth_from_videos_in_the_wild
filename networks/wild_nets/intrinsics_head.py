# All rights reserved.
import torch
import torch.nn as nn

class IntrinsicsHead(nn.Module):
    def __init__(self):
        super(IntrinsicsHead, self).__init__()
        
        self.focal_length_conv = nn.Conv2d(1024, 2, 1, bias=False)
        self.offsets_conv = nn.Conv2d(1024, 2, 1, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, bottleneck, img_width, img_height):
        curr_device = bottleneck.device
        batch_size = bottleneck.shape[0]
        intrinsics_mat = torch.eye(4).unsqueeze(0).to(curr_device)
        intrinsics_mat = intrinsics_mat.repeat(batch_size, 1, 1)

        focal_lengths = (
                self.softplus(self.focal_length_conv(bottleneck)).squeeze() *
                torch.Tensor([img_width, img_height]).to(curr_device)
                )
        offsets = (
                (self.softplus(self.offsets_conv(bottleneck)).squeeze() + 0.5) *
                torch.Tensor([img_width, img_height]).to(curr_device)
                ).unsqueeze(-1)
        foci = torch.diag_embed(focal_lengths)

        intrinsics_mat[:,:2,:2] = foci
        intrinsics_mat[:,:2,2:3] = offsets

        return intrinsics_mat

        
