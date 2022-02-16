import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from lib.img_processing import image_resize

class BaseEvaluator:
    def __init__(self, opt):
        self.opt = opt
        self._update_model_info()
        self.models = {}
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        # enforce using gpu if it is available
        # use export CUDA_VISIBLE_DEVICES="" to use cpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._init_depth_net()
        self._init_pose_net()
        self._load_models()
        self._set_eval()

    def _update_model_info(self):
        """ Obtain height, width and intrinsicis from the encoder checkpoint
        """

        load_weights_folder = os.path.expanduser(
                self.opt.load_weights_folder)
        
        enc_dict = torch.load(
                os.path.join(load_weights_folder, 'encoder.pth')
                )

        self.width = self.opt.width = enc_dict['width']
        self.height = self.opt.height = enc_dict['height']

        self.repr_intrinsics = enc_dict['repr_intrinsics']

    def _init_depth_net(self):
        raise NotImplementedError

    def _init_pose_net(self):
        raise NotImplementedError

    def _load_models(self):
        raise NotImplementedError

    def _set_eval(self):
        for name in self.models:
            self.models[name].eval()

    def get_training_res(self):
        return self.height, self.width

    def estimate_depth(self, img):
        raise NotImplementedError

    def estimate_pose(self, imgs):
        raise NotImplementedError

    def compute_point_cloud(self, img, specified_intrinsics=None):
        """ Compute the 3D points corresponding to the input image

        """
        if specified_intrinsics is not None:
            used_intrinsics = specified_intrinsics
        else:
            used_intrinsics = self.repr_intrinsics

        _, depth = self.estimate_depth(img)
        meshgrid = np.meshgrid(range(self.width), range(self.height),
                               indexing='xy')
        meshgrid.append(np.ones((self.height, self.width)))
        id_coords = np.stack(meshgrid, axis=0).squeeze().astype(np.float32)
        id_coords = torch.from_numpy(id_coords).to(self.device)
        inv_intrinsics = torch.from_numpy(
                np.linalg.inv(used_intrinsics).astype(np.float32)
                ).to(self.device)
        
        cloud_pts = torch.einsum('ij,jhw,hw->ihw',
                                 inv_intrinsics,
                                 id_coords, depth[0,0,:,:])
        return cloud_pts.detach().cpu().numpy()

    def _color_disp(self, disp):
        disp_resized_np = disp.squeeze().detach().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (
                mapper.to_rgba(
                    disp_resized_np)[:, :, :3] * 255
                ).astype(np.uint8)
        return colormapped_im

