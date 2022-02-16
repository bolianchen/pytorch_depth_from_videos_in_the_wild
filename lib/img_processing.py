import os
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def normalize_trans(x):
    """Rescale translation 

    if all values are positive, rescale the max to 1.0
    otherwise, make sure the zeros be mapped to 0.5, and
    either the max mapped to 1.0 or the min mapped to 0
    
    """
    # do not add the following to the computation graph
    x = x.detach()

    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)

    assert ma != 0 or mi != 0
    
    d = max(abs(ma), abs(mi))
    x[x>=0] = 0.5 + 0.5 * x[x>=0]/d
    x[x<0] = 0.5 + 0.5 * x[x<0]/d

    return x

def image_resize(image, target_h, target_w, shift_h, shift_w,
                 inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # get the raw image size

    is_pil = isinstance(image, Image.Image)

    if is_pil:
        image = np.array(image)

    (raw_h, raw_w) = image.shape[:2]

    assert raw_h >= target_h, 'must be downscaling'
    assert raw_w >= target_w, 'must be downscaling'

    if target_h/raw_h <= target_w/raw_w:
        # calculate the ratio of the width and construct the dimensions
        r = target_w / float(raw_w)
        dim = (target_w, int(raw_h * r))

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]
        
        start = int(new_h*shift_h)
        end = start + target_h
       
        assert start >=0
        assert end <= new_h

        if len(image.shape) == 3:
            image = image[start:end,:,:]
        else:
            image = image[start:end,:]

        delta_u = 0
        delta_v = start  

    else: 
        # calculate the ratio of the height and construct the dimensions
        r = target_h / float(raw_h)
        dim = (int(raw_w * r), target_h)

        # downscale the image
        image = cv2.resize(image, dim, interpolation = inter)
        (new_h, new_w) = image.shape[:2]

        start = int(new_w*shift_w)
        end = start + target_w
        image = image[:,start:end,:]

        assert start >=0
        assert end <= new_w

        if len(image.shape) == 3:
            image = image[:,start:end,:]
        else:
            image = image[:,start:end]

        delta_u = start
        delta_v = 0

    if is_pil:
        image = Image.fromarray(image)

    return image, r, delta_u, delta_v


class ImageProcessor:
    """ Process images as model input
    """
    def __init__(self, opts, height, width, cam_intrinsics = None):
        
        # acquire info to cut or crop images
        self.shift_h = opts.shift_h
        self.shift_w = opts.shift_w

        # the resolution used to train the model
        self.target_h = height
        self.target_w = width

        # the camera intrinsics used to record the input images to self.process
        self.cam_intrinsics = cam_intrinsics

    def process(self, img):

        self.cam_img_h, self.cam_img_w, _ = img.shape

        if (self.target_h != self.cam_img_h or self.target_w != self.cam_img_w):
            img, ratio, du, dv = image_resize(img, self.target_h,
                                              self.target_w, self.shift_h,
                                              self.shift_w)
        else:
            # in the image_resize function, a grayscale input would be squeeze
            # due to the call of cv2.resize on the input
            _, _, c = img.shape
            if c == 1:
                img = img[:,:,0]

            ratio, du, dv = 1.0, 0, 0

        if self.cam_intrinsics is not None:
            self.cam_intrinsics = self._adjust_intrinsics(
                    self.cam_intrinsics, ratio, du, dv
                    )
        return img

    def _adjust_intrinsics(self, intrinsics, ratio, du, dv):
        intrinsics = np.copy(intrinsics)
        intrinsics[:2] *= ratio
        intrinsics[0,2] -= du
        intrinsics[1,2] -= dv
        return intrinsics

    @staticmethod
    def cut_img(img, cut, cut_h=720, cut_w=1280):
        """Crop out H720 x W1280 of a image
        This Function is to crop out a portion out of the input frame. 
        Since there is not following adjustment of intrinsics, it should only be 
        applied when the frame is composed of concatenation of images from 
        different camera.
        """
        if cut == '':
            return img

        if 'up' in cut:
            h_start, h_end = 0, cut_h
        elif 'down' in cut:
            h_start, h_end = cut_h, 2*cut_h

        if 'left' in cut:
            w_start, w_end = 0, cut_w
        elif 'right' in cut:
            w_start, w_end = cut_w, 2*cut_w

        return img[h_start:h_end, w_start:w_end, :]
        
def make_boxifier(margin=1/40):
    h_start, h_end = 0, 0
    w_start, w_end = 0, 0
    def boxifier(img):
        nonlocal h_start, h_end, w_start, w_end
        if h_end > h_start and w_start > w_end:
            return img[h_start:h_end, w_start:w_end]
        h, w, _ = img.shape
        binary_h = np.all(img==1, axis=1)
        binary_h = np.all(binary_h, axis=1)
        h_start = binary_h.argmin()
        h_end = -binary_h[::-1].argmin()
        h_start = max(0, h_start-int(h*margin))
        h_end = min(-1, h_end+int(h*margin))
        binary_w = np.all(img==1, axis=0)
        binary_w = np.all(binary_w, axis=1)
        w_start = binary_w.argmin()
        w_end = -binary_w[::-1].argmin()
        w_start = max(0, w_start-int(w*margin))
        w_end = min(-1, w_end+int(w*margin))
        return img[h_start:h_end, w_start:w_end]
    return boxifier

def concat_depth_img(disp_colormap, img, shift_h=0.0, darkening=32):
    """Concatenate errormap, dephtmap and the camera img"""
    i_h, i_w, _ = img.shape
    resized_maps = []
    h, w, _ = disp_colormap.shape
    target_h = int(h * i_w/w)
    resized_maps.append(cv2.resize(disp_colormap, (i_w, target_h)))
    img = img.astype(np.int16)
    img[:int(shift_h*i_h), :] -= darkening
    img[int(shift_h*i_h)+target_h:, :] -= darkening
    img = img.clip(min=0)
    img = img.astype(np.uint8)
    img_depth = np.concatenate(resized_maps + [img], axis=0)
    return img_depth
