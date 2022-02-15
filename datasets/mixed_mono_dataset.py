# Copyright reserved.

from __future__ import absolute_import, division, print_function

import torch.utils.data as data
from .custom_mono_dataset import CustomMonoDataset

def which_dataset(frame_index, sample_nums):
    for dataset_idx, num in enumerate(sample_nums):
        frame_index -= num
        if frame_index < 0:
            sub_frame_idx = frame_index + num
            return dataset_idx, sub_frame_idx
    print('invalid frame_index is given')

class MixedMonoDataset(data.Dataset):
    """
    """
    def __init__(self,
                 data_path, ##
                 filenames, ##
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 **kwargs):
        super(MixedMonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames

        self.num_datasets = len(self.data_path)
        self.sample_nums = [len(fs) for fs in self.filenames]

        self.datasets = []
        self._proc_kwarg(kwargs)

        for idx, dpth in enumerate(self.data_path):
            sub_kwargs = {k:v[idx] for k,v in kwargs.items() if k!='crop_bound'}
            sub_kwargs['crop_bound'] = kwargs['crop_bound'][2*idx:2*(idx+1)] 
            self.datasets.append(
                    CustomMonoDataset(dpth, self.filenames[idx], height, width,
                                  frame_idxs, num_scales, **sub_kwargs)
                    )
    def _proc_kwarg(self, kwargs):
        """
        """
        for k, v in kwargs.items():
            if k == 'crop_bound':
                if len(v) == 2:
                    kwargs[k] = v * self.num_datasets
                else:
                    assert len(v)%2 == 0
                    assert len(v)//2 == self.num_datasets
            else:
                if isinstance(v, list):
                    assert len(v) == self.num_datasets
                else:
                    kwargs[k] = [v] * self.num_datasets
                    
    def __len__(self):
        data_length=0
        for fs in self.filenames:
            data_length += len(fs)
        return data_length

    def __getitem__(self, index):
        dataset_idx, sub_frame_idx = which_dataset(index, self.sample_nums)
        return self.datasets[dataset_idx].__getitem__(sub_frame_idx)

    def get_repr_intrinsics(self):
        """Return the effective intrinsics from the first dataset"""
        return self.datasets[0].get_repr_intrinsics()
        
