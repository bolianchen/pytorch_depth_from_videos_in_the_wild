# (PyTorch) Depth from Videos in the Wild

## Introduction to the Project

This project is a Pytorch re-implementation of the following Google ICCV 2019 paper:  
**[Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998)**

## Data

We provide `gen_data.py` to generate training datasets. There are two types of datasets available:

#### KITTI
Please visit the [official website](http://www.cvlibs.net/datasets/kitti/raw_data.php) to download the entire raw KITTI dataset.
To unzip the data, run:
```
cd kitti_data
unzip "*.zip"
cd ..
```
Then generate the training dataset by running:
```
python gen_data.py \
--dataset_name [kitti_raw_eigen or kitti_raw_stereo] \
--dataset_dir /path/to/raw_kitti_data \
--save_dir /path/to/save/the/generated_data \
--mask color
```

#### VIDEO 

Training datasets can also be generated from videos under the same folder. 
Please first prepare a file that contains the 9 entries of your flattented camera intrinsics. 
For example, the file might look like:
```
1344.8 0.0 640.0 0.0 1344.8 360.0 0.0 0.0 1.0
```
Then generate the training dataset by running:
```
python gen_data.py \
--dataset_name video \
--dataset_dir /path/to/your/video_folder \
--save_dir /path/to/save/the/generated_data \
--intrinsics /path/to/your/camera_intrinsics_file \
--mask color
```

More options for `gen_data.py` can be found in `options/gen_data_options.py`.
