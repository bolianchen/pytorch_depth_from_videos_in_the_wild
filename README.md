# (PyTorch) Depth from Videos in the Wild

## Introduction to the Project

This project is a Pytorch re-implementation of the following Google ICCV 2019 paper:  
**[Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998)**

<p align="center">
  <img src="demo/kitti_0926drive0001_0018.gif" width="600" />
</p>

## Data Preparation

We provide `gen_data.py` to generate training datasets. There are two types of datasets available:

<details><summary><strong>KITTI</strong></summary>
<p>  
  
#### Download Raw Data   
Please visit the [official website](http://www.cvlibs.net/datasets/kitti/raw_data.php) to download the entire raw KITTI dataset 
and unzip it to a folder named kitti_raw.  
Alternatively, you can also run the follows:
```
./datasets/data_prep/kitti_raw_downloader.sh
```
#### Generate Training Dataset   
```
python gen_data.py \
--dataset_name [kitti_raw_eigen or kitti_raw_stereo] \
--dataset_dir /path/to/kitti_raw \
--save_dir /path/to/save/the/generated_data \
--mask color
```

</p>
</details>

<details><summary><strong>VIDEOs</strong></summary>
<p>  

Training datasets can also be generated from your own videos under the same folder.   
  
*[Optional]* If the camera intrinsics are known, please put the 9 entries of its flattented camera intrinsics in a text file.
```
1344.8 0.0 640.0 0.0 1344.8 360.0 0.0 0.0 1.0
```

  
Then generate the training dataset by running:
```
python gen_data.py \
--dataset_name video \
--dataset_dir /path/to/your/video_folder \
--save_dir /path/to/save/the/generated_data \
--intrinsics /path/to/your/camera_intrinsics_file \ # None if not set, default intrinsics are produced according to IPhone 
--mask color
```
  
</p>
</details>

More options for `gen_data.py` can be found in `options/gen_data_options.py`.

## Train Models

## Run Inference

## Reference

