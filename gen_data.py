# Copyright All Rights Reserved.

"""Generates data for training/validation and save it to disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import multiprocessing
import os
from absl import app
from absl import flags
from absl import logging
from torch.utils import data
import numpy as np
import imageio
import torch
import torchvision
from tqdm import tqdm

from options import DataGenOptions
from datasets import ProcessedImageFolder
from datasets.data_prep.video_loader import Video
from datasets.data_prep.kitti_loader import KittiRaw

FLAGS = DataGenOptions().parse()

NUM_CHUNKS = 100

def _generate_data():
    r"""
    Extract sequences from dataset_dir and store them in save_dir.
    """
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    global dataloader  # pylint: disable=global-variable-undefined
    if FLAGS.dataset_name == 'video':
        dataloader = Video(
            FLAGS.dataset_dir,
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask,
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold,
            intrinsics=FLAGS.intrinsics,
            trim=FLAGS.trim,
            crop=FLAGS.crop,
            del_static_frames=FLAGS.del_static_frames,
            augment_strategy=FLAGS.augment_strategy,
            augment_shift_h=FLAGS.augment_shift_h,
            fps=FLAGS.fps,
            video_start=FLAGS.video_start.seconds,
            video_end=FLAGS.video_end.seconds,
            img_ext=FLAGS.save_img_ext
        )
    elif FLAGS.dataset_name == 'kitti_raw_eigen':
        dataloader = KittiRaw(
            FLAGS.dataset_dir,
            split='eigen',
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask, 
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold
      )
    elif FLAGS.dataset_name == 'kitti_raw_stereo':
        dataloader = KittiRaw(
            FLAGS.dataset_dir,
            split='stereo',
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask, 
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold
      )
    else:
        raise ValueError('Unknown dataset')

    all_frames = range(dataloader.num_train)
    # Split into training/validation sets. Fixed seed for repeatability.
    np.random.seed(8964)

    num_cores = multiprocessing.cpu_count()
    # number of processes while using multiple processes
    # number of workers for using either a single or multiple processes
    num_threads = num_cores if FLAGS.num_threads is None else FLAGS.num_threads

    if FLAGS.single_process:
        frame_chunks = list(all_frames)
    else:
        frame_chunks = np.array_split(all_frames, NUM_CHUNKS)
        manager = multiprocessing.Manager()
        all_examples = manager.dict()
        pool = multiprocessing.Pool(num_threads)

    with open(os.path.join(FLAGS.save_dir, 'train_files.txt'), 'w') as train_f:
        with open(os.path.join(FLAGS.save_dir, 'val_files.txt'), 'w') as val_f:
            logging.info('Generating data...')

            for index, frame_chunk in enumerate(frame_chunks):
                if FLAGS.single_process:
                    all_examples = _gen_example(frame_chunk, {})
                    if all_examples is None:
                        continue
                else:
                    all_examples.clear()
                    pool.map(
                        _gen_example_star,
                        zip(frame_chunk, itertools.repeat(all_examples))
                    )
                    logging.info(
                        'Chunk %d/%d: saving %s entries...', 
                        index + 1, NUM_CHUNKS, len(all_examples)
                    )
                for _, example in all_examples.items():
                    if example:
                        s = example['folder_name']
                        frame = example['file_name']
                        if np.random.random() < 0.1:
                            val_f.write('%s %s\n' % (s, frame))
                        else:
                            train_f.write('%s %s\n' % (s, frame))

    if not FLAGS.single_process:
        pool.close()
        pool.join()

    if FLAGS.mask != 'none':
        # Collect filenames of all processed images
        img_dataset = ProcessedImageFolder(FLAGS.save_dir,
                                           FLAGS.save_img_ext)
        img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=num_threads
        )

        # Generate masks by batch
        logging.info('Generating masks...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for imgs, img_filepaths in tqdm(img_loader):
            mrcnn_results = dataloader.run_mrcnn_model(imgs.to(device))
            for i in range(len(imgs)):
                _gen_mask(mrcnn_results[i], img_filepaths[i], FLAGS.save_img_ext)

    if FLAGS.dataset_name=='video' and FLAGS.delete_temp:
        dataloader.delete_temp_images()
  
def _gen_example(i, all_examples=None):
    r"""
    Save one example to file.  Also adds it to all_examples dict.
    """
    add_to_file, example = dataloader.get_example_with_index(i)
    if not example or dataloader.is_bad_sample(i):
        return
    image_seq_stack = _stack_image_seq(example['image_seq'])
    example.pop('image_seq', None)  # Free up memory.
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    save_dir = os.path.join(FLAGS.save_dir, example['folder_name'])
    os.makedirs(save_dir, exist_ok=True)
    img_filepath = os.path.join(save_dir, f'{example["file_name"]}.{FLAGS.save_img_ext}')
    imageio.imsave(img_filepath, image_seq_stack.astype(np.uint8))
    cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
    example['cam'] = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy)
    with open(cam_filepath, 'w') as cam_f:
        cam_f.write(example['cam'])

    if not add_to_file:
        return

    key = example['folder_name'] + '_' + example['file_name']
    all_examples[key] = example
    return all_examples

def _gen_example_star(params):
    return _gen_example(*params)

def _gen_mask(mrcnn_result, img_filepath, save_img_ext):
    f"""
    Generate a mask and save it to file.
    """
    mask_img = dataloader.generate_mask(mrcnn_result)
    mask_filepath = img_filepath[:-(len(save_img_ext)+1)] + f'-fseg.{save_img_ext}'
    imageio.imsave(mask_filepath, mask_img.astype(np.uint8))

def _gen_mask_star(params):
    return _gen_mask(*params)

def _stack_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


if __name__ == '__main__':
    _generate_data()
