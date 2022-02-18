import os
import time
import json
import shutil
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# parallelism
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import datasets
from datasets.auxiliary_datasets import ImageDataset
from lib.utils import sec_to_hm_str, same_seeds, readlines
from lib.torch_layers import select_weight_initializer, BackprojectDepth, \
                             SSIM, WeightedSSIM, l1_error, \
                             weighted_l1_error

class BaseTrainer:
    def __init__(self, options):
        self.opt = options
        self.distributed = self.opt.world_size > 1

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self._preliminary_setups()

        if self.opt.rank == 0:
            self._init_writers()

        if not hasattr(self, 'frames_to_load'):
            self.frames_to_load = self.opt.frame_ids.copy()
        if not hasattr(self, 'freeze_epoch'):
            self.freeze_epoch = -1

        self.models = {}
        self._init_depth_net()
        self._init_pose_net()
        self._init_models()

        if self.opt.load_weights_folder is not None:
            self.load_model()

        # multi_gpu wrapping
        if self.distributed:
            self._prep_distributed()

        self._prep_data_lists()
        self._init_dataloaders()

        self._init_optimizer()

        num_train_samples = len(self.train_filenames)
        steps_per_epoch = num_train_samples//self.opt.batch_size
        self.num_total_steps = steps_per_epoch * self.opt.num_epochs

        self._init_backproject_depth() 

        # add an option
        if self.opt.use_weighted_l1:
            self._compute_l1_error = weighted_l1_error
        else:
            self._compute_l1_error = l1_error

        if not self.opt.no_ssim: 
            self._init_ssim()

        if self.opt.rank == 0:
            print("There are {:d} training items and {:d} validation items\n".format(
                len(self.train_filenames), len(self.val_filenames)))
        
        if self.opt.rank == 0:
            self.save_opts()

        if self.distributed:
            dist.barrier()

    def _preliminary_setups(self):

        # check whether model exists
        if os.path.exists(self.log_path):
            if self.opt.rank == 0:
                if self.opt.overwrite_old:
                    shutil.rmtree(self.log_path)
                elif not self.distributed:
                    if self.opt.warmup_epochs > 0:
                        print("Starting post-warmup training")
                    else:
                        inp = input(f">> Model name exists, do you want to "\
                                    f"override model name?[y:N]")
                        if inp.lower()[0] == "y":
                            shutil.rmtree(self.log_path)
                        else:
                            print(">> Stop training!")
                            exit()
                else:
                    print(">>Model with the same name exists, set "\
                          "replce_old_model to enforce the overriding")
                    exit()

        # make sure height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        # Fixed seed
        same_seeds(self.opt.seed)
        # TODO: the no_cuda might be replaced by the gpus option
        if self.opt.no_cuda or not torch.cuda.is_available() or len(self.opt.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.opt.gpus[self.opt.rank]}')
            torch.cuda.set_device(self.device)

        # scales used in the loss
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        if self.opt.rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)

        print("Training is using:\n  ", self.device)

    def _init_depth_net(self):
        raise NotImplementedError

    def _init_pose_net(self):
        raise NotImplementedError

    def _init_optimizer(self):
        """Add non-freezed parameters in the model to the optimizer
        
        Applicable to any methodologies if the requirement
        is simply to assign weight decay values to corresponding
        networks
        """

        # list of dictionary
        parameters_to_train = []

        for n in self.models:
            if n in self.opt.models_to_freeze:
                print(f"freezing {n} weights")
                for param in self.models[n].parameters():
                    param.requires_grad = False
            else:
                try:
                    parameters_to_train += [
                        {'params': self.models[n].parameters(),
                         'weight_decay': eval(f'self.opt.{n}_weight_decay')}
                         ]
                except:
                    print(f'use the default weight decay for {n} network')
                    parameters_to_train += [
                        {'params': self.models[n].parameters(),
                         'weight_decay': self.opt.weight_decay}
                         ]

        self.model_optimizer = optim.Adam(
            parameters_to_train,
            self.opt.learning_rate)

        scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            self.opt.scheduler_step_size, self.opt.scheduler_gamma)

        self.model_lr_scheduler = scheduler 

    def _init_models(self):
        """Initialize model parameters

        Applicable to any methodologies

        """

        weight_initializer = select_weight_initializer(
                self.opt.init_mode, *self.opt.init_params)

        for n in self.opt.models_to_init:
            try:
                self.models[n].apply(
                        select_weight_initializer(
                            eval(f'self.opt.init_{n}_mode'),
                            *eval(f'self.opt.init_{n}_params')
                            )
                        )
            except AttributeError:
                print(f'use the default weight_initializer for {n} network')
                self.models[n].apply(weight_initializer)

    def _prep_distributed(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(self.opt.dist_backend, rank=self.opt.rank,
                                world_size=self.opt.world_size)
        for n in self.models:
            sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(self.models[n])
            sync_bn_network.to(self.device)
            self.models[n] = DDP(
                        sync_bn_network,
                        device_ids=[self.device],
                        broadcast_buffers=True,
                        find_unused_parameters=True)

    def _init_ssim(self):
        """Define SSIM Layer
        """
        if self.opt.weighted_ssim:
            self.ssim = WeightedSSIM(c1=float('inf'), c2=9e-6)
            self.ssim.to(self.device)
        else:
            self.ssim = SSIM()
            self.ssim.to(self.device)

    def _init_backproject_depth(self):
        backproject_depth = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            backproject_depth[scale].to(self.device)
        self.backproject_depth = backproject_depth

    def _init_project_3d(self):
        """Initialze the layer to project 3d points to image plane
        """
        # TODO: add an argument to select which layer is initialized
        project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            project_3d[scale].to(self.device)
        self.project_3d = project_3d

    def _prep_data_lists(self):
        """
        Applicable to all the methodologies
        """
        if isinstance(self.opt.data_path, list):
            fpath = [os.path.join(dp, "{}_files.txt") for dp in
                     self.opt.data_path]
        else:
            fpath = os.path.join(
                    self.opt.data_path,
                    "{}_files.txt")

        if isinstance(fpath, list):
            train_filenames = [readlines(fph.format("train")) for fph in
                               fpath]
            val_filenames = [readlines(fph.format("val")) for fph in
                             fpath]
        else:
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))

        train_filenames = self._sample_subset(train_filenames)
        val_filenames = self._sample_subset(val_filenames)

        self.train_filenames = train_filenames
        self.val_filenames = val_filenames

    def _sample_subset(self, filenames):
        if isinstance(self.opt.subset_ratio, list):
            assert len(self.opt.subset_ratio) == len(filenames)
            new_filenames = []
            for sub_r, fs in zip(self.opt.subset_ratio, filenames):
                if sub_r != 1.0:
                    num = int(len(fs) * sub_r)
                    new_filenames.append(random.sample(fs, num))
                else:
                    new_filenames.append(fs)
            filenames = new_filenames

        elif self.opt.subset_ratio != 1.0:
            if all(isinstance(fs, list) for fs in filenames):
                nums = [int(len(fs) * self.opt.subset_ratio) for fs in filenames]
                filenames = [random.sample(fs, num) for fs,num in zip(filenames, nums)]
            else:
                num = int(len(filenames) * self.opt.subset_ratio)
                filenames = random.sample(filenames, num)

        return filenames
        
    def _init_dataloaders(self):
        """
        Applicable to all the methodologies
        """
        if isinstance(self.opt.data_path, str):
            self.dataset = datasets.CustomMonoDataset
        else:
            self.dataset = datasets.MixedMonoDataset

        img_ext = '.png' if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.opt.data_path, self.train_filenames, self.opt.height, self.opt.width,
            self.frames_to_load, self.num_scales, is_train=True,
            img_ext=img_ext, not_do_color_aug=self.opt.not_do_color_aug,
            not_do_flip=self.opt.not_do_flip,
            do_crop=self.opt.do_crop, crop_bound=self.opt.crop_bound,
            seg_mask=self.opt.seg_mask, boxify=self.opt.boxify,
            MIN_OBJECT_AREA=self.opt.MIN_OBJECT_AREA,
            prob_to_mask_objects=self.opt.prob_to_mask_objects)
        val_dataset = self.dataset(
            self.opt.data_path, self.val_filenames, self.opt.height, self.opt.width,
            self.frames_to_load, self.num_scales, is_train=False,
            img_ext=img_ext, not_do_color_aug=self.opt.not_do_color_aug,
            not_do_flip=self.opt.not_do_flip,
            do_crop=self.opt.do_crop, crop_bound=self.opt.crop_bound,
            seg_mask=self.opt.seg_mask, boxify=self.opt.boxify,
            MIN_OBJECT_AREA=self.opt.MIN_OBJECT_AREA,
            prob_to_mask_objects=0.0)

        # TODO: check the priority
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset,
                                               shuffle=True)
            val_sampler = DistributedSampler(val_dataset,
                                             shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            sampler=val_sampler, num_workers=self.opt.num_workers,
            pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        if self.opt.images_to_predict_depth:
            self._init_test_dataloader()

        self.repr_intrinsics = val_dataset.get_repr_intrinsics()

    def _init_test_dataloader(self):
        """
        """
        test_transform = transforms.Compose([
            transforms.Resize((self.opt.height, self.opt.width)),
            transforms.ToTensor()
            ])

        test_dataset = ImageDataset(
                self.opt.images_to_predict_depth,
                transform=test_transform)


        self.test_loader = DataLoader(
                test_dataset, batch_size = self.opt.batch_size,
                shuffle = False, pin_memory = True,
                num_workers = self.opt.num_workers, drop_last=False)

    def _init_writers(self):
        """

        Applicable to all the methodologies
        """
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(
                    os.path.join(self.log_path, mode)
                    )

        if self.opt.images_to_predict_depth:
            self.writers['test'] = SummaryWriter(
                    os.path.join(self.log_path, 'test')
                    )

    def _close_writers(self):
        """
        Applicable to all the methodologies
        """
        for writer in self.writers.values():
            writer.close()

    def set_train(self):
        """Convert all models to training mode

        Applicable to all the methodologies
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode

        Applicable to all the methodologies
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline

        Applicable to all the methodologies
        """
        cnt = 0
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        min_train_loss = np.inf
        min_val_loss = np.inf

        for self.epoch in range(
                1, self.opt.warmup_epochs + self.opt.num_epochs + 1):

            if self.epoch == self.opt.warmup_epochs + 1:
                if self.opt.warmup_epochs > 0:
                    self.post_warmup_init()

            if self.epoch == self.freeze_epoch + 1:
                self.halfway_freezing()

            if self.opt.rank == 0:
                print(f">> Epoch: {self.epoch}")
            
            curr_train_loss, curr_val_loss = self.run_epoch()

            if self.opt.rank == 0:
                if self.opt.save_frequency > 0:
                    if (self.epoch + 1) % self.opt.save_frequency == 0:
                        self.save_model()
                else:
                    # only save the best model
                    if (curr_train_loss < min_train_loss and
                        curr_val_loss < min_val_loss):
                        self.save_model()
                        min_train_loss = curr_train_loss
                        min_val_loss = curr_val_loss
                        cnt = 0 # reset the count
                    else:
                        cnt += 1
                        if (self.opt.early_stop_patience > 0 and
                            cnt > self.opt.early_stop_patience):
                            print(colored(">>Early stop!!!", "cyan"))
                            break

        if self.opt.rank == 0:
            self._close_writers()
            print(f"Finish training model: {self.opt.model_name}")

    def halfway_freezing(self):
        """Freeze specific networks after freezing_epoch in train """
        raise NotImplementedError

    def post_warmup_init(self):
        """Reinitialize the trainer after the warmup epochs"""
        raise NotImplementedError

    def val(self):
        """Validate the model on a single minibatch

        Applicable to all the methodologies
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if self.opt.rank == 0:
                self.log_losses("val", losses)

        self.set_train()

    def full_val(self):
        """Validate the model on the full validation data

        Applicable to all the methodologies
        """
        self.set_eval()
        valid_loss_sum = 0.0
        trange = tqdm(self.val_loader)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(trange):
                outputs, losses = self.process_batch(inputs)

                valid_loss_sum += losses["loss"].item()
                if self.opt.rank == 0:
                    trange.set_description(
                            f"Valid loss: \33[91m>>"\
                            f"{valid_loss_sum/(batch_idx+1):.5f}<<\33[0m")

        mean_valid_loss = valid_loss_sum/(batch_idx+1)

        if self.opt.rank == 0:
            self.log_losses('val', mean_valid_loss)
            self.log("val", inputs, outputs)

        self.set_train()

        return mean_valid_loss

    def set_BN_track_running_stats(self, module, ifTrack=True):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats=ifTrack
        for child in module.children():
            self.set_BN_track_running_stats(module=child, ifTrack=ifTrack)

    def run_epoch(self):
        """Run a single epoch of training and validation

        Applicable to all the methodologies
        """

        if self.opt.rank == 0:
            print("Training")
        self.set_train()

        train_loss_sum = 0.0
        trange = tqdm(self.train_loader)

        for batch_idx, inputs in enumerate(trange):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            # clip gradient by L2 norm
            for name in self.models:
                if name not in self.opt.models_to_freeze:
                    torch.nn.utils.clip_grad_norm_(
                            self.models[name].parameters(),
                            self.opt.gradient_clip_norm
                            )

            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # clamp model weights after optimization
            self._weight_clipper()

            train_loss_sum += losses["loss"].item()

            if self.opt.rank == 0:
                trange.set_description(
                        f"Train loss: {train_loss_sum / (batch_idx+1):.5f}"
                        )

            if self.opt.log_frequency != 0:
                # log less frequently after the first 2000 steps
                early_phase = (batch_idx % self.opt.log_frequency == 0 and
                               self.step < 2000)
                late_phase = self.step % 2000 == 0

                if early_phase or late_phase:
                    self.log_time(batch_idx, duration,
                                  losses["loss"].cpu().data)

                if self.opt.rank == 0:
                    self.log_losses("train", losses)
                self.val()

            self.step += 1

        mean_train_loss = train_loss_sum/(batch_idx+1)

        if self.opt.rank == 0:
            # logging mean_train_loss
            self.log_losses("train", mean_train_loss)
            self.log("train", inputs, outputs)

            # TODO: why the test inference does not require synchronization
            if self.opt.images_to_predict_depth:
                self.log_test('test')

        mean_valid_loss = self.full_val()

        self.model_lr_scheduler.step()

        return mean_train_loss, mean_valid_loss 

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal

        Applicable to all the methodologies
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log_losses(self, mode, losses):
        """Log all the losses to tensorboard

        Applicable to all the methodologies
        """
        writer = self.writers[mode]
        if isinstance(losses, dict):
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, self.step)
        else:
            writer.add_scalar("mean_loss", losses, self.epoch)
            if mode == 'train' and self.opt.log_lr:
                writer.add_scalar(
                        'learning_rate',
                        self.model_optimizer.param_groups[0]['lr'],
                        self.epoch)

    def log(self, mode, inputs, outputs):
        """Log images and diagrams to tensorboard

        Should be customized by methodologies
        """
        raise NotImplementedError

    def log_test(self, mode):
        """Run the depth network on test images and save results to tensorboard

        Notice: Monodepth's depth network output disparities
                Wild's depth network output depths 
        Should be customized by methodologies

        """
        raise NotImplementedError

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        Applicable to all the methodologies
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk

        Applicable to all the methodologies
        """
        if self.opt.save_frequency == 0:
            print("best model saved!!!")
            save_folder = os.path.join(self.log_path, "models", "best")
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                try:
                    to_save['repr_intrinsics'] = self.repr_intrinsics
                except:
                    pass
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk

        Applicable to all the methodologies
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if os.path.isfile(path):
                print(f"Loading {n} weights")
            else:
                print(f"Cannot find {n} weights so {n} is initialized by default")
                continue

            if n != "adam":
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            else:
                optimizer_dict = torch.load(path)
                self.model_optimizer.load_state_dict(optimizer_dict)

    def _weight_clipper(self):
        """ Clip model weights and parameters after optimization
        """
        pass
