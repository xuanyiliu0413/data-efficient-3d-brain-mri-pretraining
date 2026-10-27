import os
import time
import random
from pathlib import Path

import numpy as np

import torch

import sys
sys.path.append('..')

import lib.models as models
import lib.networks as networks

import wandb

from .base_trainer import BaseTrainer
from lib.data.med_transforms import get_mae_pretrain_transforms, get_vis_transforms
from lib.data.med_datasets import get_train_loader, get_val_loader
from lib.tools.visualization import patches3d_to_grid

from timm.models.layers.helpers import to_3tuple

class MAE3DTrainer(BaseTrainer):



    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3D'
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch),
                            decoder=getattr(networks, args.dec_arch),
                            args=args)
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None),\
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()

        self.optimizer = torch.optim.AdamW(optim_params,
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats', 'fomo50k']:
                train_transform = get_mae_pretrain_transforms(args)
                self.dataloader = get_train_loader(args,
                                                   batch_size=self.batch_size,
                                                   workers=self.workers,
                                                   train_transform=train_transform)
                val_transform = get_vis_transforms(args)
                self.val_dataloader = get_val_loader(args,
                                                     batch_size=args.vis_batch_size,
                                                     workers=self.workers,
                                                     val_transform=val_transform)
            elif args.dataset == 'brats20':

                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")

            self.iters_per_epoch = len(self.dataloader)
            print(f"==> Length of train dataloader is {self.iters_per_epoch}")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args

        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()







            niters = self.epoch_train(epoch, niters)

            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                print(f"=> start visualizing after {epoch + 1} epochs")
                self.vis_reconstruction(niters)
                print("=> finish visualizing")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(),
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar')
                    print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler


        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time

            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)


            image = batch_data['image']

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)


            forward_start_time = time.time()
            with torch.cuda.amp.autocast(True):
                loss = model(image, return_image=False)
            forward_time = time.time() - forward_start_time


            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time


            if i % args.print_freq == 0:
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
            load_start_time = time.time()
        return niters

    def vis_reconstruction(self, niters=0):
        args = self.args
        loader = self.val_dataloader
        model = self.wrapped_model

        model.eval()

        for batch_data in loader:
            image = batch_data['image']
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)


            _, x, recon, masked_x = model(image, return_image=True)

            vis_tensor = torch.cat([x, masked_x, recon], dim=0)


            grid_size = []
            for pa_size, in_size in zip(to_3tuple(args.patch_size), to_3tuple(args.input_size)):
                grid_size.append(in_size // pa_size)
            vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='d')





            if getattr(args, 'disable_wandb', False):
                print("wandb disabled; skip reconstruction logging")
                break
            print("wandb logging")
            vis_grid_hw = wandb.Image(vis_grid_hw, caption=f"hw_iter{niters:06d}")



            wandb.log(
                {
                "vis_hw": vis_grid_hw,


                },
                step=niters,
            )
            break
        print("finish wandb logging")


    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:

                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
