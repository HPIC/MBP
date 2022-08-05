import itertools
import math, random
import json
import time
from typing import Dict, List

import wandb

# Get dataset and model
from dataloader import get_dataset
from models.resnet import (
    resnet50,
    resnet101,
    resnet152
)

# PyTorch
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device
from torch.nn.modules import Module

# Get Micro-batch streaming.
from mbs import MicroBatchStreaming

class ResNetTrainer:
    def __init__(self, config: ConfigParser, args) -> None:
        self.config = config
        self.args = args

        self.train_loss = {}
        self.epoch_time = {}
        self.val_accuracy = {}
        self.magnitude = {}

        self.max_top1 = 0
        self.max_top5 = 0
        self.epoch_avg_time = []

        self.bn_layer = {}

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _get_train_dataloader(cls, config: DotDict) -> DataLoader:
        dataset = get_dataset(
            path=config.path,
            dataset_type=config.type,
            args=config,
            is_train=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size  =config.train_batch,
            num_workers =config.num_workers,
            shuffle     =config.shuffle,
            pin_memory  =config.pin_memory,
        )
        return dataloader

    @classmethod
    def _get_test_dataloader(cls, config: DotDict) -> DataLoader:
        dataset = get_dataset(
            path=config.path,
            dataset_type=config.type,
            args=config,
            is_train=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size  =config.test_batch,
            num_workers =config.num_workers,
            shuffle     =config.shuffle,
            pin_memory  =config.pin_memory,
        )
        return dataloader

    def _select_model(self, num_classes: int):
        if self.config.data.model.version == 50:
            return resnet50(num_classes)
        elif self.config.data.model.version == 101:
            return resnet101(num_classes)
        elif self.config.data.model.version == 152:
            return resnet152(num_classes)

    def _print_learning_info(self, dataloader: DataLoader):
        print(f'WanDB? {self.config.data.wandb.enable}')
        print(f"Random Seed : {self.config.data.train.seed}")
        print(f"Epoch : {self.config.data.train.epoch}")
        print(f"Batch size : {dataloader.batch_size}")
        print(f"Image size : {self.config.data.dataset.train.image_size}")
        print(f"num of classes : {self.config.data.dataset.train.num_classes}")
        print(f"pin memory : {dataloader.pin_memory}")
        print(f"num workers : {dataloader.num_workers}")

        if self.config.data.mbs.enable:
            print(f">>> micro batch size : {self.config.data.mbs.micro_batch_size}")
            print(f"*** Training ResNet-{self.config.data.model.version} with MBS")
        else:
            print(f"*** Training ResNet-{self.config.data.model.version} (Baseline) ***")

    def _check_before_running(self, dataloader: DataLoader):
        if self.config.data.dataset.train.train_batch != dataloader.batch_size:
            raise ValueError("Batch size is not equal!")
        if self.config.data.dataset.train.pin_memory != dataloader.pin_memory:
            raise ValueError("Status of pin memory is not equal!")
        if self.config.data.dataset.train.num_workers != dataloader.num_workers:
            raise ValueError("Num of wokers is not equal!")

        if self.config.data.wandb.enable:
            name = None
            if self.config.data.mbs.enable:
                name = f'resnet-{self.config.data.model.version}(mbs)'
            else:
                name = f'resnet-{self.config.data.model.version}(baseline)'

            tags = []
            tags.append( f'{self.args.server}')
            tags.append( f'batch {dataloader.batch_size}' )
            tags.append( f'image {self.config.data.dataset.train.image_size}' )
            tags.append( f'worker {dataloader.num_workers}' )
            tags.append( f'pin memory {dataloader.pin_memory}' )
            tags.append( f'seed {self.config.data.train.seed}' )

            if self.config.data.mbs.enable:
                tags.append( f'mbs {self.config.data.mbs.micro_batch_size}' )

            wandb.init(
                project='mbs_ml_conference',
                entity='xypiao97',
                name=f'{name}',
                tags=tags,
            )


    def train(self) -> None:
        # Setting Random seed.
        random_seed = self.config.data.train.seed

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        random.seed(random_seed)

        device, _ = prepare_device(target=self.config.data.gpu.device)
        train_dataloader = self._get_train_dataloader( self.config.data.dataset.train )
        val_dataloader = self._get_test_dataloader( self.config.data.dataset.test )

        self._print_learning_info(train_dataloader)
        self._check_before_running(train_dataloader)

        # build model and loss, optimizer
        self.model = self._select_model(self.config.data.dataset.train.num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.opt = torch.optim.SGD( 
            self.model.parameters(), 
            lr=self.config.data.optimizer.lr,
            momentum=self.config.data.optimizer.mometum,
            weight_decay=self.config.data.optimizer.decay,
        )

        if self.config.data.mbs.enable:
            mbs_trainer, self.model = MicroBatchStreaming(
                dataloader=train_dataloader,
                model=self.model,
                criterion=self.criterion,
                optimizer=self.opt,
                lr_scheduler=None,
                device_index=self.config.data.gpu.device,
                batch_size=self.config.data.dataset.train.train_batch,
                micro_batch_size=self.config.data.mbs.micro_batch_size,
            ).get_trainer()

            for epoch in range(self.config.data.train.epoch):
                print( f"[{epoch+1}/{self.config.data.train.epoch}]" )
                # Train
                start = time.perf_counter()
                train_loss = mbs_trainer.train()
                end = time.perf_counter()
                self.epoch_avg_time.append( end - start )

                # Validation
                acc = self._val_accuracy(epoch, val_dataloader, device)

                # Update status to WandB
                if self.config.data.wandb.enable:
                    wandb.log( {'train loss': train_loss}, step=epoch )
                    wandb.log( {'epoch time' : sum(self.epoch_avg_time)/len(self.epoch_avg_time)}, step=epoch)
                print(  f"acc:{acc:.2f}",
                        f"epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}",
                        f"loss : {train_loss}",
                    )

                self._save_state_dict( 
                    self.model, 
                    f"./checkpoint/mbs/resnet_{self.config.data.model.version}/batch_{self.config.data.dataset.train.train_batch}/mb_{self.config.data.mbs.micro_batch_size}/seed_{self.config.data.train.seed}/para.pth" 
                )
        else:
            for epoch in range(self.config.data.train.epoch):
                # Train
                epoch_avg_loss, epoch_time = self._train_epoch(epoch, train_dataloader, device)

                # Validation
                acc = self._val_accuracy(epoch, val_dataloader, device)

                # Update status to WandB
                print(  f"acc:{acc:.2f}",
                        f"epoch time: {epoch_time}",
                        f"loss : {epoch_avg_loss}",
                        # end='\r',
                    )

                self._save_state_dict( 
                    self.model, 
                    f"./checkpoint/base/resnet_{self.config.data.model.version}/batch_{self.config.data.dataset.train.train_batch}/seed_{self.config.data.train.seed}/para.pth" 
                )


    def _train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        losses = []

        epoch_start = time.perf_counter()
        self.model.train()
        for idx, (input, label) in enumerate(dataloader):
            print(f'[{epoch + 1}/{self.config.data.train.epoch}] [{idx+1}/{len(dataloader)}]', end='\r')
            input: Tensor = input.to(device)
            label: Tensor = label.to(device)

            output: Tensor = self.model( input )
            loss: Tensor = self.criterion( output, label )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append( loss.detach().item() )
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_avg_loss = sum( losses ) / len( losses )

        if self.config.data.wandb.enable:
            wandb.log( {'train loss': epoch_avg_loss}, step=epoch )
            wandb.log( {'epoch time' : epoch_time}, step=epoch)

        return epoch_avg_loss, epoch_time


    def _val_accuracy(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        total = 0
        correct_top1 = 0
        self.model.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
                input: Tensor = input.to(device)
                label: Tensor = label.to(device)
                output : Tensor = self.model(input)

                # rank 1
                _, pred = torch.max(output, 1)
                total += label.size(0)
                correct_top1 += (pred == label).sum().item()
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        accuracy = 100 * ( correct_top1 / total )
        print( f"\nAccuracy : {accuracy:.2f}" )
        if self.config.data.wandb.enable:
            wandb.log( {'acc': accuracy}, step=epoch )
            wandb.log( {'inf time': inference_time}, step=epoch)

        return accuracy




def train(config: ConfigParser, args):
    trainer = ResNetTrainer(config, args)
    trainer.train()
