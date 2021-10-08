import wandb
import math, random
import json
import time
from typing import List

# Get dataset and model
from dataloader import get_dataset
from models.resnet import (
    resnet50,
    resnet101,
    resnet152
)

# PyTorch
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

# Get Micro-batch streaming.
from mbs.micro_batch_streaming import MicroBatchStreaming


class XcepTrainer:
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

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _save_log(
        cls, log, is_mbs: bool, batch_size: int, dataset_info: str, mbs_batchnorm: bool, random_seed: int
    ) -> None:
        ensure_dir("./loss/")
        if is_mbs:
            if mbs_batchnorm:
                with open(f"./loss/mbs_with_bn_{batch_size}_{dataset_info}_seed_{random_seed}.json", "w") as file:
                    json.dump(log, file, indent=4)
            else:
                with open(f"./loss/mbs_without_bn_{batch_size}_{dataset_info}_seed_{random_seed}.json", "w") as file:
                    json.dump(log, file, indent=4)
        else:
            with open(f"./loss/baseline_{batch_size}_{dataset_info}_seed_{random_seed}.json", "w") as file:
                json.dump(log, file, indent=4)

    @classmethod
    def _get_data_loader(cls, config: DotDict, args, is_train: bool) -> DataLoader:
        if is_train:
            dataset_type = config.data.dataset.train.type
        else:
            dataset_type = config.data.dataset.test.type

        dataset = get_dataset(
            path=config.data.dataset.train.path + config.data.dataset.train.type,
            dataset_type=dataset_type,
            config=config,
            args=args,
            is_train=is_train
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.dataset.train.batch_size,
            num_workers=config.data.dataset.train.num_worker,
            shuffle=config.data.dataset.train.shuffle,
            pin_memory=config.data.dataset.train.num_worker,
        )
        return dataloader

    def _select_model(self, num_classes: int):
        if self.args.version == 50:
            return resnet50(num_classes)
        elif self.args.version == 101:
            return resnet101(num_classes)
        elif self.args.version == 152:
            return resnet152(num_classes)

    def _print_learning_info(self):
        print(f'WanDB? {self.args.wandb}')
        print(f"Random Seed : {self.args.random_seed}")
        print(f"Epoch : {self.config.data.train.epoch}")
        print(f"Batch size : {self.config.data.dataset.train.batch_size}")
        print(f"Image size : {self.config.data.dataset.train.image_size}")
        print(f"num of classes : {self.config.data.dataset.train.num_classes}")

    def train(self) -> None:
        # wandb
        if self.args.wandb:
            name = None
            name = f'resnet_{self.args.version}_'
            name += f'batch{self.config.data.dataset.train.batch_size}_'
            name += f'image{self.config.data.dataset.train.image_size}_'
            name += f'cifar{self.config.data.dataset.train.num_classes}'
            # if self.config.data.microbatchstream.enable:
            if not self.args.origin:
                name += f'_mbs_{self.config.data.microbatchstream.micro_batch_size}'
            name += f'_worker{self.config.data.dataset.train.num_worker}'
            name += f'_pinmemory_{self.config.data.dataset.train.pin_memory}'

            wandb.init(
                project='mbs_with_pure_code',
                entity='xypiao97',
                name=f'{name}')

        # Setting Random seed.
        random_seed = self.args.random_seed
        self._print_learning_info()

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        random.seed(random_seed)

        device, _ = prepare_device(target=self.config.data.gpu.device)
        train_dataloader = self._get_data_loader(self.config, self.args, True)
        val_dataloader = self._get_data_loader(self.config, self.args, False)

        # build model and loss, optimizer
        model = self._select_model(self.config.data.dataset.train.num_classes)
        if self.args.origin:
            model = model.to(device)
        else:
            self.mbs = MicroBatchStreaming()
            model = self.mbs.set_batch_norm(model).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        opt = torch.optim.SGD(
            model.parameters(), 
            lr=self.config.data.optimizer.lr,
            momentum=self.config.data.optimizer.mometum,
            weight_decay=self.config.data.optimizer.decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.1, patience=5
        )

        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.scheduler = scheduler

        if self.args.origin:
            for epoch in range(self.config.data.train.epoch):
                self._origin_train_epoch(epoch, train_dataloader, device)
                self._val_accuracy(epoch, val_dataloader, device)
                print(f"top1:{self.max_top1}, top5:{self.max_top5}, epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}\r")
        else:
            for epoch in range(self.config.data.train.epoch):
                self._train_epoch(epoch, train_dataloader, device)
                self._val_accuracy(epoch, val_dataloader, device)
                print(f"top1:{self.max_top1}, top5:{self.max_top5}, epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}\r")
                # self._grad_magnitude(epoch)

        # for epoch in self.train_loss:
        #     self.val_accuracy[epoch]['train loss'] = self.train_loss[epoch]
        #     self.val_accuracy[epoch]['epoch avg time'] = self.epoch_time[epoch]
        #     # self.val_accuracy[epoch]['magnitude'] = self.magnitude[epoch]

        # self._save_log(
        #     self.val_accuracy,
        #     self.config.data.microbatchstream.enable,
        #     self.config.data.dataset.train.batch_size,
        #     self.config.data.dataset.train.type,
        #     self.config.data.microbatchstream.batchnorm,
        #     random_seed
        # )

        # if self.config.data.microbatchstream.enable:
        #     if self.config.data.microbatchstream.batchnorm:
        #         self._save_state_dict(
        #             self.model.state_dict(),
        #             f"./parameters/{self.config.data.dataset.train.type}_mbs_with_bn_{random_seed}/model.pth"
        #         )
        #     else:
        #         self._save_state_dict(
        #             self.model.state_dict(),
        #             f"./parameters/{self.config.data.dataset.train.type}_mbs_without_bn_{random_seed}/model.pth"
        #         )
        # else:
        #     self._save_state_dict(
        #         self.model.state_dict(),
        #         f"./parameters/{self.config.data.dataset.train.type}_baseline_{random_seed}/model.pth"
        #     )

        print(f"top1:{self.max_top1}, top5:{self.max_top5}, epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}\n")

    def _train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        # Define check performance vars
        mini_loss = []
        dataloader_len = len(dataloader)

        epoch_start = time.perf_counter()
        self.model.train()
        for idx, (input, label) in enumerate(dataloader):
            print(f'mbs > [{epoch + 1}/{self.config.data.train.epoch}] [{idx+1}/{dataloader_len}]    ', end='\r')

            if input.size(0) == self.config.data.dataset.train.batch_size:
                chunks = math.ceil( self.config.data.dataset.train.batch_size / self.config.data.microbatchstream.micro_batch_size )
            else:
                chunks = math.ceil( input.size(0) / self.config.data.microbatchstream.micro_batch_size )

            input : torch.Tensor
            label : torch.Tensor

            chunk_input = input.chunk( chunks )
            chunk_label = label.chunk( chunks )

            self.opt.zero_grad()
            micro_loss = 0

            self.mbs._update_timing = False

            for idx, (iinput, llabel) in enumerate(zip( chunk_input, chunk_label )):
                if idx + 1 == chunks:
                    self.mbs._update_timing = True

                input = iinput.to(device)
                label = llabel.to(device)

                output : torch.Tensor = self.model( input )
                loss : torch.Tensor = self.criterion( output, label ) / chunks
                micro_loss += loss.item()
                loss.backward()

            self.opt.step()
            mini_loss.append( micro_loss )
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        avg_loss = sum(mini_loss)/len(mini_loss)

        if self.args.wandb:
            wandb.log( {'train loss': avg_loss}, step=epoch )
            wandb.log( {'epoch time' : epoch_time}, step=epoch)
        self.epoch_avg_time.append( epoch_time )
        # self._epoch_writer(
        #     epoch,
        #     self.config.data.train.epoch,
        #     train_time, train_iter,
        #     epoch_time, epoch_iter,
        #     avg_loss
        # )

    def _origin_train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        # Define check performance vars
        mini_loss = []
        dataloader_len = len(dataloader)

        epoch_start = time.perf_counter()
        self.model.train()
        for idx, (input, label) in enumerate(dataloader):
            print(f'origin > [{epoch + 1}/{self.config.data.train.epoch}] [{idx+1}/{dataloader_len}]    ', end='\r')

            input : torch.Tensor = input.to(device)
            label : torch.Tensor = label.to(device)

            self.opt.zero_grad()

            output : torch.Tensor = self.model( input )
            loss : torch.Tensor = self.criterion( output, label )
            loss.backward()

            self.opt.step()
            mini_loss.append(loss.item())

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        avg_loss = sum(mini_loss)/len(mini_loss)

        if self.args.wandb:
            wandb.log( {'train loss': avg_loss}, step=epoch )
            wandb.log( {'epoch time' : epoch_time}, step=epoch)
        self.epoch_avg_time.append( epoch_time )
        # self._epoch_writer(
        #     epoch,
        #     self.config.data.train.epoch,
        #     train_time, train_iter,
        #     epoch_time, epoch_iter,
        #     avg_loss
        # )

    def _val_accuracy(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        total = 0
        correct_top1 = 0
        correct_top5 = 0
        self.model.eval()
        with torch.no_grad():
            for _, (input, label) in enumerate(dataloader):
                input = input.to(device)
                label = label.to(device)
                output : torch.Tensor = self.model(input)

                # rank 1
                _, pred = torch.max(output, 1)
                total += label.size(0)
                correct_top1 += (pred == label).sum().item()

                # rank 5
                _, rank5 = output.topk(5, 1, True, True)
                rank5 = rank5.t()
                correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))

                for k in range(6):
                    correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)
                correct_top5 += correct_k.item()

        if self.args.wandb:
            wandb.log( {'top-1': 100 * ( correct_top1 / total )}, step=epoch )
            wandb.log( {'top-5': 100 * ( correct_top5 / total )}, step=epoch )
        self.val_accuracy[epoch + 1] = {
            'top-1' : 100 * ( correct_top1 / total ),
            'top-5' : 100 * ( correct_top5 / total )
        }
        if self.max_top1 < (100 * (correct_top1 / total)):
            self.max_top1 = (100 * (correct_top1 / total))
        if self.max_top5 < (100 * (correct_top5 / total)):
            self.max_top5 = (100 * (correct_top5 / total))

        # print(
        #     'top-1 :',
        #     format(100 * ( correct_top1 / total ), ".2f"),
        #     'top-5 :',
        #     format(100 * ( correct_top5 / total ), ".2f"),
        # )

    def _epoch_writer(
        self,
        epoch: int,
        epochs: int,
        train_time: float,
        train_iter: int,
        epoch_time: float,
        epoch_iter: int,
        epoch_loss: float
    ) -> None:
        self.train_loss[epoch + 1] = epoch_loss
        self.epoch_time[epoch + 1] = epoch_time / epoch_iter

        print(
            f"\n[{epoch+1}/{epochs}]",
            "train time :",
            format(train_time / train_iter, ".3f") + "s",
            "epoch time :",
            format(epoch_time / epoch_iter, ".3f") + "s",
            end=" ",
        )
        print(
            # f"loss : {self.train_loss[epoch+1]}",
            f"loss : ",
            format(self.train_loss[epoch + 1], ".4f"),
            end=' '
        )

    def _grad_magnitude(self, epoch : int) -> None:
        self.magnitude[epoch + 1] = {}
        for layer, para in self.model.named_parameters():
            magnitude : torch.Tensor = torch.sum( para.grad )
            self.magnitude[epoch + 1][layer] = magnitude.item()

def train(config: ConfigParser, args):
    trainer = XcepTrainer(config, args)
    trainer.train()