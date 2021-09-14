import math, random
import json
import time
from typing import List

# Get dataset and model
from dataloader import get_dataset
from models.efficientnet.effinet import EfficientNetB0

# PyTorch
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

# Get Micro-batch streaming.
from mbs.micro_batch_streaming import MicroBatchStreaming


class ENETTrainer:
    def __init__(self, config: ConfigParser) -> None:
        self.config = config

        self.train_loss = {}
        self.train_time = {}
        self.val_accuracy = {}

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _save_log(cls, log, is_mbs, batch_size) -> None:
        ensure_dir("./loss/")
        with open(f"./loss/xcep_mbs_{is_mbs}_{batch_size}.json", "w") as file:
            json.dump(log, file, indent=4)

    @classmethod
    def _get_data_loader(cls, dataset_config: DotDict, is_train: bool) -> DataLoader:
        dataloader = get_dataset(
            path=dataset_config.path + dataset_config.type,
            dataset_type=dataset_config.type,
            batch_size=dataset_config.batch_size,
            image_size=dataset_config.image_size,
            is_train=is_train
        )
        return dataloader

    @staticmethod
    def _init_microbatch_stream(
        dataloader: DataLoader, optim_list: List[Optimizer], micro_batch_size: int
    ) -> List[Optimizer]:
        # Define MicroBatchStreaming
        mbs = MicroBatchStreaming()
        dataloader = mbs.set_dataloader(dataloader, micro_batch_size)
        for optim in optim_list:
            optim = mbs.set_optimizer(optim)
        return dataloader, optim_list

    def _get_model_optimizer(self, device: torch.device, image_size: int) -> None:
        # Define Models
        model = EfficientNetB0(
            self.config.data.dataset.train.num_classes
        ).to(device)

        # Define loss function.
        criterion = nn.CrossEntropyLoss().to(device)

        # Define optimizers
        opt = torch.optim.SGD( 
            model.parameters(), 
            lr=self.config.data.optimizer.lr,
            momentum=self.config.data.optimizer.mometum,
            weight_decay=self.config.data.optimizer.decay,
        )

        return model, criterion, opt

    def train(self) -> None:
        # Setting Random seed.
        random_seed = 42

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        random.seed(random_seed)

        device, _ = prepare_device(target=self.config.data.gpu.device)
        train_dataloader = self._get_data_loader(self.config.data.dataset.train, True)
        val_dataloader = self._get_data_loader(self.config.data.dataset.test, False)
        model, criterion, opt = self._get_model_optimizer(
            device=device, image_size=self.config.data.dataset.train.image_size
        )

        if self.config.data.microbatchstream.enable:
            mbs = MicroBatchStreaming()
            train_dataloader = mbs.set_dataloader( train_dataloader, self.config.data.microbatchstream.micro_batch_size )
            criterion = mbs.set_loss( criterion )
            opt = mbs.set_optimizer( opt )
            print("Apply Micro Batch Streaming method!")

        self.model = model
        self.criterion = criterion
        self.opt = opt

        for epoch in range(self.config.data.train.epoch):
            self._train_epoch(epoch, train_dataloader, device)
            self._val_accuracy(epoch, val_dataloader, device)

        for epoch in self.train_loss:
            self.val_accuracy[epoch]['train loss'] = self.train_loss[epoch]
            self.val_accuracy[epoch]['epoch avg time'] = self.train_time[epoch]

        self._save_log(
            self.val_accuracy,
            self.config.data.microbatchstream.enable,
            self.config.data.dataset.train.batch_size,
            self.config.data.model.version,
            self.config.data.normbatch
        )

        self._save_state_dict(
            self.model.state_dict(),
            f"./parameters/{self.config.data.dataset.train.type}_mbs_{self.config.data.microbatchstream.enable}/model.pth"
        )

    def _train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        # Define check performance vars
        epoch_time = 0
        epoch_iter = 0
        train_time = 0
        train_iter = 0

        losses = 0

        epoch_start = time.perf_counter()
        # pre_para = None
        for idx, (image, label) in enumerate(dataloader):
            train_start = time.perf_counter()

            input = image.to(device)
            label = label.to(device)
            output = self.model( input )
            loss = self.criterion( output, label )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            train_end = time.perf_counter()
            train_time += train_end - train_start
            train_iter += 1
            losses += loss.detach().item()
        epoch_end = time.perf_counter()
        epoch_time += epoch_end - epoch_start
        epoch_iter += 1

        if self.config.data.microbatchstream.enable:
            losses *= math.ceil(self.config.data.dataset.train.batch_size / self.config.data.microbatchstream.micro_batch_size) 
        losses /= idx

        self._epoch_writer(
            epoch,
            self.config.data.train.epoch,
            train_time, train_iter,
            epoch_time, epoch_iter,
            losses
        )

    def _val_accuracy(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        total = 0
        correct_top1 = 0
        correct_top5 = 0
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

        self.val_accuracy[epoch + 1] = {
            'top-1' : 100 * ( correct_top1 / total ),
            'top-5' : 100 * ( correct_top5 / total )
        }
        print(
            'top-1 :',
            format(100 * ( correct_top1 / total ), ".2f"),
            'top-5 :',
            format(100 * ( correct_top5 / total ), ".2f"),
            end='\r'
        )

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
        self.train_time[epoch + 1] = epoch_time / epoch_iter

        print(
            f"[{epoch+1}/{epochs}]",
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


def train(config: ConfigParser):
    trainer = ENETTrainer(config)
    trainer.train()
