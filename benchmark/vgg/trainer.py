import itertools

import json
import time
from typing import List

# Get dataset and model
from dataloader import get_dataset
from models.vgg.vgg import select_model

# PyTorch
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

# Get Micro-batch streaming.
from mbs.micro_batch_streaming import MicroBatchStreaming


class VGGTrainer:
    def __init__(self, config: ConfigParser) -> None:
        self.config = config
        self.json_file = {}

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _save_log(cls, log, is_mbs, batch_size) -> None:
        ensure_dir("./loss/")
        with open(f"./loss/vgg_mbs_{is_mbs}_{batch_size}_loss_value.json", "w") as file:
            json.dump(log, file, indent=4)

    @classmethod
    def _get_data_loader(cls, dataset_config: DotDict) -> DataLoader:
        dataloader = get_dataset(
            path=dataset_config.path + dataset_config.type,
            dataset_type=dataset_config.type,
            batch_size=dataset_config.batch_size,
            image_size=dataset_config.image_size,
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
        self.vgg_model = select_model(
            self.config.data.model.normbatch,
            self.config.data.model.version,
            self.config.data.dataset.train.num_classes
        ).to(device)

        # Define loss function.
        self.criterion = nn.CrossEntropyLoss().to(device)

        # Define optimizers
        self.opt = torch.optim.SGD( 
            self.vgg_model.parameters(), 
            lr=self.config.data.optimizer.lr,
            weight_decay=self.config.data.optimizer.decay,
        )

    def train(self) -> None:
        device, _ = prepare_device(target=self.config.data.gpu.device)
        dataloader = self._get_data_loader(self.config.data.dataset.train)
        self._get_model_optimizer(
            device=device, image_size=self.config.data.dataset.train.image_size
        )

        optim_list = [self.opt]

        if self.config.data.microbatchstream.enable:
            dataloader, optim_list = self._init_microbatch_stream(
                dataloader, optim_list, self.config.data.microbatchstream.micro_batch_size
            )

        self.loss_values = {}

        for epoch in range(self.config.data.train.epoch):
            self._train_epoch(epoch, dataloader, self.loss_values, device)

        self._save_log(self.json_file, self.config.data.microbatchstream.enable, self.config.data.dataset.train.batch_size)

        self._save_state_dict(
            self.vgg_model.state_dict(),
            f"./parameters/{self.config.data.dataset.train.type}_mbs_{self.config.data.microbatchstream.enable}/vgg.pth"
        )

    def _train_epoch(self, epoch, dataloader, loss_values, device) -> None:

        # Define check performance vars
        epoch_time = 0
        epoch_iter = 0
        train_time = 0
        train_iter = 0

        epoch_start = time.perf_counter()
        loss_values[epoch] = {"loss": 0.0}
        # pre_para = None
        for idx, (image, label) in enumerate(dataloader):
            train_start = time.perf_counter()

            input = image.to(device)
            label = label.to(device)
            output = self.vgg_model( input )
            loss = self.criterion( output, label )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            train_end = time.perf_counter()
            train_time += train_end - train_start
            train_iter += 1
            loss_values[epoch]["loss"] += loss.detach().item()
        epoch_end = time.perf_counter()
        epoch_time += epoch_end - epoch_start
        epoch_iter += 1

        loss_values[epoch]["loss"] /= idx
        if epoch == 0:
            self.init_loss_value = loss_values[epoch]["loss"]

        self._epoch_writer(
            epoch,
            self.config.data.train.epoch,
            train_time,
            train_iter,
            epoch_time,
            epoch_iter
        )

    def _epoch_writer(
        self,
        epoch: int,
        epochs: int,
        train_time: float,
        train_iter: int,
        epoch_time: float,
        epoch_iter: int,
    ) -> None:
        print(
            f"[{epoch+1}/{epochs}]",
            "train time :",
            format(train_time / train_iter, ".3f") + "s",
            "epoch time :",
            format(epoch_time / epoch_iter, ".3f") + "s",
            end=" ",
        )
        self.json_file[epoch + 1] = {}
        for _, name in enumerate(self.loss_values[epoch]):
            print(
                f"{name} :",
                format(self.loss_values[epoch][name], ".2f")
            )
            self.json_file[epoch + 1][name] = self.loss_values[epoch][name]


def train(config: ConfigParser):
    trainer = VGGTrainer(config)
    trainer.train()
