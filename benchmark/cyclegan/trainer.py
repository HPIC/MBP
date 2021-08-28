import itertools

# import json
import time
from typing import List

import torch
from dataloader import cyclegan_dataset
from models.cyclegan.cyclegan import (
    PatchGAN_Discriminator,
    Resnet_Generator,
    initialize_model_list,
)
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

from mbs.micro_batch_streaming import MicroBatchStreaming


class CycleGANTrainer:
    def __init__(self, config: ConfigParser) -> None:
        self.config = config

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _save_log(cls, log, batch_size) -> None:
        ensure_dir("./loss/")
        # with open(f"./loss/cyclegan_mbs_{batch_size}_loss_value.json", "w") as file:
        #     json_file = {}

    @classmethod
    def _get_cyclegan_data_loader(cls, dataset_config: DotDict) -> DataLoader:
        dataloader = cyclegan_dataset(
            path=dataset_config.path,
            image_size=dataset_config.image_size,
            batch_size=dataset_config.batch_size
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

    def _get_cyclegan_models_optimizers(self, device: torch.device, image_size: int) -> None:
        # Define Models
        block = 9 if image_size == 256 else 6
        self.size_ = 15 if image_size == 256 else 7
        self.G_A = Resnet_Generator(block=block).to(device)
        self.G_B = Resnet_Generator(block=block).to(device)
        self.D_A = PatchGAN_Discriminator().to(device)
        self.D_B = PatchGAN_Discriminator().to(device)

        initialize_model_list(models=[self.G_A, self.G_B, self.D_A, self.D_B])

        # Define loss function.
        self.cyc_l1loss = nn.L1Loss().to(device) 
        self.idt_l1loss = nn.L1Loss().to(device)
        self.adv_loss = nn.MSELoss().to(device)

        # Define optimizers
        self.opt_G = torch.optim.Adam(
            itertools.chain(self.G_A.parameters(), self.G_B.parameters()), lr=0.0002
        )
        self.opt_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr=0.0002
        )

        # Define vars
        self.lambda_A = 10
        self.lambda_B = 10
        self.lambda_idt = 0.5

        self.loss_values = {}

    def train(self) -> None:
        device, _ = prepare_device(target=self.config.data.gpu.device)
        dataloader = self._get_cyclegan_data_loader(self.config.data.dataset.train)
        self._get_cyclegan_models_optimizers(
            device=device, image_size=self.config.data.dataset.train.image_size
        )

        optim_list = [self.opt_G, self.opt_D]

        if self.config.data.microbatchstream.enable:
            dataloader, optim_list = self._init_microbatch_stream(
                dataloader, optim_list, self.config.data.microbatchstream.micro_batch_size
            )

        self.loss_values = {}

        for epoch in range(self.config.data.train.epoch):
            self._train_epoch(epoch, dataloader, self.loss_values, device)

        self._save_state_dict(
            self.G_A.state_dict(), "./parameters/cyclegan_mbs/G_A.pth"
        )
        self._save_state_dict(
            self.G_B.state_dict(), "./parameters/cyclegan_mbs/G_B.pth"
        )
        self._save_state_dict(
            self.D_A.state_dict(), "./parameters/cyclegan_mbs/D_A.pth"
        )
        self._save_state_dict(
            self.D_B.state_dict(), "./parameters/cyclegan_mbs/D_B.pth"
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
        # json_file[epoch + 1] = {}
        for _, name in enumerate(self.loss_values[epoch]):
            print(
                f"{name} :",
                format(self.loss_values[epoch][name], ".2f"),
                end=" ",
            )
        #     json_file[epoch + 1][name] = loss_values[epoch][name]
        # print()
        # json.dump(json_file, file, indent=4)

    def _train_epoch(self, epoch, dataloader, loss_values, device) -> None:

        # Define check performance vars
        epoch_time = 0
        epoch_iter = 0
        train_time = 0
        train_iter = 0

        epoch_start = time.perf_counter()
        loss_values[epoch] = {"g_loss": 0.0, "A_loss": 0.0, "B_loss": 0.0}
        # pre_para = None
        for idx, (ze, up, (real_A, real_B, _, _)) in enumerate(dataloader):
            train_start = time.perf_counter()

            # forward and backward
            self.opt_G.zero_grad_accu(ze)
            self.opt_D.zero_grad_accu(ze)

            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = self.G_A(real_A)
            recover_A = self.G_B(fake_B)

            fake_A = self.G_B(real_B)
            recover_B = self.G_A(fake_A)

            idt_B = self.G_A(real_B)
            idt_A = self.G_B(real_A)

            """ create real and fake label for adversarial loss """
            batch_size = real_A.size(0)
            real_label = torch.ones(batch_size, 1, self.size_, self.size_).to(device)
            fake_label = torch.zeros(batch_size, 1, self.size_, self.size_).to(device)

            fake_output_A = self.D_A(fake_A)
            adv_loss_A = self.adv_loss(fake_output_A, real_label)
            fake_output_B = self.D_B(fake_B)
            adv_loss_B = self.adv_loss(fake_output_B, real_label)

            """ Calculate cycle loss """
            cyc_loss_A = self.cyc_l1loss(recover_A, real_A) * self.lambda_A
            cyc_loss_B = self.cyc_l1loss(recover_B, real_B) * self.lambda_B

            """ Calculate idt loss """
            idt_loss_A = (
                self.idt_l1loss(idt_A, real_A) * self.lambda_A * self.lambda_idt
            )
            idt_loss_B = (
                self.idt_l1loss(idt_B, real_B) * self.lambda_B * self.lambda_idt
            )
            g_loss = (
                adv_loss_A
                + adv_loss_B
                + cyc_loss_A
                + cyc_loss_B
                + idt_loss_A
                + idt_loss_B
            )

            real_output_A = self.D_A(real_A)

            real_A_loss = self.adv_loss(real_output_A, real_label)

            fake_output_A = self.D_A(fake_A.detach())
            fake_A_loss = self.adv_loss(fake_output_A, fake_label)

            real_output_B = self.D_B(real_B)
            real_B_loss = self.adv_loss(real_output_B, real_label)

            fake_output_B = self.D_B(fake_B.detach())
            fake_B_loss = self.adv_loss(fake_output_B, fake_label)

            A_loss = real_A_loss + fake_A_loss
            B_loss = real_B_loss + fake_B_loss

            g_loss.backward()
            A_loss.backward()
            B_loss.backward()

            self.opt_G.step_accu(up)
            self.opt_D.step_accu(up)

            train_end = time.perf_counter()
            train_time += train_end - train_start
            train_iter += 1 if up else 0
            loss_values[epoch]["g_loss"] += g_loss.detach().item()
            loss_values[epoch]["A_loss"] += A_loss.detach().item()
            loss_values[epoch]["B_loss"] += B_loss.detach().item()
        epoch_end = time.perf_counter()
        epoch_time += epoch_end - epoch_start
        epoch_iter += 1

        loss_values[epoch]["g_loss"] /= idx
        loss_values[epoch]["A_loss"] /= idx
        loss_values[epoch]["B_loss"] /= idx


def train(config: ConfigParser):
    trainer = CycleGANTrainer(config)
    trainer.train()
