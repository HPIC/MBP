import itertools
import math, random
import json
import time
from typing import List

import wandb

# Get dataset and model
from dataloader import get_dataset
from models.vgg.vgg import select_model

# PyTorch
import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

# Get Micro-batch streaming.
from mbs import MicroBatchStreaming

class VGG:
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

    # @classmethod
    # def _get_data_loader(cls, config: DotDict, args, is_train: bool) -> DataLoader:
    #     dataset = get_dataset(
    #         path=config.path + config.type,
    #         dataset_type=config.type,
    #         config=config,
    #         args=args,
    #         is_train=is_train
    #     )
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=config.batch_size,
    #         num_workers=config.num_worker,
    #         shuffle=config.shuffle,
    #         pin_memory=config.pin_memory,
    #     )
    #     return dataloader

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
            pin_memory  =config.pin_memory
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
            pin_memory  =config.pin_memory
        )
        return dataloader

    def _print_learning_info(self, dataloader: DataLoader):
        print(f'WanDB? {self.config.data.wandb}')
        print(f"Random Seed : {self.config.data.random_seed}")
        print(f"Epoch : {self.config.data.train.epoch}")
        print(f"Batch size : {dataloader.batch_size}")
        print(f"Image size : {self.config.data.dataset.train.image_size}")
        print(f"num of classes : {self.config.data.dataset.train.num_classes}")
        print(f"pin memory : {dataloader.pin_memory}")
        print(f"num workers : {dataloader.num_workers}")

        if self.config.data.mbs.enable:
            print(f">>> micro batch size : {self.config.data.micro_batch_size}")
            print(f"*** Training VGG-{self.config.data.model.version} with MBS")
        else:
            print(f"*** Training VGG-{self.config.data.model.version} (Baseline) ***")

    def _check_before_running(self, dataloader: DataLoader):
        if self.config.data.dataset.train.train_batch != dataloader.batch_size:
            raise ValueError("Batch size is not equal!")
        if self.config.data.dataset.train.pin_memory != dataloader.pin_memory:
            raise ValueError("Status of pin memory is not equal!")
        if self.config.data.dataset.train.num_workers != dataloader.num_workers:
            raise ValueError("Num of wokers is not equal!")

        if self.config.data.wandb.wandb:
            name = None
            if self.config.data.mbs.enable:
                name = f'VGG-{self.config.data.model.version}(mbs)'
            else:
                name = f'VGG-{self.config.data.model.version}(baseline)'

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
        self.model = select_model(
            self.config.data.model.normbatch,
            self.config.data.model.version,
            self.config.data.dataset.train.num_classes
        ).to(device)
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
                # bn_factor=self.args.bn
            ).get_trainer()

            for epoch in range(self.config.data.train.epoch):

                start = time.perf_counter()
                mbs_trainer.train()
                end = time.perf_counter()
                self.epoch_avg_time.append( end - start )

                acc = self._val_accuracy(epoch, val_dataloader, device)

                if self.config.data.wandb.wandb:
                    wandb.log( {'train loss': mbs_trainer.get_loss()}, step=epoch )
                    wandb.log( {'epoch time' : sum(self.epoch_avg_time)/len(self.epoch_avg_time)}, step=epoch)
                print(  f"[{epoch+1}/{self.config.data.train.epoch}]",
                        f"acc:{acc:.2f}",
                        f"epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}",
                        f"loss : {mbs_trainer.get_loss()}",
                    )
        else:
            for epoch in range(self.config.data.train.epoch):
                epoch_avg_loss, epoch_time = self._train_epoch(epoch, train_dataloader, device)
                acc = self._val_accuracy(epoch, val_dataloader, device)
                print(  f"acc:{acc:.2f}",
                        f"epoch time: {epoch_time}",
                        f"loss : {epoch_avg_loss}",
                        # end='\r',
                    )

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

        print("\n")

    def _train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        # Define check performance vars
        losses = []

        # if self.config.data.microbatchstream.enable:
        #     dataloader_len = dataloader.micro_len()
        # else:
        dataloader_len = len(dataloader)

        epoch_start = time.perf_counter()
        self.model.train()
        for idx, (image, label) in enumerate(dataloader):
            print(f'[{epoch + 1}/{self.config.data.train.epoch}] [{idx+1}/{dataloader_len}]', end='\r')

            input: Tensor = image.to(device)
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

        # if self.config.data.mbs.enable:
        #     losses *= math.ceil(self.args.batch_size / self.args.micro_batch_size) 
        # losses /= idx

        if self.config.data.wandb:
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

                # # rank 5
                # _, rank5 = output.topk(5, 1, True, True)
                # rank5 = rank5.t()
                # correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))

                # for k in range(6):
                #     correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)
                # correct_top5 += correct_k.item()
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        accuracy = 100 * ( correct_top1 / total )
        print( f"\nAccuracy : {accuracy:.2f}" )

        if self.config.data.wandb.wandb:
            wandb.log( {'acc': accuracy}, step=epoch )
            wandb.log( {'inf time': inference_time}, step=epoch)

        return accuracy
        # if self.args.wandb:
        #     wandb.log( {'top-1': 100 * ( correct_top1 / total )}, step=epoch )
        #     wandb.log( {'top-5': 100 * ( correct_top5 / total )}, step=epoch )
        # self.val_accuracy[epoch + 1] = {
        #     'top-1' : 100 * ( correct_top1 / total ),
        #     'top-5' : 100 * ( correct_top5 / total )
        # }
        # if self.max_top1 < (100 * (correct_top1 / total)):
        #     self.max_top1 = (100 * (correct_top1 / total))
        # if self.max_top5 < (100 * (correct_top5 / total)):
        #     self.max_top5 = (100 * (correct_top5 / total))
        # print(
        #     'top-1 :',
        #     format(100 * ( correct_top1 / total ), ".2f"),
        #     'top-5 :',
        #     format(100 * ( correct_top5 / total ), ".2f"),
        #     end='\r'
        # )

    # def _epoch_writer(
    #     self,
    #     epoch: int,
    #     epochs: int,
    #     train_time: float,
    #     train_iter: int,
    #     epoch_time: float,
    #     epoch_iter: int,
    #     epoch_loss: float
    # ) -> None:
    #     self.train_loss[epoch + 1] = epoch_loss
    #     self.epoch_time[epoch + 1] = epoch_time / epoch_iter

    #     print(
    #         f"\n[{epoch+1}/{epochs}]",
    #         "train time :",
    #         format(train_time / train_iter, ".3f") + "s",
    #         "epoch time :",
    #         format(epoch_time / epoch_iter, ".3f") + "s",
    #         end=" ",
    #     )
    #     print(
    #         # f"loss : {self.train_loss[epoch+1]}",
    #         f"loss : ",
    #         format(self.train_loss[epoch + 1], ".4f"),
    #         end=' '
    #     )

    # def _grad_magnitude(self, epoch : int) -> None:
    #     self.magnitude[epoch + 1] = {}
    #     for layer, para in self.model.named_parameters():
    #         magnitude : torch.Tensor = torch.sum( para.grad )
    #         self.magnitude[epoch + 1][layer] = magnitude.item()

def train(config: ConfigParser, args):
    trainer = VGG(config, args)
    trainer.train()