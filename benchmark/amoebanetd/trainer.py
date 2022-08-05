import math, random
import json
import time
from typing import List

# Get dataset and model
from dataloader import get_dataset
from amoebanet import amoebanetd

# PyTorch
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device

# Get Micro-batch streaming.
from mbs.micro_batch_streaming import MicroBatchStreaming

import wandb

class AmoebaNetTrainer:
    def __init__(self, config: ConfigParser, args) -> None:
        self.config = config
        self.args = args

        self.train_loss = {}
        self.train_time = {}
        self.val_accuracy = {}
        self.epoch_avg_time = []

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _save_log(cls, log, is_mbs, batch_size) -> None:
        ensure_dir("./loss/")
        with open(f"./loss/enet_mbs_{is_mbs}_{batch_size}.json", "w") as file:
            json.dump(log, file, indent=4)

    @classmethod
    def _get_data_loader(cls, dataset_config: DotDict, is_train: bool) -> DataLoader:
        dataloader = get_dataset(
            path=dataset_config.path,
            dataset_type=dataset_config.type,
            args=dataset_config,
            is_train=is_train
        )
        return dataloader

    def _print_learning_info(self, dataloader: DataLoader):
        print(f'WanDB? {self.config.data.wandb.wandb}')
        print(f"Random Seed : {self.config.data.train.seed}")
        print(f"Epoch : {self.config.data.train.epoch}")
        print(f"Batch size : {dataloader.batch_size}")
        print(f"Image size : {self.config.data.dataset.train.image_size}")
        print(f"num of classes : {self.config.data.dataset.train.num_classes}")
        print(f"pin memory : {dataloader.pin_memory}")
        print(f"num workers : {dataloader.num_workers}")
        print(f"num layers : {self.config.data.model.num_layers}")
        print(f"num filters : {self.config.data.model.num_filters}")

        if self.config.data.mbs.enable:
            print(f">>> micro batch size : {self.config.data.mbs.micro_batch_size}")
            print(f"*** Training AmoebaNet-D ({self.config.data.model.num_layers}, {self.config.data.model.num_filters}) with MBS")
        else:
            print(f"*** Training AmoebaNet-D ({self.config.data.model.num_layers}, {self.config.data.model.num_filters}) (Baseline) ***")

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
                name = f'AmoebaNet-D ({self.config.data.model.num_layers}, {self.config.data.model.num_filters})(mbs)'
            else:
                name = f'AmoebaNet-D ({self.config.data.model.num_layers}, {self.config.data.model.num_filters})(baseline)'

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
        train_dataloader = self._get_data_loader(self.config.data.dataset.train, True)
        val_dataloader = self._get_data_loader(self.config.data.dataset.test, False)

        self._print_learning_info(train_dataloader)
        self._check_before_running(train_dataloader)

        # Define model
        self.model = amoebanetd(
            num_classes=self.config.data.dataset.train.num_classes,
            num_layers=self.config.data.model.num_layers,
            num_filters=self.config.data.model.num_filters
        ).to(device)

        # Define loss function.
        self.criterion = nn.CrossEntropyLoss().to(device)

        # Define SGD optimizer
        self.opt = torch.optim.SGD( 
            self.model.parameters(), 
            lr=self.config.data.optimizer.lr,
            momentum=self.config.data.optimizer.mometum,
            weight_decay=self.config.data.optimizer.decay,
            # nesterov=True
        )
        # self.opt = torch.optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.config.data.optimizer.lr,
        #     momentum=self.config.data.optimizer.mometum,
        #     weight_decay=self.config.data.optimizer.decay,
        # )

        # Define Scheduler
        steps = int(len(train_dataloader) * 2)
        lr_multiplier = max(
            1.0, 
            int(self.config.data.dataset.train.train_batch / 2) / self.config.data.model.num_filters
        )

        def gradual_warmup_linear_scaling(step: int) -> float:
            epoch = step / steps

            # Gradual warmup
            warmup_ratio = min(4.0, epoch) / 4.0
            multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

            # print(
            #         step, steps, epoch, lr_multiplier, warmup_ratio, multiplier
            #     )

            if epoch < 30:
                return 1.0 * multiplier
            elif epoch < 60:
                return 0.5 * multiplier
            elif epoch < 80:
                return 0.1 * multiplier
            return 0.05 * multiplier

        self.scheduler = LambdaLR(self.opt, lr_lambda=gradual_warmup_linear_scaling)

        if self.config.data.mbs.enable:
            mbs_trainer, self.model = MicroBatchStreaming(
                dataloader=train_dataloader,
                model=self.model,
                criterion=self.criterion,
                optimizer=self.opt,
                lr_scheduler=self.scheduler,
                device_index=self.config.data.gpu.device,
                batch_size=self.config.data.dataset.train.train_batch,
                micro_batch_size=self.config.data.mbs.micro_batch_size,
            ).get_trainer()

            for epoch in range(self.config.data.train.epoch):
                print( f"[{epoch+1}/{self.config.data.train.epoch}]" )
                # Train
                start = time.perf_counter()
                mbs_trainer.train()
                end = time.perf_counter()
                self.epoch_avg_time.append( end - start )

                # Validation
                acc = self._val_accuracy(epoch, val_dataloader, device)

                # Update status to WandB
                if self.config.data.wandb.wandb:
                    wandb.log( {'train loss': mbs_trainer.get_loss()}, step=epoch )
                    wandb.log( {'epoch time' : sum(self.epoch_avg_time)/len(self.epoch_avg_time)}, step=epoch)
                print(  f"acc:{acc:.2f}",
                        f"epoch time: {sum(self.epoch_avg_time)/len(self.epoch_avg_time)}",
                        f"loss : {mbs_trainer.get_loss()}",
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
            self.scheduler.step()
            losses.append( loss.detach().item() )
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_avg_loss = sum( losses ) / len( losses )

        if self.config.data.wandb.wandb:
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
        if self.config.data.wandb.wandb:
            wandb.log( {'acc': accuracy}, step=epoch )
            wandb.log( {'inf time': inference_time}, step=epoch)

        return accuracy

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


def train(config: ConfigParser, args):
    trainer = AmoebaNetTrainer(config, args)
    trainer.train()
