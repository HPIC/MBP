import random
import time
import os

import wandb

# Get dataset and model
from models.unet import unet_3156

# PyTorch
import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader
import torchvision.utils as vutils

from util.config_parser import ConfigParser, DotDict
from util.util import ensure_dir, ensure_file_can_create, prepare_device
from util.losses import DiceBCELoss, DiceLoss
from util.dataloder import get_dataloader


# Get Micro-batch streaming.
from mbs import MBSSegmentation

class UNetTrainer:
    def __init__(self, config: ConfigParser, args) -> None:
        self.config = config
        self.args = args

    @classmethod
    def _save_state_dict(cls, state_dict: dict, path: str) -> None:
        ensure_file_can_create(path)
        torch.save(state_dict, path)

    @classmethod
    def _get_train_dataloader(cls, config: DotDict) -> DataLoader:
        return get_dataloader(
            root        =config.path,
            train       =True,
            batch_size  =config.train_batch,
            num_workers =config.num_workers,
            shuffle     =config.shuffle,
            pin_memory  =config.pin_memory,
        )

    @classmethod
    def _get_test_dataloader(cls, config: DotDict) -> DataLoader:
        return get_dataloader(
            root        =config.path,
            train       =False,
            batch_size  =config.test_batch,
            num_workers =config.num_workers,
            shuffle     =config.shuffle,
            pin_memory  =config.pin_memory,
        )

    def save_images(
        self, origin: Tensor, pred: Tensor, mask: Tensor, epoch: int, path: str
    ):
        origin_path = f'{path}/origin-{epoch}.png'
        pred_path = f"{path}/pred-{epoch}.png"
        mask_path = f"{path}/mask-{epoch}.png"

        vutils.save_image( origin.float(), origin_path, normalize=True )
        vutils.save_image( pred.float(), pred_path, normalize=True )
        vutils.save_image( mask.float(), mask_path, normalize=True )

    def _create_directory(self, path):
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)

    def _create_directories(self, path_list):
        for path in path_list:
            self._create_directory(path)

    def _print_learning_info(self):
        print(f"Random Seed : {self.env_config['Random_seed']}")
        print(f"Epoch : {self.config.data.train.epoch}")
        print(f"Batch size : {self.env_config['Dataset']['batch_size']}")
        print(f"Image size : {self.env_config['Dataset']['image_size']}")
        print(f"pin memory : {self.env_config['Dataset']['pin_memory']}")
        print(f"num workers : {self.env_config['Dataset']['num_workers']}")

        print(f"*** {self.env_config['Name']} ***")

    def _check_before_running(self, dataloader: DataLoader):
        if self.config.data.dataset.train.train_batch != dataloader.batch_size:
            raise ValueError("Batch size is not equal!")
        if self.config.data.dataset.train.pin_memory != dataloader.pin_memory:
            raise ValueError("Status of pin memory is not equal!")
        if self.config.data.dataset.train.num_workers != dataloader.num_workers:
            raise ValueError("Num of wokers is not equal!")

        if self.config.data.wandb.enable:
            tags = []
            tags.append( f'{self.args.server}')
            tags.append( f'{self.env_config["Model"]}')
            tags.append( f'batch {dataloader.batch_size}' )
            tags.append( f'image {self.config.data.dataset.train.image_size}' )
            tags.append( f'seed {self.config.data.train.seed}' )

            if self.config.data.mbs.enable:
                tags.append( f'mbs {self.config.data.mbs.micro_batch_size}' )

            wandb.init(
                project='mbs_aaai23',
                entity='xypiao97',
                name=f'{self.env_config["Name"]}',
                tags=tags,
                config=self.env_config
            )

    def _wandb_config(self, config: DotDict):
        env_config = {
            "Name": None,
            "Random_seed": None,
            "Model": "U-Net",
            "Dataset": {
                "name": None,
                "batch_size": None,
                "image_size": None,
                "num_workers": None,
                "pin_memory": None,
            },
            "Optimizer": {
                "name": None,
                "learning_rate": None,
                "momentum": None,
                "weight_decay": None,
            },
            "MBS": {
                "micro_size": None
            }
        }

        name_run = f"{config.dataset.train.train_batch} w"

        env_config["Random_seed"] = config.train.seed

        env_config["Dataset"]["name"] = config.dataset.train.type
        env_config["Dataset"]["batch_size"] = config.dataset.train.train_batch
        env_config["Dataset"]["image_size"] = config.dataset.train.image_size
        env_config["Dataset"]["num_workers"] = config.dataset.train.num_workers
        env_config["Dataset"]["pin_memory"] = config.dataset.train.pin_memory

        name_optim = str(self.opt.__class__)
        name_optim = name_optim.split(".")
        name_optim = name_optim[-1].replace("'>", "")
        env_config["Optimizer"]["name"] = name_optim
        env_config["Optimizer"]["learning_rate"] = self.opt.param_groups[0]['lr']
        env_config["Optimizer"]["weight_decay"] = self.opt.param_groups[0]['weight_decay']
        try:
            env_config["Optimizer"]["momentum"] = self.opt.param_groups[0]['momentum']
        except:
            env_config["Optimizer"]["momentum"] = None

        if config.mbs.enable:
            name_run += f" MBS={config.mbs.micro_batch_size}"
            env_config["MBS"]["micro_size"] = config.mbs.micro_batch_size
            self.seg_path = f"./results/mbs/{name_run}/{config.train.seed}"
            self.chk_path = f"./checkpoint/mbs/{name_run}/{config.train.seed}"
        else:
            name_run += f"o MBS"
            self.seg_path = f"./results/base/{name_run}/{config.train.seed}"
            self.chk_path = f"./checkpoint/base/{name_run}/{config.train.seed}"

        env_config["Name"] = name_run

        return env_config


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

        # build model and loss, optimizer
        self.model = unet_3156().to(device)
        self.criterion = DiceLoss().to(device)
        self.opt = torch.optim.SGD( 
            self.model.parameters(), 
            lr=self.config.data.optimizer.lr,
            momentum=self.config.data.optimizer.mometum,
            weight_decay=self.config.data.optimizer.decay,
        )

        self.env_config = self._wandb_config(self.config.data)
        self._check_before_running(train_dataloader)
        self._print_learning_info()
        self._create_directories( [self.chk_path, self.seg_path] )

        if self.config.data.mbs.enable:
            mbs_trainer, self.model = MBSSegmentation(
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
                train_loss, train_acc =mbs_trainer.train()
                end = time.perf_counter()
                epoch_time = end - start

                # Validation
                test_loss, test_acc = self._val_accuracy(epoch, val_dataloader, device)

                # train_loss = mbs_trainer.get_loss()
                # Update status to WandB
                if self.config.data.wandb.enable:
                    wandb.log( {'train loss': train_loss}, step=epoch )
                    wandb.log( {'train acc': train_acc}, step=epoch )
                    wandb.log( {'epoch time' : epoch_time}, step=epoch)

                print(
                        f"[{epoch}/{self.config.data.train.epoch}]",
                        f"[{train_dataloader.__len__()}/{train_dataloader.__len__()}]",
                        f"train acc:{epoch_avg_dice:.2f}",
                        f"test acc:{test_acc:.2f}",
                        f"epoch time: {epoch_time}",
                        f"train loss : {epoch_avg_loss}",
                        f"test loss: {test_loss}"
                        # end='\r',
                    )

                self._save_state_dict( 
                    self.model, 
                    f"{self.chk_path}/para.pth" 
                )
        else:
            for epoch in range(self.config.data.train.epoch):
                # Train
                epoch_avg_loss, epoch_avg_dice, epoch_time = self._train_epoch(epoch, train_dataloader, device)

                # Validation
                test_loss, test_acc = self._val_accuracy(epoch, val_dataloader, device)

                # Update status to WandB
                print(  
                        f"[{epoch}/{self.config.data.train.epoch}]",
                        f"[{train_dataloader.__len__()}/{train_dataloader.__len__()}]",
                        f"train acc:{epoch_avg_dice:.2f}",
                        f"test acc:{test_acc:.2f}",
                        f"epoch time: {epoch_time}",
                        f"train loss : {epoch_avg_loss}",
                        f"test loss: {test_loss}"
                        # end='\r',
                    )

                self._save_state_dict( 
                    self.model, 
                    f"{self.chk_path}/para.pth" 
                )


    def _train_epoch(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        losses = []
        dices = []
        loss: Tensor
        dice: Tensor

        epoch_start = time.perf_counter()
        self.model.train()
        for idx, (input, masks) in enumerate(dataloader):
            print(f'[{epoch + 1}/{self.config.data.train.epoch}] [{idx+1}/{len(dataloader)}]', end='\r')
            input: Tensor = input.to(device)
            masks: Tensor = masks.to(device)

            # print(input.shape, masks.shape)

            output: Tensor = self.model( input )
            loss, dice = self.criterion( output, masks )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            losses.append( loss.detach().item() )
            dices.append( dice.detach().item() )
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_avg_loss = sum( losses ) / len( losses )
        epoch_avg_dice = sum( dices ) / len( dices )

        if self.config.data.wandb.enable:
            wandb.log( {'train loss': epoch_avg_loss}, step=epoch )
            wandb.log( {'train acc': epoch_avg_dice}, step=epoch )
            wandb.log( {'epoch time' : epoch_time}, step=epoch)

        return epoch_avg_loss, epoch_avg_dice, epoch_time


    def _val_accuracy(
        self, epoch: int, dataloader: DataLoader, device: torch.device
    ) -> None:
        losses = []
        dices = []
        loss: Tensor
        dice: Tensor
    
        self.model.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            for _, (input, masks) in enumerate(dataloader):
                input: Tensor = input.to(device)
                masks: Tensor = masks.to(device)

                preds : Tensor = self.model(input)
                loss, dice = self.criterion(preds, masks)

                losses.append( loss.item() )
                dices.append( dice.item() )

            self.save_images(input, preds, masks, epoch, path=self.seg_path)

        end_time = time.perf_counter()
        inference_time = end_time - start_time

        avg_loss = sum(losses) / len(losses)
        avg_dice = sum(dices) / len(dices)

        if self.config.data.wandb.enable:
            wandb.log( {'test loss': avg_loss}, step=epoch )
            wandb.log( {'test acc': avg_dice}, step=epoch )
            wandb.log( {'inf time': inference_time}, step=epoch)

        return avg_loss, avg_dice




def train(config: ConfigParser, args):
    trainer = UNetTrainer(config, args)
    trainer.train()
