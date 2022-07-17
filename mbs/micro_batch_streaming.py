import math
import tqdm
import wandb
from typing import List, Optional, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from torch.cuda import (
    Stream,
    stream,
    current_stream,
    default_stream,
)

from .wrap_model import MBSBatchNorm

class _MBSBlock:
    def __init__(
        self,
        debug: Optional[str] = None
    ) -> None:
        self._init = True
        self._bn = False

        self.debug_msg = debug

    def _debug(self):
        if self.debug_msg == 'early stop':
            for name, para in self.module.named_parameters():
                print(name, '*'*30)
                print(para.data)
            print("\n\n\n")
            raise Exception(f"[MBS Debug] early stop")

class MicroBatchStreaming(_MBSBlock):
    def __init__(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        warmup_factor: Optional[int] = None,
        device_index: Optional[int] = None,
        batch_size: int = 1,
        micro_batch_size: int = 1,
        bn_factor: bool = False,
        pretrain: bool = False,
        wandb: bool = False,
        debug: Optional[str] = None,
    ) -> None:
        super().__init__(debug=debug)
        self.device = torch.device(f'cuda:{device_index}')

        self.dataloader = dataloader
        if bn_factor:
            print("[MBS] Consider BatchNorm layers")
            self.module = MBSBatchNorm.wrap_batch_norm(model, self).to(self.device)
        else:
            print("[MBS] Does not consider BatchNorm layers")
            self.module = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.pretrain = pretrain
        if self.pretrain:
            print("[MBS] ViT Pretrain Mode")
        self.wandb = wandb

        ''' Warmup arguments '''
        self.scheduler = lr_scheduler
        self.warmup_factor = warmup_factor
        if self.scheduler is None or self.warmup_factor is None:
            print("[MBS] Does not consider Scheduler or Warmup algorithm")
        else:
            print("[MBS] Consider Scheduler or Warmup algorithm")

        self.batch_size = batch_size
        self.micro_batch = micro_batch_size
        self.chunks = math.ceil( self.batch_size / self.micro_batch )

        # self.cpy_strm = Stream(self.device)
        # self.cmp_strm = Stream(self.device)
        self.debug_msg = debug

    def get_model(self):
        return self.module

    def get_trainer(self):
        return self, self.module

    def train(self, current_epoch:int, total_epoch:int):
        data0: torch.Tensor
        data1: torch.Tensor
        self.epoch_loss = 0
        self.module.train()
        epoch_iterator = tqdm.tqdm(self.dataloader, desc="Training (X / X Steps) (lr=X.X) (loss=X.X)",dynamic_ncols=True,)
        
        for idx, (data0, data1) in enumerate(epoch_iterator):
            mini_loss = 0
            # size = self.micro_batch
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )
                # size = math.ceil( data0.size(0) / chunks )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                self._bn = (jdx + 1) == chunks

                input = i.to(self.device)
                label = l.to(self.device)

                if self.pretrain:
                    loss = self.module( input, label )
                    loss: torch.Tensor = torch.div( loss, chunks )
                    mini_loss += loss.detach().item()
                    loss.backward()
                else:
                    output, _ = self.module( input )
                    # loss: torch.Tensor = self.criterion( output, label ) / chunks
                    loss: torch.Tensor = self.criterion( output, label )
                    loss: torch.Tensor = torch.div( loss, chunks )
                    mini_loss += loss.detach().item()
                    loss.backward()

            # for cidx in range(chunks):
            #     with stream( self.cpy_strm ):
            #         lower = cidx * size
            #         upper = ( cidx + 1 ) * size
            #         da0: torch.Tensor = data0[lower:upper].to(self.device)
            #         da1: torch.Tensor = data1[lower:upper].to(self.device)

            #     self.cpy_strm.wait_stream( self.cmp_strm )
            #     self.cmp_strm.wait_stream( self.cpy_strm )
            #     with stream( self.cmp_strm ):
            #         da0.record_stream( self.cmp_strm )
            #         da1.record_stream( self.cmp_strm )
            #         output: torch.Tensor = self.module( da0 )
            #         loss: torch.Tensor = self.criterion( output, da1 ) / chunks
            #         loss.backward()
            #         mini_loss += loss.detach().item()

            self.optimizer.step()
            if self.scheduler is not None:
            # print("\t\t\t\tScheduler Doing!", end='\r')
                self.scheduler.step()
            self.epoch_loss += mini_loss

            self._init = self._init and False
            
            self._debug()
            if self.wandb:
                wandb.log( {'train loss': mini_loss, 'learning rate': self.scheduler.get_last_lr()[0], 'epoch': current_epoch, 'batch': idx})

            epoch_iterator.set_description(
                    "Training (%d / %d Steps) (lr=%2.5f) (loss=%2.5f)" % (current_epoch+1, total_epoch, self.scheduler.get_last_lr()[0], mini_loss))

            
    def get_loss(self):
        return self.epoch_loss / self.dataloader.__len__()


class MBSSegmentation(MicroBatchStreaming):
    def __init__(
        self, 
        dataloader: DataLoader, 
        model: Module, 
        criterion: Module, 
        optimizer: Optimizer, 
        lr_scheduler: _LRScheduler = None, 
        warmup_factor: Optional[int] = None, 
        device_index: Optional[int] = None, 
        batch_size: int = 1, 
        micro_batch_size: int = 1, 
        bn_factor: bool = False,
        debug: Optional[str] = None
    ) -> None:
        super().__init__(
            dataloader, 
            model, 
            criterion, 
            optimizer, 
            lr_scheduler=lr_scheduler, 
            warmup_factor=warmup_factor, 
            device_index=device_index, 
            batch_size=batch_size, 
            micro_batch_size=micro_batch_size, 
            bn_factor=bn_factor,
            debug=debug
        )

    def train(self):
        data0: torch.Tensor
        data1: torch.Tensor

        pred: torch.Tensor
        loss: torch.Tensor
        dice: torch.Tensor

        self.epoch_loss = 0
        self.epoch_dice = []
        self.module.train()
        for idx, (data0, data1) in enumerate( self.dataloader ):
            mini_loss = 0
            mini_dice = []
            chunks = self.chunks
            if data0.size(0) != self.batch_size:
                chunks = math.ceil( data0.size(0) / self.micro_batch )

            chunk_data0 = data0.chunk( chunks )
            chunk_data1 = data1.chunk( chunks )

            self.optimizer.zero_grad()

            for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                self._bn = (jdx + 1) == chunks

                input = i.to(self.device)
                mask = l.to(self.device)

                pred = self.module( input )
                loss, dice = self.criterion( pred, mask )
                # loss /= chunks
                # dice /= chunks
                mini_loss += loss.detach().item()
                mini_dice.append( dice.detach().item() )
                loss.backward()

            self.optimizer.step()
            self.epoch_loss += mini_loss
            self.epoch_dice.append( sum(mini_dice)/len(mini_dice) )

            self._init = self._init and False

            if self.warmup_factor is not None and self.scheduler is not None:
                if idx <= self.warmup_factor:
                    self.scheduler.step()

            self._debug()

    def get_dice(self):
        return sum(self.epoch_dice) / len(self.epoch_dice)

    def train_profile(self):
        data0: torch.Tensor
        data1: torch.Tensor

        pred: torch.Tensor
        loss: torch.Tensor
        dice: torch.Tensor

        self.epoch_loss = 0
        self.epoch_dice = []
        self.module.train()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for idx, (data0, data1) in enumerate( self.dataloader ):
                mini_loss = 0
                mini_dice = []
                chunks = self.chunks
                if data0.size(0) != self.batch_size:
                    chunks = math.ceil( data0.size(0) / self.micro_batch )

                chunk_data0 = data0.chunk( chunks )
                chunk_data1 = data1.chunk( chunks )

                self.optimizer.zero_grad()

                for jdx, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
                    self._bn = (jdx + 1) == chunks

                    input = i.to(self.device)
                    mask = l.to(self.device)

                    pred = self.module( input )
                    loss, dice = self.criterion( pred, mask )
                    # loss /= chunks
                    # dice /= chunks
                    mini_loss += loss.detach().item()
                    mini_dice.append( dice.detach().item() )
                    loss.backward()

                self.optimizer.step()
                self.epoch_loss += mini_loss
                self.epoch_dice.append( sum(mini_dice)/len(mini_dice) )

                self._init = self._init and False

                if self.warmup_factor is not None and self.scheduler is not None:
                    if idx <= self.warmup_factor:
                        self.scheduler.step()

                if idx > 1:
                    break
        prof: profile
        prof.export_chrome_trace("./profiling_mbs.json")

# def micro_batch_streaming(
#     dataloader: DataLoader,
#     model: Module,
#     criterion: Module,
#     optimizer: Optimizer,
#     lr_scheduler: _LRScheduler = None,
#     dev: Union[ torch.device, int ] = 0,
#     micro_batch_size: int = 4,
# ):
#     epoch_loss = 0
#     for _, (data0, data1) in enumerate( dataloader ):
#         data0: torch.Tensor
#         data1: torch.Tensor

#         mini_loss = 0
#         chunks = math.ceil( dataloader.batch_size / micro_batch_size )
#         if data0.size(0) != dataloader.batch_size:
#             chunks = math.ceil( data0.size(0) / micro_batch_size )

#         chunk_data0 = data0.chunk( chunks )
#         chunk_data1 = data1.chunk( chunks )

#         optimizer.zero_grad()

#         for _, (i, l) in enumerate( zip(chunk_data0, chunk_data1) ):
#             input = i.to(dev)
#             label = l.to(dev)
#             output: torch.Tensor = model( input )
#             loss: torch.Tensor = criterion( output, label ) / chunks
#             loss.backward()
#             mini_loss += loss.detach().item()

#         optimizer.step()
#         epoch_loss += mini_loss
#     return epoch_loss / len(dataloader)
