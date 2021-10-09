from typing import List, Tuple, Union, Optional, Sequence, Any

import math
import warnings
from queue import Queue

from torch.utils.data import DataLoader
from torch import Tensor
import torch
from torch._utils import ExceptionWrapper
from torch.utils.data import (
    IterableDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    Dataset
)
from torch.utils.data.dataloader import (
    T_co, T, _worker_init_fn_t, _collate_fn_t,
    _utils,
    _DatasetKind,
    _BaseDataLoaderIter,
    _SingleProcessDataLoaderIter,
    _MultiProcessingDataLoaderIter
)
from torch.cuda import (
    Stream,
    StreamContext,
    stream,
    default_stream
)

from mbs.types import (
    MultiTensors,
    ChunkTensor,
    ChunkTensors,
    ChunkTensororTensors
)

from threading import Thread

class MBSDataloader(DataLoader):
    def __init__(self,
        micro_batch_size: int,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(MBSDataloader, self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
        )
        self.micro_batch_size = micro_batch_size

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.micro_batch_size == None:
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                self.check_worker_number_rationality()
                return _MultiProcessingDataLoaderIter(self)
        else:
            if self.num_workers == 0:
                return _MBSSingleProcessDataLoaderIter(self, self.micro_batch_size)
            else:
                self.check_worker_number_rationality()
                return _MBSMultiProcessingDataLoaderIter(self, self.micro_batch_size)

    def micro_len(self):
        total_num_dataset = len(self.dataset)
        std_micro_len = ( total_num_dataset // self.micro_batch_size ) + \
            math.ceil( (total_num_dataset % self.micro_batch_size) / self.micro_batch_size )

        return std_micro_len

    @classmethod
    def wrap_dataloader(
        cls, dataloader: DataLoader, shuffle: bool, micro_batch_size: Optional[int],
    ) -> 'MBSDataloader':
        if not isinstance(dataloader, DataLoader):
            raise TypeError( 'DataLoader Type Error, only torch::dataloader type.' )
        # if not isinstance(device, (torch.device, int)):
        #     raise TypeError( 'Device Type Error, only torch::device, int type.' )
        # if not isinstance(depth, int):
        #     raise TypeError( 'Device Type Error, only torch::device, int type.' )

        mbs_dataloader : MBSDataloader
        mbs_dataloader = MBSDataloader(
            micro_batch_size,
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=shuffle,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )

        return mbs_dataloader


class _MBSBaseDataLoaderIter(_BaseDataLoaderIter):
    def __init__(
        self, loader: DataLoader, micro_batch_size: int
    ) -> None:
        super(_MBSBaseDataLoaderIter, self).__init__(loader)
        self.micro_batch_size = micro_batch_size
        self.batch_size = loader.batch_size
        self.chunks = math.ceil( self.batch_size / self.micro_batch_size )

        self.cur_iter = 0
        self.micro_iter = 0
        self.micro_data = None
        self.num_tensor = 0

    def _chunk(self, data: List[Tensor] ) -> List[Tensor] :
        r'''
            Args:
                data : List[Tensor]
                    [ input tensor, target tensor ]

            Returns:
                data : List[Tensor]
                    [ List[input Tensor], List[target Tensor] ]
        '''
        chunks: int = self.chunks
        if data[0].size(0) < chunks:
            chunks = math.ceil( data[0].size(0) / self.micro_batch_size )

        chunk_data = [ None for i in range(self.num_tensor) ]

        for i, tensor in enumerate(data):
            chunk_data[i] = tensor.chunk(chunks)

        return chunks, chunk_data

    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self.cur_iter == 0:
                if self._sampler_iter is None:
                    self._reset()
                data = self._next_data()

                # Make micro-batch dataset
                self.num_tensor = len(data)
                self.micro_iter, self.micro_data = self._chunk(data)

                self._num_yielded += 1
                if self._dataset_kind == _DatasetKind.Iterable and \
                        self._IterableDataset_len_called is not None and \
                        self._num_yielded > self._IterableDataset_len_called:
                    warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                                "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                    self._num_yielded)
                    if self._num_workers > 0:
                        warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                    "IterableDataset replica at each worker. Please see "
                                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                    warnings.warn(warn_msg)

            data = [ self.micro_data[i][self.cur_iter] for i in range(self.num_tensor) ]
            self.cur_iter += 1
            if self.cur_iter == self.micro_iter:
                self.cur_iter = 0
            return data


class _MBSMultiProcessingDataLoaderIter(_MBSBaseDataLoaderIter, _MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader, micro_batch_size: int) -> None:
        super().__init__(loader, micro_batch_size)


class _MBSSingleProcessDataLoaderIter(_MBSBaseDataLoaderIter, _SingleProcessDataLoaderIter):
    def __init__(self, loader: DataLoader, micro_batch_size: int) -> None:
        super().__init__(loader, micro_batch_size)


'''
    Test Code to apply multi-stream based dataloader.

'''



class TestMBSDatalaoder(DataLoader):
    def __init__(
        self,
        mbs_block,
        device: torch.device,
        micro_batch_size: int,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super(TestMBSDatalaoder, self).__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=pin_memory,
        )
        self.device = device
        self.micro_batch = micro_batch_size
        self.mbs_block = mbs_block

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _TestMBSSingleProcessDataLoaderIter(
                self, self.mbs_block
            )
        else:
            self.check_worker_number_rationality()
            return _TestMBSMultiProcessingDataLoaderIter(
                self, self.mbs_block
            )

    def __len__(self) -> int:
        return math.ceil(self.dataset.__len__() / self.micro_batch )

    @classmethod
    def wrap_dataloader(
        cls,
        mbs_block,
        dataloader: DataLoader,
        shuffle: bool,
        device: torch.device,
        micro_batch_size: Optional[int]
    ) -> 'TestMBSDatalaoder':
        if not isinstance(dataloader, DataLoader):
            raise TypeError( 'DataLoader Type Error, only torch::dataloader type.' )
        # if not isinstance(device, (torch.device, int)):
        #     raise TypeError( 'Device Type Error, only torch::device, int type.' )
        # if not isinstance(depth, int):
        #     raise TypeError( 'Device Type Error, only torch::device, int type.' )

        mbs_dataloader : TestMBSDatalaoder
        mbs_dataloader = TestMBSDatalaoder(
            mbs_block=mbs_block,
            device=device,
            micro_batch_size=micro_batch_size,
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=shuffle,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )

        return mbs_dataloader

class _TestMBSBaseDataLoaderIter(_BaseDataLoaderIter):
    def __init__(
        self,
        mbs_block,
        loader: DataLoader,
    ) -> None:
        super(_TestMBSBaseDataLoaderIter, self).__init__(loader)
        # self.batch_size = micro_batch_size
        self.mbs_block = mbs_block
        self.call_counter = 0
        self.data = None
        self.num = 0

    def __next__(self) -> Any:
        self._update()
        with torch.autograd.profiler.record_function(self._profile_name):
            if (self.call_counter - 1) % self.mbs_block.chunks == 0:
                if self._sampler_iter is None:
                    self._reset()
                self.data = self._next_data()
                self.num = len(self.data)

                self._num_yielded += 1
                if self._dataset_kind == _DatasetKind.Iterable and \
                        self._IterableDataset_len_called is not None and \
                        self._num_yielded > self._IterableDataset_len_called:
                    warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                                "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                    self._num_yielded)
                    if self._num_workers > 0:
                        warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                    "IterableDataset replica at each worker. Please see "
                                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
                    warnings.warn(warn_msg)

            if self.data[0].size(0) != self.mbs_block.micro_batch_size:
                chunks = math.ceil( self.data[0].size(0) / self.mbs_block.micro_batch_size )
            else:
                chunks = self.mbs_block.chunks

            idx = (self.call_counter-1) % chunks

            return [
                self.data[i][idx * self.mbs_block.micro_batch_size: (idx+1) * self.mbs_block.micro_batch_size]
                for i in range(self.num)
            ]

    def _update(self):
        self.call_counter += 1
        self.mbs_block.current_iter = self.call_counter
        if self.call_counter == 1:
            self.mbs_block.temp = 0
            self.mbs_block.mini_iter = 0
        self.mbs_block.update()


class _TestMBSMultiProcessingDataLoaderIter(_TestMBSBaseDataLoaderIter, _MultiProcessingDataLoaderIter):
    def __init__(self, loader: DataLoader, mbs_block) -> None:
        super().__init__(mbs_block, loader)


class _TestMBSSingleProcessDataLoaderIter(_TestMBSBaseDataLoaderIter, _SingleProcessDataLoaderIter):
    def __init__(self, loader: DataLoader, mbs_block) -> None:
        super().__init__(mbs_block, loader)


def _backend_iter(cls: _TestMBSBaseDataLoaderIter, queue: Queue):

    pass



class SimpleStreamingDataloader:
    def __init__(
        self,
        mbs_block,
        dataloader: DataLoader,
        micro_batch_size: int,
    ) -> None:
        self.mbs_block = mbs_block
        self.dataloader = dataloader

        self.size_dataset = self.dataloader.dataset.__len__()
        self.batch_size = self.dataloader.batch_size
        self.micro_batch = micro_batch_size

        self.chunks = math.ceil( self.batch_size / self.micro_batch )

    def __len__(self):
        return math.ceil( self.size_dataset / self.micro_batch )

    def mini_len(self):
        return math.ceil( self.size_dataset / self.batch_size )

    def __iter__(self):
        for _ in range( self.__len__() ):
            data = self.dataloader._get_iterator().__next__()
            num = len(data)
            chunks = self.chunks
            if self.micro_batch != data[-1].size(0):
                chunks = math.ceil( data[0].size(0) / self.micro_batch )

            for idx in range(chunks):
                self.mbs_block.zero_grad_timing = idx == 0
                self.mbs_block.update_timing = (idx + 1) == chunks
                self.mbs_block.remaining_step = chunks
                yield [ 
                    data[i][ idx * self.micro_batch : (idx+1) * self.micro_batch ]
                    for i in range(num)
                ]

