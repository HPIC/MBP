import torch

import sys
import math
import time
sys.path.append('../')

from benchmark.cyclegan.dataloader import cyclegan_dataset
from mbs.micro_batch_streaming import MicroBatchStreaming

def test_grad_accumulate():
    mini_batch_size = 128
    batch_size = 4
    total_load_time = 0
    mini_batch_load_time = 0
    temp_time = 0
    counter = 0

    dataloader = cyclegan_dataset('../benchmark/cyclegan/dataset/horse2zebra/train', image_size=256, batch_size=batch_size)

    start_time = time.perf_counter()
    for idx, data in enumerate(dataloader):
        data0 = data['A'].to(torch.device('cuda:0'))
        data1 = data['B'].to(torch.device('cuda:0'))
        end_time = time.perf_counter()

        # calculate time.
        load_time = (end_time - start_time)
        total_load_time += load_time
        temp_time += load_time
        if (idx+1) % (mini_batch_size/batch_size) == 0 or (idx+1) == len(dataloader):
            mini_batch_load_time += temp_time
            temp_time = 0
            counter += 1
        print(f'[{idx+1}] {load_time}')

        start_time = time.perf_counter()

    print(
        f'just accumulate gradients :',
        f'mini_batch_load_time = {mini_batch_load_time / counter}',
        f'load time = {total_load_time/len(dataloader)}'
    )

def test_micro_batch_streaming():
    batch_size = 128
    micro_batch_size = 4
    total_load_time = 0
    mini_batch_load_time = 0
    temp_time = 0
    counter = 0
    mbs = MicroBatchStreaming(micro_batch_size=micro_batch_size)
    dataloader = cyclegan_dataset('../benchmark/cyclegan/dataset/horse2zebra/train', image_size=256, batch_size=batch_size)
    dataloader = mbs.set_dataloader(dataloader)

    start_time = time.perf_counter()
    for idx, (ze, up, (data0, data1, _, _)) in enumerate(dataloader):
        data0 = data0.to(torch.device('cuda:0'))
        data1 = data1.to(torch.device('cuda:0'))
        end_time = time.perf_counter()

        # calculate time
        load_time = (end_time - start_time)
        total_load_time += load_time
        temp_time += load_time
        if (idx+1) % (batch_size/micro_batch_size) == 0 or (idx+1) == 332:
            mini_batch_load_time += temp_time
            temp_time = 0
            counter += 1
        print(f'[{idx+1}] {load_time}')

        start_time = time.perf_counter()

    print(
        f'mbs :',
        f'mini_batch_load_time = {mini_batch_load_time/counter}',
        f'load time = {total_load_time/332}'
    )

if __name__ == '__main__':
    test_grad_accumulate()
    test_micro_batch_streaming()

