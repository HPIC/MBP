# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

from mbs.wrap_dataloader import MBSDataloader
from mbs.wrap_loss import MBSLoss
from mbs.wrap_optimizer import MBSOptimizer
import os
from pickle import FALSE
import sys
import argparse
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.optim as optim
import torchvision
from torchvision.datasets import cifar
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from mbs.micro_batch_streaming import MicroBatchStreaming
import random

# def _init_microbatch_stream(
#     train_dataloader: DataLoader, valid_dataloader: DataLoader, optim_list: List[optim.Optimizer], micro_batch_size: int
# ) -> List[optim.Optimizer]:
#     # Define MicroBatchStreaming
#     mbs = MicroBatchStreaming()
#     train_dataloader = mbs.set_dataloader(train_dataloader, micro_batch_size)
#     valid_dataloader = mbs.set_dataloader(valid_dataloader, micro_batch_size)
#     for optim in optim_list:
#         optim = mbs.set_optimizer(optim)
#     return train_dataloader, valid_dataloader, optim_list

def _init_microbatch_stream(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optim: optim.Optimizer,
    loss_fn: Module,
    micro_batch_size: int
) -> List[optim.Optimizer]:
    # Define MicroBatchStreaming
    mbs = MicroBatchStreaming()
    train_dataloader = mbs.set_dataloader(train_dataloader, micro_batch_size)
    valid_dataloader = valid_dataloader
    loss_fn = mbs.set_loss(loss_fn)
    optim = mbs.set_optimizer(optim)
    return train_dataloader, valid_dataloader, optim, loss_fn


def train(epoch):

    start = time.time()
    net.train()
    loss_accum = 0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda(device)
            images = images.cuda(device)

        mbs_optimizer.zero_grad()
        outputs = net(images)
        loss : torch.Tensor = mbs_loss(outputs, labels)
        loss.backward()
        mbs_optimizer.step()

        n_iter = (epoch - 1) * cifar100_training_loader.micro_len() + batch_index + 1
        loss_accum += loss.item()
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        
        if (batch_index + 1) % (batch_size // micro_batch_size) == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss_accum,
                mbs_optimizer.optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * micro_batch_size + len(images),
                total_samples=cifar100_training_loader_dataset_size
            ))
            loss_accum = 0

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda(device)
            labels = labels.cuda(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(device), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / cifar100_test_loader_dataset_size,
        correct.float() / cifar100_test_loader_dataset_size,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / cifar100_test_loader_dataset_size, epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / cifar100_test_loader_dataset_size, epoch)

    return correct.float() / cifar100_test_loader_dataset_size

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='select gpu device')
    parser.add_argument('-mbs', action='store_true', default=False, help='use mbs framework or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-mb', type=int, default=64, help='micro batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-seed', type=int, default=42, help='use seed as torch random seed')
    parser.add_argument('-stepLR', action='store_true', default=False, help='use stepLR')
    args = parser.parse_args()

    random_seed = args.seed

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(random_seed)



    _device = 'cuda:' + str(args.gpu_device)
    device = torch.device(_device)

    net = get_network(args, device)
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=6,
        batch_size=args.b,
        shuffle=True,
        pin_memory=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=6,
        batch_size=args.b,
        shuffle=True,
        pin_memory=True
    )

    cifar100_training_loader_dataset_size = len(cifar100_training_loader.dataset)
    cifar100_test_loader_dataset_size = len(cifar100_test_loader.dataset)

    batch_size = args.b
    loss_function = nn.CrossEntropyLoss().cuda(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    #add mbs capability
    micro_batch_size = args.mb
    cifar100_training_loader, cifar100_test_loader, mbs_optimizer, mbs_loss = _init_microbatch_stream(
        cifar100_training_loader, cifar100_test_loader, optimizer, loss_function, micro_batch_size)

    cifar100_training_loader : MBSDataloader
    mbs_optimizer : MBSOptimizer
    mbs_loss : MBSLoss

    if args.stepLR:
        train_scheduler = optim.lr_scheduler.MultiStepLR(mbs_optimizer.optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = cifar100_training_loader_dataset_size/micro_batch_size
    warmup_scheduler = WarmUpLR(mbs_optimizer.optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR_MBS):
        os.mkdir(settings.LOG_DIR_MBS)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR_MBS, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda(device)
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if args.stepLR:
            if epoch > args.warm:
                train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()