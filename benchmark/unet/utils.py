from typing import List
import torch

import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler

import numpy
import os
import dataset
import datetime
import re
import math


'''
    Add new code
    ---
'''
def dice_loss(pred: Tensor, mask: Tensor, smooth: int = 1e-5):
    bce_output = F.binary_cross_entropy_with_logits(
        pred, mask, reduction="sum"
    )
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + mask.sum(dim=(2,3))
    
    # dice coefficient
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    
    # dice loss
    dice_loss = 1.0 - dice
    
    # total loss
    loss: Tensor = bce_output + dice_loss
    return loss.sum(), dice.sum() * 100

class DiceLoss(Module):
    def __init__(self, weight=None, size_average=True) -> None:
        super().__init__()
    
    def forward(self, inputs: Tensor, masks: Tensor, smooth=1):
        inputs = F.sigmoid( inputs )
        inputs = inputs.view(-1)
        masks = masks.view(-1)

        intersection = ( inputs * masks ).sum()
        dice = ( 2. * intersection + smooth )/(inputs.sum() + masks.sum() + smooth)
        dice_loss = 1 - dice
        bce = F.binary_cross_entropy(inputs, masks, reduction="mean")
        loss = bce + dice_loss

        return loss, dice_loss * 100


def get_network_name(name: str):
    if name == 'unet1156':
        from model import unet_1156
        return unet_1156()

    if name == 'unet3156':
        from model import unet_3156
        return unet_3156()


def get_dataset(
    mean: List[float], std: List[float], 
    scale: float = 1.0
):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize( ( int(1280 * scale), int(1918 * scale) ) )
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( ( int(1280 * scale), int(1918 * scale) ) )
    ])

    return dataset.CarvanaDataset(
            root='data',
            image_transform=image_transform, 
            mask_transform=mask_transform
        )


'''
    Original Code
    ---
'''

def get_network(args):
    if args.net == 'unet1156':
        from model import unet_1156
        return unet_1156()

    if args.net == 'unet3156':
        from model import unet_3156
        return unet_3156()


def get_training_dataloader(
    mean: float, std: float, 
    batch_size: int =16, num_workers: int =6, 
    shuffle: bool =True, pin_memory: bool =True, 
    scale: float = 1.0
):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize( ( int(1280 * scale), int(1918 * scale) ) )
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize( ( int(1280 * scale), int(1918 * scale) ) )
    ])

    carvana_training = dataset.CarvanaTrain(root='./data', train=True, image_transform=image_transform, mask_transform=mask_transform)
    carvana_training_loader = DataLoader(carvana_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return carvana_training_loader


def get_test_dataloader(
    mean: float, std: float, 
    batch_size: int =16, num_workers: int =6, 
    shuffle: bool =True, pin_memory: bool =True, 
    scale: float = 1.0
):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize( ( int(1280 * scale), int(1918 * scale) ) )
    ])

    carvana_testing = dataset.CarvanaTest(root='./data', train=False, image_transform=image_transform)
    carvana_testing_loader = DataLoader(carvana_testing, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return carvana_testing_loader

def compute_mean_std():
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    tmp_dataset = dataset.CarvanaTrain(root='./data', train=True, image_transform=None, mask_transform=None)
    data_r = numpy.dstack([tmp_dataset[i][:, :, 0] for i in range(math.trunc(len(tmp_dataset)/4))])
    r_mean = numpy.mean(data_r)
    r_std = numpy.std(data_r)
    print("R is done \n r_mean =", r_mean, "r_std =", r_std)

    data_g = numpy.dstack([tmp_dataset[i][:, :, 1] for i in range(math.trunc(len(tmp_dataset)/4))])
    r_mean = numpy.mean(data_r)
    g_mean = numpy.mean(data_g)
    g_std = numpy.std(data_g)
    print("G is done \n g_mean =", g_mean, "g_std =", g_std)

    data_b = numpy.dstack([tmp_dataset[i][:, :, 2] for i in range(math.trunc(len(tmp_dataset)/4))])
    r_mean = numpy.mean(data_r)
    b_mean = numpy.mean(data_b)
    b_std = numpy.std(data_b)
    print("B is done \n r_mean =", b_mean, "r_std =", b_std)
    mean = r_mean, g_mean, b_mean
    std = r_std, g_std, b_std

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

# if __name__ == '__main__':
#     mean, std = compute_mean_std()
#     print(mean, std)