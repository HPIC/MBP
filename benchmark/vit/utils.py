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

import termplotlib as tpl
import plotext as plx


'''
    Add new code
    ---
'''
def save_parameters(
    model: Module, path: str = 'checkpoint/vit/test.pth'
):
    torch.save( model.state_dict(), path )


def load_parameters(
    model: Module, path: str = 'checkpoint/vit/test.pth'
):
    model.load_state_dict(torch.load(path))
    return model
