#!/bin/bash
seed = 3481630172
net = ('vgg19', 'densenet121', 'googlenet', 'inceptionv3', 'xception', 'resnet101', 'preactresnet101', 'squeezenet', 'mobilenetv2', )
# need warmup list
# preactresnet
# 256 boom list
# inceptionv4 inceptionresnetv2

python train-naive.py -net resnet101 -gpu -gpu_device 0 -seed ${seed}
python train-naive.py -net preactresnet152 -gpu -gpu_device 0 -seed 3481630172
python train-naive.py -net resnet101 -gpu -gpu_device 1 -seed 3481630172

for 