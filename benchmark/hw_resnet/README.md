# The benchmark for ResNet

## ResNet Toml file
```
title = "ResNet"

# Model
[model]
version=50

# Setting Optimizer
[optimizer]
lr=0.01
mometum=0.9
decay=0.0005

# Device Related Configurations
[gpu]
device=1

# Train Related Configurations
[train]
epoch=100
seed=10000

# Dataset Related Configurations
[dataset]
  [dataset.train]
    type="flower102"
    path="./dataset/flower102/train"
    num_classes=102
    train_batch=1
    image_size=500
    num_workers=6
    shuffle=true
    pin_memory=true

  [dataset.test]
    type="flower102"
    path="./dataset/flower102/valid"
    num_classes=102
    test_batch=1
    image_size=500
    num_workers=6
    shuffle=true
    pin_memory=true

# Micro Batch Streaming
[mbs]
enable=false
micro_batch_size=1

# Monitoring tool
[wandb]
enable=false
```
- If you want to train ResNet models using MBS, change ```enablle``` arugment ```false``` to ```true```(line [46](./toml/flower.toml)).


## How to test
```
$ python main.py -c toml/flower.toml
```
- Before testing, please check if it is a ```toml``` file with the configuration you want.