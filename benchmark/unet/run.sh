python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 1 \
    -image_factor 1 \
    -warm 0 \
    -lr 1e-2 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 1 \
    -image_factor 1 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 1 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 1 \
    -image_factor 1 \
    -warm 0 \
    -lr 1e-2 \
    -mbs_bn \
    -usize 1 \
    -wandb

