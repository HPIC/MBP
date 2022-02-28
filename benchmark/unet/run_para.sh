python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -image_factor 4 \
    -warm 0 \
    -lr 1e-2 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -image_factor 4 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 2 \
    -wandb