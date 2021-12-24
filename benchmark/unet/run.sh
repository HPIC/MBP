python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -image_factor 2 \
    -warm 0 \
    -lr 1e-2 \
    -wandb
