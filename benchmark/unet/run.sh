python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 2 \
    -image_factor 0.5 \
    -warm 0 \
    -lr 1e-4 
    # -wandb
