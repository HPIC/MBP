python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 32 \
    -image_factor 2 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 2 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 64 \
    -image_factor 4 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 2 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -image_factor 8 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 2 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 256 \
    -image_factor 16 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 2 \
    -wandb




python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 32 \
    -image_factor 2 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 1 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 64 \
    -image_factor 4 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 1 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -image_factor 8 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 1 \
    -wandb

python train.py \
    -net unet3156 \
    -gpu \
    -gpu_device 0 \
    -b 256 \
    -image_factor 16 \
    -warm 0 \
    -lr 1e-2 \
    -mbs \
    -usize 1 \
    -wandb