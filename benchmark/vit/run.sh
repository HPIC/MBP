python train.py \
    -gpu \
    -gpu_device 0 \
    -b 32 \
    -i 32 -patch_size 32 -ncls 10 \
    -warm 0 \
    -lr 1e-3 \
    # -wandb \
    # -mbs -usize 64
