# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 16 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 16 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 16 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 8

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 16 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 8

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 32 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 8

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 32 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 8

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb 

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb 

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 32 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 32 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb \
#     -mbs -usize 4

#####################################################################

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 2

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 4



python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 8

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 256 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 16



python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 64

python train.py \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 512 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 128


#####################################################################


python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 4 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 2

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 4



python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 8

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 256 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 16



python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 64

python train.py \
    -r 5000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 512 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 128


#####################################################################


python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 2 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 4



python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 16 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 8

python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 256 \
    -scale 4 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 16



python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb 

python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 128 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 64

python train.py \
    -r 500000000 \
    -unet \
    -gpu \
    -gpu_device 0 \
    -b 512 \
    -scale 8 \
    -warm 0 \
    -lr 1e-3 \
    -wandb \
    -mbs -usize 128

# python train.py \
#     -r 5000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb 

# python train.py \
#     -r 500000000 \
#     -gpu \
#     -gpu_device 0 \
#     -b 8 \
#     -i 224 -patch_size 32 -ncls 102 \
#     -warm 0 \
#     -lr 1e-3 \
#     -wandb 