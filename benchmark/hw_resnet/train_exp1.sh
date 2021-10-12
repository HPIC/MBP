# ResNet-50 with baseline False
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# # ResNet-50 with mbs False
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 50 \
# -i 32 -p False -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# # ResNet-50 with baseline true
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 256 --micro_batch_size 64 \
# -w True

# # ResNet-50 with mbs true
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 50 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

#####################################################################

# ResNet-152 with baseline False
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 152 \
# -i 32 -d cifar10 --num_classes 10 \
# --num_workers 0 \
# -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 152 \
# -i 32 -d cifar10 --num_classes 10 \
# --num_workers 0 \
# -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000 -v 152 \
-i 32 -d cifar10 --num_classes 10 \
--num_workers 0 \
-b 128 --micro_batch_size 32 \
-w True

# # ResNet-152 with mbs False
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 152 \
# -i 32 -d cifar10 --num_classes 10 \
# --num_workers 0 \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 152 \
# -i 32 -d cifar10 --num_classes 10 \
# --num_workers 0 \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000 -v 152 \
-i 32 -d cifar10 --num_classes 10 \
--num_workers 0 \
--mbs True -b 128 --micro_batch_size 32 \
-w True

# ResNet-152 with baseline true
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs False -b 128 --micro_batch_size 32 \
# -w True

# ResNet-152 with mbs true
# python main.py -m train -c toml/cifar10.toml \
# -r 1000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 100000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 50000000 -v 152 \
# -i 32 -p True -d cifar10 --num_classes 10 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

#####################################################################

# ResNet-50 with baseline False
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 50 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0 \
# -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 50 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0   \
# -b 256 --micro_batch_size 64 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 50 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0   \
-b 256 --micro_batch_size 64 \
-w True

# ResNet-50 with mbs False
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 50 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0   \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 50 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0   \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 50 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0   \
--mbs True -b 256 --micro_batch_size 64 \
-w True

# ResNet-50 with baseline true
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 50 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 50 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 256 --micro_batch_size 64 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 50 \
-i 32 -p True -d cifar100 --num_classes 100 \
--num_workers 0  \
-b 256 --micro_batch_size 64 \
-w True

# ResNet-50 with mbs true
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 50 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 50 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 256 --micro_batch_size 64 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 50 \
-i 32 -p True -d cifar100 --num_classes 100 \
--num_workers 0  \
--mbs True -b 256 --micro_batch_size 64 \
-w True

#####################################################################

# ResNet-152 with baseline False
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 152 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 152 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 152 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0  \
-b 128 --micro_batch_size 32 \
-w True

# ResNet-152 with mbs False
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 152 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 152 \
# -i 32 -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 152 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0  \
--mbs True -b 128 --micro_batch_size 32 \
-w True

# ResNet-152 with baseline true
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 152 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 152 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 152 \
-i 32 -p True -d cifar100 --num_classes 100 \
--num_workers 0  \
-b 128 --micro_batch_size 32 \
-w True

# ResNet-152 with mbs true
# python main.py -m train -c toml/cifar100.toml \
# -r 1000 -v 152 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

# python main.py -m train -c toml/cifar100.toml \
# -r 100000 -v 152 \
# -i 32 -p True -d cifar100 --num_classes 100 \
# --num_workers 0  \
# --mbs True -b 128 --micro_batch_size 32 \
# -w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 152 \
-i 32 -p True -d cifar100 --num_classes 100 \
--num_workers 0  \
--mbs True -b 128 --micro_batch_size 32 \
-w True

#####################################################################