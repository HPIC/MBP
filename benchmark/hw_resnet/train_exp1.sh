python main.py -m train -c toml/cifar10.toml \
-r 50000000 -v 50 \
-i 32 -d cifar10 --num_classes 10 \
--num_workers 0  \
--mbs True -b 256 --micro_batch_size 64 \
--exp 1 \
-w True --bn True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 50 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0  \
--mbs True -b 256 --micro_batch_size 64 \
--exp 1 \
-w True --bn True

python main.py -m train -c toml/cifar10.toml \
-r 50000000 -v 152 \
-i 32 -d cifar10 --num_classes 10 \
--num_workers 0  \
--mbs True -b 128 --micro_batch_size 32 \
--exp 1 \
-w True --bn True

python main.py -m train -c toml/cifar100.toml \
-r 50000000 -v 152 \
-i 32 -d cifar100 --num_classes 100 \
--num_workers 0  \
--mbs True -b 128 --micro_batch_size 32 \
--exp 1 \
-w True --bn True
