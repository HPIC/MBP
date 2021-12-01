###################### MBS #######################
# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 50 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 128 \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 50 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 32 \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 50 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 16 \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 152 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 128 \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 152 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 32 \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 152 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 16 \
# --exp 33 \
# -w True

# ####################### MBS-BN #######################
# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 50 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 128 --bn True \
# --exp 33 \
# -w True

# python main.py -m train -c toml/cifar10.toml \
# -r 5000000 -v 50 \
# -d cifar10 --num_classes 10 \
# -b 256 \
# --mbs True --micro_batch_size 32 --bn True \
# --exp 33 \
# -w True

python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 16 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 128 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 32 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 16 --bn True \
--exp 33 \
-w True


###################### MBS #######################
python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 128 \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 32 \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 16 \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 128 \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 32 \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 16 \
--exp 33 \
-w True

####################### MBS-BN #######################
python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 128 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 32 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 16 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 128 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 32 --bn True \
--exp 33 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 16 --bn True \
--exp 33 \
-w True
