python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--exp 31 \
-w True

####################### MBS #######################
python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 31 \
-w True

####################### MBS-BN #######################
python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 50 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 5000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 500000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True

python main.py -m train -c toml/cifar10.toml \
-r 50000000000 -v 152 \
-d cifar10 --num_classes 10 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 31 \
-w True
