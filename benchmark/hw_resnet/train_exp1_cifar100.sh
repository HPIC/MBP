python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--exp 21 \
-w True

####################### MBS #######################
python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 \
--exp 21 \
-w True

####################### MBS-BN #######################
python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 50 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 5000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 500000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True

python main.py -m train -c toml/cifar100.toml \
-r 50000000000 -v 152 \
-d cifar100 --num_classes 100 \
-b 256 \
--mbs True --micro_batch_size 64 --bn True \
--exp 21 \
-w True
