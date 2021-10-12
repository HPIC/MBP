python main.py -m train -c toml/cifar100_base.toml -r 1000  -v 50 -w True
python main.py -m train -c toml/cifar100_base.toml -r 10000 -v 50 -w True
python main.py -m train -c toml/cifar100_base.toml -r 50000 -v 50 -w True

python main.py -m train -c toml/cifar100_base.toml -r 1000  -v 152 -w True
python main.py -m train -c toml/cifar100_base.toml -r 10000 -v 152 -w True
python main.py -m train -c toml/cifar100_base.toml -r 50000 -v 152 -w True

python main.py -m train -c toml/cifar100_mbs.toml -r 1000  -v 50 -w True
python main.py -m train -c toml/cifar100_mbs.toml -r 10000 -v 50 -w True
python main.py -m train -c toml/cifar100_mbs.toml -r 50000 -v 50 -w True

python main.py -m train -c toml/cifar100_mbs.toml -r 1000  -v 152 -w True
python main.py -m train -c toml/cifar100_mbs.toml -r 10000 -v 152 -w True
python main.py -m train -c toml/cifar100_mbs.toml -r 50000 -v 152 -w True

