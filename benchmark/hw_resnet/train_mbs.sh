# python main.py -m train -c toml/cifar10_mbs_bn.toml -r 50000 -v 50 -w True
# python main.py -m train -c toml/cifar100_mbs_bn.toml -r 50000 -v 50 -w True


# python main.py -m train -c toml/cifar10_base.toml -r 60000 -v 152 -w True
# python main.py -m train -c toml/cifar100_base.toml -r 60000 -v 152 -w True
python main.py -m train -c toml/cifar10_mbs_bn.toml -r 60000 -v 50 -w True
python main.py -m train -c toml/cifar100_mbs_bn.toml -r 60000 -v 50 -w True
python main.py -m train -c toml/cifar10_mbs_bn.toml -r 60000 -v 152 -w True
python main.py -m train -c toml/cifar100_mbs_bn.toml -r 60000 -v 152 -w True