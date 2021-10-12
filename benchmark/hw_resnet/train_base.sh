# python main.py -m train -c toml/cifar10_base.toml -r 1000  -v 50 -w True
# python main.py -m train -c toml/cifar10_base.toml -r 100000 -v 50 -w True
# python main.py -m train -c toml/cifar10_base.toml -r 50000000 -v 50 -w True

python main.py -m train -c toml/cifar10_base.toml -r 1000  -v 152 -w True
python main.py -m train -c toml/cifar10_base.toml -r 100000 -v 152 -w True
python main.py -m train -c toml/cifar10_base.toml -r 50000000 -v 152 -w True

# python main.py -m train -c toml/cifar10_mbs.toml -r 1000  -v 50 -w True
# python main.py -m train -c toml/cifar10_mbs.toml -r 100000 -v 50 -w True
# python main.py -m train -c toml/cifar10_mbs.toml -r 50000000 -v 50 -w True

python main.py -m train -c toml/cifar10_mbs.toml -r 1000  -v 152 -w True
python main.py -m train -c toml/cifar10_mbs.toml -r 100000 -v 152 -w True
python main.py -m train -c toml/cifar10_mbs.toml -r 50000000 -v 152 -w True

