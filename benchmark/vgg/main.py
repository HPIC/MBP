import argparse
from collections import namedtuple

from trainer import train
from util.config_parser import ConfigParser


class Runner:
    def __init__(self) -> None:
        pass

    @classmethod
    def _parse_args(cls) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Test our framework with given configurations"
        )

        args_template = namedtuple("PredefinedArgs", "flags type default help")
        predefined_args = [
            args_template(
                ["-m", "--mode"],
                type=str,
                default="train",
                help="Configuration for mode, should be 'train' or 'test' Default: train",
            ),
            args_template(
                ["-c", "--config_path"],
                type=str,
                default=None,
                help="config file path (default: None)",
            ),
            args_template(
                ["-r", "--random_seed"],
                type=int,
                default=100,
                help="Seed value for training (default: 100)",
            ),
            args_template(
                ["-v", "--version"],
                type=int,
                default=16,
                help="Model version (default: 16 in VGG)",
            ),
            args_template(
                ["-d", "--data_type"],
                type=str,
                default='cifar10',
                help="What is dataset type? (default: CIFAR-10) [ 'cifar10', 'cifar100' ]",
            ),
            args_template(
                ["--num_classes"],
                type=int,
                default=10,
                help="Number of class in dataset (default: 10)",
            ),
            args_template(
                ["-i", "--image_size"],
                type=int,
                default=32,
                help="Set up image size for training (default: 32)",
            ),
            args_template(
                ["-b", "--batch_size"],
                type=int,
                default=256,
                help="Set up (mini) batch size for training (default: 256)",
            ),
            args_template(
                ["-p", "--pin_memory"],
                type=bool,
                default=False,
                help="Use Pinned memory? (default: False)",
            ),
            args_template(
                ["--num_workers"],
                type=int,
                default=0,
                help="Set up number of workers for loading data from dataset. (default: 0)",
            ),
            args_template(
                ["-s", "--shuffle"],
                type=bool,
                default=False,
                help="Do you want to load data randomly? (default: False)",
            ),
            args_template(
                ["--mbs"],
                type=bool,
                default=False,
                help="Use Micro-Batch Streaming method for training (default: False)",
            ),
            args_template(
                ["--micro_batch_size"],
                type=int,
                default=32,
                help="Size of micro-batch that is a unit of streaming/computing in GPU (default: 32)",
            ),
            args_template(
                ["--bn"],
                type=bool,
                default=False,
                help="Do you want to consider BN layer when using MBS? (default: False)",
            ),
            args_template(
                ["-w", "--wandb"],
                type=bool,
                default=False,
                help="Use W&B tool? (default: False)",
            ),
            args_template(
                ["--exp"],
                type=int,
                default=1,
                help="Set up experiment number when using W&B tool (default: 1)",
            ),
        ]

        for arg in predefined_args:
            parser.add_argument(
                *arg.flags, type=arg.type, default=arg.default, help=arg.help
            )

        args = parser.parse_args()
        return args

    @classmethod
    def _cli(cls) -> None:
        args = cls._parse_args()
        mode = args.mode
        config = ConfigParser.parse(args.config_path)
        cls._router(mode, config, args)

    @classmethod
    def _router(cls, mode: str, config: dict, args) -> None:
        if mode == "train":
            train(config, args)
        elif mode == "test":
            pass

    def run(self) -> None:
        self._cli()


def main() -> None:
    runner = Runner()
    runner.run()


if __name__ == "__main__":
    main()
