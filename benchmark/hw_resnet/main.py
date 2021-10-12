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
                help="config file path (default: None)",
            ),
            args_template(
                ["-v", "--version"],
                type=int,
                default=50,
                help="config file path (default: None)",
            ),
            args_template(
                ["-w", "--wandb"],
                type=bool,
                default=False,
                help="config file path (default: None)",
            ),
            args_template(
                ["-i", "--image_size"],
                type=int,
                default=32,
                help="config file path (default: None)",
            ),
            args_template(
                ["-b", "--batch_size"],
                type=int,
                default=256,
                help="config file path (default: None)",
            ),
            args_template(
                ["-p", "--pin_memory"],
                type=bool,
                default=False,
                help="config file path (default: None)",
            ),
            args_template(
                ["-d", "--data_type"],
                type=str,
                default='cifar10',
                help="config file path (default: None)",
            ),
            args_template(
                ["--num_workers"],
                type=int,
                default=0,
                help="config file path (default: None)",
            ),
            args_template(
                ["--num_classes"],
                type=int,
                default=10,
                help="config file path (default: None)",
            ),
            args_template(
                ["-s", "--shuffle"],
                type=bool,
                default=False,
                help="config file path (default: None)",
            ),
            args_template(
                ["--mbs"],
                type=bool,
                default=False,
                help="config file path (default: None)",
            ),
            args_template(
                ["--micro_batch_size"],
                type=int,
                default=32,
                help="config file path (default: None)",
            ),
            args_template(
                ["--exp"],
                type=int,
                default=1,
                help="config file path (default: None)",
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
