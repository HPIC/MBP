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
