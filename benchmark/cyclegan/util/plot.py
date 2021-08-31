import argparse
import json
from pathlib import PurePath

import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = [".json"]


def _plot_json(target_file: str):
    with open(target_file) as json_file:
        json_data = json.load(json_file)

        g_loss = []
        for index in json_data:
            curr = json_data[index]
            g_loss.append(curr["loss"])

        plt.plot(g_loss)
        plt.legend(["loss"])
        plt.ylabel("loss-value")
        plt.xlabel("epoch")
        plt.xlim([0, len(g_loss)])

        plt.savefig(f"{target_file}.png", dpi=600)


def _route_by_file_extension(file_path: str) -> None:
    file_suffix = PurePath(file_path).suffix
    if file_suffix == ".json":
        _plot_json(file_path)
    else:
        raise TypeError(
            "File is not supported, Should be {}".format(
                ", ".join(SUPPORTED_EXTENSIONS)
            )
        )


def plot(file_path: str) -> None:
    _route_by_file_extension(file_path)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot given data to graph")
    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        default=None,
        help="Configuration for mode, should be 'train' or 'test' Default: train",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = _cli()
    target_path = args.target_path
    plot(target_path)


if __name__ == "__main__":
    main()
