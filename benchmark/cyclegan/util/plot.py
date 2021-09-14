import argparse
import json
from pathlib import PurePath

import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = [".json"]


def _plot_json(target_path, base_path, file_name):
    print(target_path, base_path)
    with open(target_path) as mbs:
        target_data = json.load(mbs)
    
    with open(base_path) as base:
        compare_data = json.load(base)

    True_train_loss = []
    True_val_loss = []
    True_top1 = []
    True_top5 = []

    False_train_loss = []
    False_val_loss = []
    False_top1 = []
    False_top5 = []

    True_max_top1 = 0
    True_max_top5 = 0
    False_max_top1 = 0
    False_max_top5 = 0

    for epoch in target_data:
        curr = target_data[epoch]
        True_train_loss.append(curr["train loss"])
        True_top1.append(curr["top-1"])
        True_top5.append(curr["top-5"])

        if True_max_top1 < curr['top-1']:
            True_max_top1 = curr['top-1']

        if True_max_top5 < curr['top-5']:
            True_max_top5 = curr['top-5']

    for epoch in compare_data:
        curr = compare_data[epoch]
        False_train_loss.append(curr["train loss"])
        False_top1.append(curr["top-1"])
        False_top5.append(curr["top-5"])

        if False_max_top1 < curr['top-1']:
            False_max_top1 = curr['top-1']

        if False_max_top5 < curr['top-5']:
            False_max_top5 = curr['top-5']


    plt.rcParams["figure.figsize"] = (10,20)

    plt.subplot(311)
    plt.plot(True_train_loss, label='mbs', color='b')
    plt.plot(False_train_loss, label='baseline', color='r')
    plt.ylabel("loss-value")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{file_name} train loss')

    plt.subplot(312)
    plt.plot(True_top1, label=f'mbs (max : {True_max_top1})', color='b')
    plt.plot(False_top1, label=f'baseline (max : {False_max_top1})', color='r')
    plt.ylabel("accuracy (%)")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{file_name} Top-1 accuracy')

    plt.subplot(313)
    plt.plot(True_top5, label=f'mbs (max : {True_max_top5})', color='b')
    plt.plot(False_top5, label=f'baseline (max : {False_max_top5})', color='r')
    plt.ylabel("accuracy (%)")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{file_name} Top-5 accuracy')

    plt.tight_layout()
    plt.savefig(f"{file_name}.png", dpi=600)


def _route_by_file_extension(file_path: str, base_path: str, file_name: str) -> None:
    file_suffix = PurePath(file_path).suffix
    base_suffix = PurePath(base_path).suffix
    if file_suffix == ".json" and base_suffix == '.json':
        _plot_json(file_path, base_path, file_name)
    else:
        raise TypeError(
            "File is not supported, Should be {}".format(
                ", ".join(SUPPORTED_EXTENSIONS)
            )
        )


def plot(target_path: str, base_path: str, file_name: str) -> None:
    _route_by_file_extension(target_path, base_path, file_name)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot given data to graph")
    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        default=None,
        help="Configuration for mode, should be 'train' or 'test' Default: train",
    )
    parser.add_argument(
        "-b",
        "--base_path",
        type=str,
        default=None,
        help="Configuration for mode, should be 'train' or 'test' Default: train",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Configuration for mode, should be 'train' or 'test' Default: train",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = _cli()
    target_path = args.target_path
    base_path = args.base_path
    file_name = args.name
    plot(target_path, base_path, file_name)


if __name__ == "__main__":
    main()
