import argparse
import json
from pathlib import PurePath

import matplotlib.pyplot as plt

SUPPORTED_EXTENSIONS = [".json"]


def _plot_json(args):
    print(f"target path : {args.target_path}, name : {args.target_name}")
    print(f"base path : {args.base_path}, name : {args.base_name}")

    with open(args.target_path) as mbs:
        target_data = json.load(mbs)
    
    with open(args.base_path) as base:
        compare_data = json.load(base)

    target_train_loss = []
    target_train_magnitude = None
    target_top1 = []
    target_top5 = []

    base_train_loss = []
    base_train_magnitude = None
    base_top1 = []
    base_top5 = []

    target_max_top1 = 0
    target_max_top5 = 0
    base_max_top1 = 0
    base_max_top5 = 0

    for epoch in target_data:
        curr = target_data[epoch]
        target_train_loss.append(curr["train loss"])
        target_top1.append(curr["top-1"])
        target_top5.append(curr["top-5"])
        # True_train_magnitude = curr['magnitude']

        if target_max_top1 < curr['top-1']:
            target_max_top1 = curr['top-1']

        if target_max_top5 < curr['top-5']:
            target_max_top5 = curr['top-5']

    for epoch in compare_data:
        curr = compare_data[epoch]
        base_train_loss.append(curr["train loss"])
        base_top1.append(curr["top-1"])
        base_top5.append(curr["top-5"])
        # False_train_magnitude = curr['magnitude']

        if base_max_top1 < curr['top-1']:
            base_max_top1 = curr['top-1']

        if base_max_top5 < curr['top-5']:
            base_max_top5 = curr['top-5']


    plt.rcParams["figure.figsize"] = (4,8)

    plt.subplot(311)
    plt.plot(target_train_loss, label=f'{args.target_name}', color='b')
    plt.plot(base_train_loss, label=f'{args.base_name}', color='r')
    plt.ylabel("loss-value")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{args.name} train loss')

<<<<<<< HEAD
    plt.subplot(312)
=======
    plt.subplot(321)
>>>>>>> f2306ef8eb7bc9a9a0af65693227c56d651e6043
    plt.plot(target_top1,
        label='{name} (max : {num:.2f})'.format(name=args.target_name, num=target_max_top1 ),
        color='b'
    )
    plt.plot(base_top1,
        label='{name} (max : {num:.2f})'.format(name=args.base_name, num=base_max_top1 ),
        color='r'
    )
    plt.ylabel("accuracy (%)")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{args.name} Top-1 accuracy')

<<<<<<< HEAD
    plt.subplot(313)
=======
    plt.subplot(331)
>>>>>>> f2306ef8eb7bc9a9a0af65693227c56d651e6043
    plt.plot(target_top5,
        label='{name} (max : {num:.2f})'.format(name=args.target_name, num=target_max_top5 ),
        color='b'
    )
    plt.plot(base_top5,
        label='{name} (max : {num:.2f})'.format(name=args.base_name, num=base_max_top5 ),
        color='r'
    )
    plt.ylabel("accuracy (%)")
    plt.xlabel("epoch")
    plt.xlim([0, int(epoch)])
    plt.legend()
    plt.title(f'{args.name} Top-5 accuracy')

    plt.tight_layout()
    plt.savefig(f"{args.name}.png", dpi=200)



def _route_by_file_extension(args) -> None:
    target_path = args.target_path
    base_path = args.base_path
    file_suffix = PurePath(target_path).suffix
    base_suffix = PurePath(base_path).suffix

    if file_suffix == ".json" and base_suffix == '.json':
        _plot_json(args)
    else:
        raise TypeError(
            "File is not supported, Should be {}".format(
                ", ".join(SUPPORTED_EXTENSIONS)
            )
        )


# def plot(target_path: str, base_path: str, file_name: str) -> None:
def plot(args) -> None:
    _route_by_file_extension(args)


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
    parser.add_argument("--target_name", type=str, default=None)
    parser.add_argument("--base_name", type=str, default=None)

    args = parser.parse_args()
    return args


def main() -> None:
    args = _cli()
    plot(args)


if __name__ == "__main__":
    main()
