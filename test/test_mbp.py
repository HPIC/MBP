import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet50_Weights, resnet50

import mbp


@contextmanager
def measure_time(method_name: str, runtimes: list):
    stime = time.perf_counter()
    yield
    etime = time.perf_counter()
    runtime = etime - stime
    runtimes.append(runtime)
    print(f"[{method_name}] Time: {runtime:.1f} (sec)")


def get_arguments():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--num_class", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--method", type=str, default="default", help="default, mbp")
    parser.add_argument("-u", "--micro_batch", type=int, default=8)
    parser.add_argument("--dp", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = torch.device(f"cuda:{args.device}")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = (
        nn.DataParallel(model, output_device=dev) if args.dp else model
    )  # TODO: Check if this is correct
    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Define train function with mbp.autobackward()
    @mbp.autobackward(
        ["input", "target"], micro_batch_size=args.micro_batch, device_=dev
    )
    def train_fn(model, criterion, input, target):
        output = model(input)
        loss = criterion(output, target)
        return loss

    avg_runtime = []
    if args.method == "mbp":
        with measure_time(args.method, avg_runtime):
            for epoch in range(args.epochs):
                input, target = torch.randn(
                    args.batch_size, 3, args.image_size, args.image_size
                ), torch.randint(0, args.num_class, (args.batch_size,))
                optimizer.zero_grad()
                loss = train_fn(model, criterion, input=input, target=target)
                optimizer.step()
    else:
        with measure_time(args.method, avg_runtime):
            for epoch in range(args.epochs):
                input, target = torch.randn(
                    args.batch_size, 3, args.image_size, args.image_size
                ).to(dev), torch.randint(0, args.num_class, (args.batch_size,)).to(dev)
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
