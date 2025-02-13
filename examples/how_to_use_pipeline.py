from contextlib import contextmanager
from time import perf_counter_ns

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet152_Weights, resnet152

import mbp
from mbp._pipeline import apply_pipeline


@contextmanager
def runtime():
    start = perf_counter_ns()
    yield
    torch.cuda.synchronize()
    end = perf_counter_ns()
    print(f"Runtime: {(end-start) * 1e-6:.1f} ms")


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model = apply_pipeline(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    train_loader = DataLoader(
        CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=False,
    )

    @mbp.apply(["image", "label"], ub_size=16, device=device)
    def train_fn(model, criterion, image, label):
        output = model(image)
        loss = criterion(output, label)
        return loss, output

    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        with runtime():
            loss, *_ = train_fn(model, criterion, image=image, label=label)
        optimizer.step()
        print(f"loss: {loss}")
        if i == 4:
            break
    print("Done!")

    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    train_loader = DataLoader(
        CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            ),
        ),
        batch_size=64,
        shuffle=False,
    )

    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        with runtime():
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
        optimizer.step()
        print(f"loss: {loss}")
        if i == 4:
            break
    print("Done!")
