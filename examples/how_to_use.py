import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet50_Weights, resnet50

import mbp

if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
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
        batch_size=512,
        shuffle=True,
    )

    @mbp.apply(["image", "label"], ub_size=64, device=device)
    def train_fn(model, criterion, image, label):
        output = model(image)
        loss = criterion(output, label)
        return loss, output

    epochs = 10
    for e in range(epochs):
        for image, label in train_loader:
            optimizer.zero_grad()
            loss, *_ = train_fn(model, criterion, image=image, label=label)
            optimizer.step()
            print(f"[{e}/{epochs}] loss: {loss}")
