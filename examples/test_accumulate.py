import copy, random, argparse, json

import torch
from torch.nn.modules.batchnorm import BatchNorm1d
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import sys
sys.path.append('..')
from mbs.micro_batch_streaming import MicroBatchStreaming

'''
    Test code to detect NaN (Why does NaN occur?).
    ---
    - 
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.conv3 = nn.Conv2d(6, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(6, 16, 5)
        self.conv5 = nn.Conv2d(16, 16, 5)
        self.conv6 = nn.Conv2d(16, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)


    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))

        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataloader(batch_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_datasets = datasets.CIFAR100(
            root='../../common_dataset/cifar100',
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    train_dataloader = DataLoader(
        train_datasets,
        batch_size=batch_size,
        # shuffle=True,
        pin_memory=True
    )

    test_datasets = datasets.CIFAR100(
            root='../../common_dataset/cifar100',
            train=False,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize
                ]
            )
    )
    test_dataloader = DataLoader(
        test_datasets,
        batch_size=batch_size,
        # shuffle=True,
        pin_memory=True
    )

    return train_dataloader, test_dataloader


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Accumulate test')
    parser.add_argument('-r', '--random-seed', type=int, default=42)
    parser.add_argument('-m', '--mode', type=int, default=0, help='0 is baseline, 1 is mbs-based')

    args = parser.parse_args()

    random_seed = args.random_seed

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(random_seed)

    batch_size = 64
    parameter_data = {}
    grad_data = {}
    mode = None

    print('dataset : Cifar100')

    if args.mode == 0: # base
        mode = 'baseline'
        train_dataloader, test_dataloader = get_dataloader(batch_size=batch_size)
        dev = torch.device('cuda:0')
        baseline_model = Net().to(dev)
        criterion_base = nn.CrossEntropyLoss().to(dev)
        optimizer_base = optim.SGD(
            baseline_model.parameters(),
            lr = 0.01,
            momentum=0.9
        )

        # baseline_model.train()
        for idx, (image, label) in enumerate(train_dataloader):
            optimizer_base.zero_grad()

            image : torch.Tensor = image.to(dev)
            label : torch.Tensor = label.to(dev)

            output : torch.Tensor = baseline_model(image)
            loss : torch.Tensor = criterion_base(output, label)
            loss.backward()

            # print(idx+1)
            # for name, para in baseline_model.named_parameters():
            #     print(name, '*'*30)
            #     print(para.data)
            # print("\n\n\n\n", '='*20, '\n\n\n\n')

            for name, buf in baseline_model.named_buffers():
                if 'running' in name:
                    print(name, '\n', buf)

            print("\n\n\n")

            optimizer_base.step()

            if idx + 1 == 20:
                # for name, para in baseline_model.named_parameters():
                #     if 'bn' in name:
                #         print(name, '*'*30)
                #         print(para.data)
                # print("\n\n\n")
                # for name, buf in baseline_model.named_buffers():
                #     print(name, '\n', buf)
                break
    elif args.mode == 1: # mbs-like
        mode = 'mbs-like'
        mbs_train_dataloader, test_dataloader = get_dataloader(batch_size=16)
        dev = torch.device('cuda:1')
        mbs_model = Net().to(dev)
        criterion_mbs = nn.CrossEntropyLoss().to(dev)
        optimizer_mbs = optim.SGD(
            mbs_model.parameters(),
            lr = 0.01,
            momentum=0.9
        )

        count = 0
        optimizer_mbs.zero_grad()
        for idx, (image, label) in enumerate(mbs_train_dataloader):
            image : torch.Tensor = image.to(dev)
            label : torch.Tensor = label.to(dev)

            output : torch.Tensor = mbs_model(image)
            loss : torch.Tensor = criterion_mbs(output, label) / (batch_size//16)
            loss.backward()

            if ( (idx + 1) % (batch_size // 16) == 0 ):
                count += 1
                for name, buf in mbs_model.named_buffers():
                    if 'running' in name:
                        print(name, '\n', buf)
                print("\n\n\n")
                optimizer_mbs.step()
                optimizer_mbs.zero_grad()
            if count == 20:
                for name, para in mbs_model.named_parameters():
                    if 'bn' in name:
                        print(name, '*'*30)
                        print(para.data)
                print("\n\n\n")
                for name, buf in mbs_model.named_buffers():
                    if 'running' in name:
                        print(name, '\n', buf)
                break
    elif args.mode == 2: # mbs
        mode = 'mbs'
        mbs_train_dataloader, test_dataloader = get_dataloader(batch_size=batch_size)
        dev = torch.device('cuda:1')

        criterion_mbs = nn.CrossEntropyLoss().to(dev)

        mbs = MicroBatchStreaming()
        mbs_model = mbs.set_batch_norm( Net() ).to(dev)
        optimizer_mbs = optim.SGD(
            mbs_model.parameters(),
            lr = 0.01,
            momentum=0.9
        )

        mbs_train_dataloader = mbs.set_dataloader(mbs_train_dataloader, micro_batch_size=16)
        criterion_mbs = mbs.set_loss(criterion_mbs)
        optimizer_mbs = mbs.set_optimizer(optimizer_mbs)

        count = 0

        # mbs_model.train()
        for idx, (image, label) in enumerate(mbs_train_dataloader):
            optimizer_mbs.zero_grad()

            image : torch.Tensor = image.to(dev)
            label : torch.Tensor = label.to(dev)

            output : torch.Tensor = mbs_model(image)
            loss : torch.Tensor = criterion_mbs(output, label)
            loss.backward()

            optimizer_mbs.step()

            if ( (idx + 1) % (batch_size // 16) == 0 ):
                count += 1
                # print(count)
                # for name, para in mbs_model.named_parameters():
                #     print(name, '*'*30)
                #     print(para.data)
                # print("\n\n\n\n", '='*20, '\n\n\n\n')
                for name, buf in mbs_model.named_buffers():
                    if 'running' in name:
                        print(name, '\n', buf)
                print("\n\n\n")
            if count == 20:
                # for name, para in mbs_model.named_parameters():
                #     if 'bn' in name:
                #         print(name, '*'*30)
                #         print(para.data)
                # print("\n\n\n")
                for name, buf in mbs_model.named_buffers():
                    if name in ['running_mean', 'running_var']:
                        print(name, '\n', buf)
                break

    # with open(f"{mode}_para.json", "w") as file:
    #     json.dump(parameter_data, file, indent=4)
    # with open(f"{mode}_grad.json", "w") as file:
    #     json.dump(grad_data, file, indent=4)
