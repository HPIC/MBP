import copy, random
import sys
sys.path.append('../')

from mbs.micro_batch_streaming import MicroBatchStreaming

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

'''
    Test code to detect NaN (Why does NaN occur?).
    ---
    - 
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
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
        pin_memory=True
    )

    return train_dataloader, train_datasets, test_dataloader, test_datasets


if __name__ ==  '__main__':
    random_seed = 42

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(random_seed)

    batch_size = 64
    train_dataloader, train_datasets, test_dataloader, test_datasets = get_dataloader(batch_size=batch_size)

    print('dataset : Cifar100')

    dev0 = torch.device('cuda:0')
    dev1 = torch.device('cuda:1')

    baseline_model = Net()
    mbs_model = copy.deepcopy(baseline_model)
    baseline_model = baseline_model.to(dev0)
    mbs_model = mbs_model.to(dev1)

    criterion_0 = nn.CrossEntropyLoss().to(dev0)
    criterion_1 = nn.CrossEntropyLoss().to(dev1)

    optimizer0 = optim.SGD(
        baseline_model.parameters(),
        lr = 0.0002,
        momentum=0.9
    )
    optimizer1 = optim.SGD(
        mbs_model.parameters(),
        lr = 0.0002,
        momentum=0.9
    )

    mbs = MicroBatchStreaming()
    mbs_train_dataloader = mbs.set_dataloader(train_dataloader, micro_batch_size=16)
    mbs_test_dataloader = mbs.set_dataloader(test_dataloader, micro_batch_size=16)

    # for (name, base_mod), (_, mbs_mod) in zip(baseline_model.named_parameters(), mbs_model.named_parameters()):
    #     print(
    #         f'---- {name} ---- \n',
    #         base_mod.grad.data, base_mod.grad.data.size(), '\n',
    #         mbs_mod.grad.data, mbs_mod.grad.data.size()
    #     )

    # print('\n\n\n\n\n\n')

    baseline_model.zero_grad()
    for idx, (image, label) in enumerate(train_dataloader):
        image = image.to(dev0)
        label = label.to(dev0)

        output = baseline_model(image)
        loss = criterion_0(output, label)
        loss.backward(create_graph=False)

        if idx == 0:
            print(loss.item())
            break

    accu_loss = None
    accu_grad = []
    mbs_model.zero_grad()
    for idx, (image, label) in enumerate(mbs_train_dataloader):
        image = image.to(dev1)
        label = label.to(dev1)

        output = mbs_model(image)
        loss : torch.Tensor = criterion_1(output, label)
        loss.backward(create_graph=True)
        # accu_grad.append([])
        # for para in mbs_model.parameters():
        #     accu_grad[-1].append( para.grad.data )

        if ( idx == (batch_size/16 - 1) ):
            # for a, b, c, d in zip(accu_grad[0], accu_grad[1], accu_grad[2], accu_grad[3]):
            #     avg = (a + b + c + d) / 4
            #     print(avg)
            break

    for (name, base_mod), (_, mbs_mod) in zip(baseline_model.named_parameters(), mbs_model.named_parameters()):
        print(
            f'---- {name} ---- \n',
            # base_mod.data,'\n',
            # mbs_mod.data, '\n',
            base_mod.grad.data, base_mod.grad.data.size(), '\n',
            mbs_mod.grad.data, mbs_mod.grad.data.size(),
        #     f'parameter equal? {torch.equal( base_mod.data, mbs_mod.data.to(dev0) )} \n',
        #     f'grad equal? {torch.equal( base_mod.grad.data, mbs_mod.grad.data.to(dev0) )}',
        )

