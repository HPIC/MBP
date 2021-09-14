import argparse
import torch

from dataloader import get_dataset
from models.vgg.vgg import select_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Test our framework with given configurations"
        )
    parser.add_argument( '--file', type=str, default=None, help="Saved model parameter" )
    parser.add_argument( '-m', '--model', type=str, default=None, help="Saved model parameter" )
    parser.add_argument( '-d', '--data', type=str, default=None, help="Saved model parameter" )
    args = parser.parse_args()

    dataloader, _ = get_dataset(
        path=f'./dataset/{args.data}',
        dataset_type=args.data,
        batch_size=32,
        image_size=32,
        is_train=False
    )
    device = torch.device('cuda:0')
    model = select_model(
        normbatch=False,
        version=11,
        num_classes=100
    ).to(device)

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    with torch.no_grad():
        for input, label in dataloader:
            input = input.to(device)
            label = label.to(device)
            output : torch.Tensor = model(input)

            # rank 1
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct_top1 += (pred == label).sum().item()

            # rank 5
            _, rank5 = output.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct5 = rank5.eq(label.view(1, -1).expand_as(rank5))

            for k in range(6):
                correct_k = correct5[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_top5 += correct_k.item()

            print(
                f'top-1 : {100 * ( correct_top1 / total )}, top-5 : {100 * ( correct_top5 / total )}',
            )
