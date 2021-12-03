import argparse
import torch
import torch.nn as nn

from models.resnet import (
    resnet50,
    resnet152
)
from models.vgg import select_model
from flowers_classifier.utils_ic import load_data
from flowers_classifier.model_ic import NN_Classifier

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("-d", "--data_dir", help="load data directory")
parser.add_argument("-a", "--arch", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int, default=8)
parser.add_argument("-i", "--image_size", type=int, default=32)
parser.add_argument("--bn", type=bool, default=False)
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--epochs", type=int, default=100, help="set epochs")
parser.add_argument("--gpu", type=int, default=0, help="use gpu")
parser.add_argument("--wandb", type=bool, default=False)

args = parser.parse_args()

print(f"data_dir: {args.data_dir}")
print(f"learning_rate: {args.learning_rate}")
print(f"epochs: {args.epochs}")
print(f"gpu: {args.gpu}")
print(f"wandb: {args.wandb}")
print(f"batch size: {args.batch_size}")
print(f"image_size: {args.image_size}")
print(f"batchNorm?: {args.bn}")

# load dataloader
trainloader, testloader, validloader, train_data = load_data(args.data_dir)
num_classes = len(train_data.classes)
print(f"num classes : {train_data.classes}, {num_classes}")

# Define Model
model = None
if args.arch == 50:
    model = resnet50( num_classes )
elif args.arch == 152:
    model = resnet152( num_classes )
else:
    model = select_model( args.bn, args.arch, num_classes )
model = model.to(args.gpu)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam( model.parameters(), lr = args.learning_rate )

for epoch in range(args.epochs):
    # Trian
    running_loss = 0
    model.train()
    for idx, (images, labels) in enumerate(trainloader):
        print(f'[{idx+1}/{len(trainloader)}]', end='\r')
        images: torch.Tensor = images.to(args.gpu)
        labels: torch.Tensor = labels.to(args.gpu)

        optimizer.zero_grad()
        output: torch.Tensor = model.forward( images )
        output: torch.Tensor = nn.LogSoftmax(dim=1)(output)
        loss: torch.Tensor = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Test
    accuracy = 0
    test_loss = 0
    model.eval()
    for images, labels in testloader:
        images: torch.Tensor = images.to(args.gpu)
        labels: torch.Tensor = labels.to(args.gpu)

        output: torch.Tensor = model.forward( images )
        output: torch.Tensor = nn.LogSoftmax(dim=1)(output)
        loss: torch.Tensor = criterion(output, labels)

        test_loss += loss.item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type( torch.FloatTensor ).mean()

    print(
        f"[{epoch+1}/{args.epochs}]",
        " *train loss: {:.4f}".format(running_loss/len(trainloader)),
        " *test loss: {:4f}".format(test_loss/len(testloader)),
        " *test accuracy: {:.4f}".format(accuracy/len(testloader))
    )

