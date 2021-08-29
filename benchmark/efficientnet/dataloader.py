from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop

def get_dataset(path, image_size, batch_size):
    imagenet_dataset = ImageFolder(
        path,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    )
    imagenet_dataloader = DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return imagenet_dataloader

