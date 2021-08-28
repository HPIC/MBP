from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

def get_dataset(imagenet_path, image_size, batch_size):
    imagenet_dataset = ImageNet(
        imagenet_path,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        unaligned=True,
    )
    imagenet_dataloader = DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    return imagenet_dataloader

