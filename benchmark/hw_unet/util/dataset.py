from typing import Callable, Optional
from torchvision.datasets.voc import VOCSegmentation
from torchvision.transforms.functional import to_tensor
import torch

from albumentations import HorizontalFlip, Compose, Resize, Normalize

from PIL import Image
import numpy as np


class SegmentationVOC(VOCSegmentation): 
    def __getitem__(self, index): 
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None: 
            augmented = self.transforms(image=np.array(img), mask=np.array(target)) 
            img = augmented['image'] 
            target = augmented['mask'] 
            target[target>20] = 0 

        img = to_tensor(img) 
        target = torch.from_numpy(target).type(torch.long) 
        return img, target


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
h = 520
w = 520


def get_dataset(
    root: str,
    train: bool = True,
    download: bool = False,
    transform: Optional[Callable] = None,
):
    if transform == None:
        if train:
            transform = Compose([
                Resize( height=h, width=w ),
                HorizontalFlip(p=0.5),
                Normalize( mean=mean, std=std ),
            ])
        else:
            transform = Compose([
                Resize( height=h, width=w ),
                Normalize( mean=mean, std=std ),
            ])
    image_set = "train" if train else "val"

    return SegmentationVOC(
        root=root,
        image_set=image_set,
        download=download,
        transforms=transform,
    )


if __name__ == "__main__":
    traindataset = get_dataset("../dataset/voc2012", train=True)
    testdataset = get_dataset("../dataset/voc2012", train=False)