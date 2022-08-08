from typing import Callable, List, Optional, Tuple
from torchvision.datasets.voc import VOCSegmentation
from torchvision.transforms.functional import to_tensor
import torch
from torch.utils.data.dataset import random_split, Dataset

from torchvision import transforms
from albumentations import HorizontalFlip, Compose, Resize, Normalize

from PIL import Image
import numpy as np
import os


class  CarvanaDataset(Dataset):
    def __init__(
        self,
        root: str = "dataset",
        image_transform: transforms = None,
        mask_transform: transforms = None
    ) -> None:
        super().__init__()
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_path = os.path.join( root, "train" )
        self.mask_path = os.path.join( root, "train_masks" )

        self.images = os.listdir( self.image_path )
        self.masks = os.listdir( self.mask_path )
        assert len(self.images) == len(self.masks), "The length does not equal!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_path, self.images[index])
        mask_path = os.path.join(self.mask_path, self.images[index])
        mask_path = mask_path.split(".")[0] + '_mask.gif'

        image = np.array( Image.open( image_path ).convert("RGB") )
        mask = np.array( Image.open( mask_path ).convert("L"), dtype=np.float32 )
        mask[mask == 255.0] = 1.0

        if self.image_transform is not None:
            image = self.image_transform( image )
        if self.mask_transform is not None:
            mask = self.mask_transform( mask )

        return image, mask


def  carvana_dataset(
    root: str,
    image_size: int,
    train_val_split: float = 0.2
):
    # Setup transforms for inputs and masks
    carvana_mean = [0.0, 0.0, 0.0]
    carvana_std = [1.0, 1.0, 1.0]

    image_shape = (image_size, image_size)

    image_compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(carvana_mean, carvana_std),
            transforms.Resize( image_shape )
        ])
    masks_compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize( image_shape )
        ])

    # Define dataset
    carvana_dataset = CarvanaDataset(
        root=root,
        image_transform=image_compose,
        mask_transform=masks_compose 
    )

    # Split dataset
    num_val     = int( len(carvana_dataset) * train_val_split )
    num_train   = len(carvana_dataset) - num_val
    train_dataset, val_dataset =  random_split(
        carvana_dataset,
        [num_train, num_val],
    )

    return train_dataset, val_dataset


d =    [[[0, 0, 0],      [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 0, 0],    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 128, 0],    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 128, 0],  [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 0, 128],    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 0, 128],  [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[0, 128, 128],  [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[128, 128, 128],[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 0, 0],     [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 0, 0],    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 128, 0],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 128, 0],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 0, 128],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 0, 128],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]],
        [[64, 128, 128], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]],
        [[192, 128, 128],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]],
        [[0, 64, 0],     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]],
        [[128, 64, 0],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]],
        [[0, 192, 0],    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
        [[128, 192, 0],  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]],
        [[0, 64, 128],   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]],
        [[224, 224, 192],[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]]


class VOCDataset(Dataset):
    def __init__(self, root: str, image_size: int, transform = None) -> None:
        super().__init__()
        self.files = open( 
                os.path.join(
                    root, 
                    "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
                ), 'r'
            ).read().split('\n')
        del self.files[-1]
        self.image_paths = os.path.join(root, "VOCdevkit/VOC2012/JPEGImages")
        self.mask_paths = os.path.join(root, "VOCdevkit/VOC2012/SegmentationClass")

        self.image_shape = (image_size, image_size)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.image_shape),
                transforms.ToTensor(),
                transforms.Normalize( 
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225] 
                ),
            ])
        self.mask_transforms = transforms.CenterCrop(self.image_shape)

    def __getitem__(self, index):
        image = Image.open( self.image_paths + "/" + self.files[index] + ".jpg" ).convert('RGB')
        mask = Image.open( self.mask_paths + "/" + self.files[index] + ".png" ).convert('RGB')

        # image = image.resize(self.image_shape)
        # mask = mask.resize(self.image_shape)

        image = self.transform( image )
        mask = self.mask_transforms( mask )
        s = np.asarray(mask).tolist()
        l  = []

        for itm in range(self.image_shape[0]):
            for item in range(self.image_shape[1]):
                for i, j in enumerate(d):
                    if s[itm][item] == j[0]:
                        l.append(j[1])

        l = torch.from_numpy(np.resize( l, ( self.image_shape[0], self.image_shape[1], 22 ) )).permute(2, 0, 1)
        return image, l

    def __len__(self):
        return len(self.files)


def get_vocdataset(
    image_size: int,
    root: Optional[str] = None,
    train_val_split: float = 0.2
):
    dataset = VOCDataset(
        root=root,
        image_size=image_size,
    )

    num_val     = int( len(dataset) * train_val_split )
    num_train   = len(dataset) - num_val
    train_dataset, val_dataset =  random_split(
        dataset,
        [num_train, num_val],
    )

    return train_dataset, val_dataset


LABELS = [
            [0, 0, 0],
            [128, 0, 0],        # aeroplane
            [0, 128, 0],        # bicycle
            [128, 128, 0],      # bird
            [0, 0, 128],        # boat
            [128, 0, 128],      # bottle
            [0, 128, 128],      # bus
            [128, 128, 128],    # car
            [64, 0, 0],         # cat
            [192, 0, 0],        # chair
            [64, 128, 0],       # cow
            [192, 128, 0],      # diningtable
            [64, 0, 128],       # dog
            [192, 0, 128],      # horse
            [64, 128, 128],     # motorbike
            [192, 128, 128],    # person
            [0, 64, 0],         # pottedplant
            [128, 64, 0],       # sheep
            [0, 192, 0],        # sofa
            [128, 192, 0],      # train
            [0, 64, 128],       # tvmonitor
            [224, 224, 192],    # 
]

LABELS_DICT = {
            'aeroplane':    [128, 0, 0],
            'bicycle' :     [0, 128, 0],
            'bird': [128, 128, 0],
            'boat': [0, 0, 128],
            'bottle': [128, 0, 128],
            'bus': [0, 128, 128],
            'car': [128, 128, 128],
            'cat': [64, 0, 0],
            'chair': [192, 0, 0],
            'cow': [64, 128, 0],
            'diningtable': [192, 128, 0],
            'dog': [64, 0, 128],
            'horse': [192, 0, 128],
            'motorbike': [64, 128, 128],
            'person': [192, 128, 128],
            'pottedplant': [0, 64, 0],
            'sheep': [128, 64, 0],
            'sofa': [0, 192, 0],
            'train': [128, 192, 0],
            'tvmonitor': [0, 64, 128],
}


class MultiLabelVOC(VOCSegmentation):
    def __init__( self, image_size: int = 224, **kwargs ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_shape = ( image_size, image_size )
        self.mask_transform = transforms.CenterCrop( self.image_shape )

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert('RGB')
        image= self.transform(image)

        masks = Image.open(self.masks[index]).convert('RGB')
        masks = self.mask_transform( masks )
        masks = self._masking(masks)

        return image, masks

    def _masking(self, masks):
        masks = np.asarray( masks )
        h, w, _ = masks.shape
        masks = masks.tolist()
        new_masks = [ np.zeros( self.image_shape ).tolist() for _ in range( len(LABELS) ) ]

        for ih in range(h):
            for iw in range(w):
                color = masks[ih][iw]
                for lidx, lcolor in enumerate(LABELS):
                    if color == lcolor:
                        new_masks[lidx][ih][iw]=1.
        return torch.from_numpy( np.asarray(new_masks) )


def get_voc_dataset( root: str, image_size: int, train: bool ):
    image_set = 'train' if train else 'val'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop( (image_size, image_size) ),
        transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
    ])

    dataset = MultiLabelVOC( 
        root=root,
        image_size=image_size,
        image_set=image_set,
        transform=transform
    )

    return dataset
