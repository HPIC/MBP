from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(
        self, root: str, image_transform=None, mask_transform=None
    ) -> None:
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
        mask_path = os.path.join( self.mask_path, self.images[index].replace(".jpg", "_mask.gif") )
        # mask_path = mask_path.split(".")[0] + '_mask.gif'

        image = np.array( Image.open( image_path ).convert("RGB") )
        mask = np.array( Image.open( mask_path ).convert("L"), dtype=np.float32 )
        mask[mask == 255.0] = 1.0

        if self.image_transform is not None:
            image = self.image_transform( image )
        if self.mask_transform is not None:
            mask = self.mask_transform( mask )

        return image, mask


class FlowerDataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        super().__init__()
        self.path = path
        self.files = []
        for (dirpath, _, filenames) in os.walk(self.path):
            for f in filenames:
                if f.endswith('.jpg'):
                    p = {}
                    p['image_path'] = dirpath + '/' + f
                    self.files.append(p)
        if transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path: str = self.files[idx]['image_path']
        image_label = img_path.split('/')[-2]
        image = np.array( Image.open( img_path ).convert('RGB') )
        image = self.transform(image)

        return image, image_label

