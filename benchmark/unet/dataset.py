from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
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
        mask_path = os.path.join(self.mask_path, self.masks[index])

        image = np.array( Image.open( image_path ).convert("RGB") )
        mask = np.array( Image.open( mask_path ).convert("L"), dtype=np.float32 )
        mask[mask == 255.0] = 1.0

        if self.image_transform is not None:
            image = self.image_transform( image )
        if self.mask_transform is not None:
            mask = self.mask_transform( mask )

        return image, mask


class CarvanaTrain(Dataset):
    def __init__(
        self, root: str, train: bool=True, image_transform=None, mask_transform=None
    ):
        
        self.root = root
        self.train = train
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.input_path = os.path.join(self.root, "train")
        self.target_path = os.path.join(self.root, 'train_masks')
        self.inputs = os.listdir(self.input_path)
        self.targets = os.listdir(self.target_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        image = imread(os.path.join(self.input_path, input_ID))
        mask = imread(os.path.join(self.target_path, target_ID))

        # Preprocessing
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class CarvanaTest(Dataset):
    def __init__(
        self, root: str, train: bool=True, image_transform=None, mask_transform=None
    ):
        
        self.root = root
        self.train = train
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.input_path = os.path.join(self.root, "train")
        self.inputs = os.listdir(self.input_path)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]

        # Load input and target
        image = imread(os.path.join(self.input_path, input_ID))

        # Preprocessing
        if self.image_transform is not None:
            image = self.image_transform(image)

        return image