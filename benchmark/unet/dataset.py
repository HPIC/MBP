from skimage.io import imread
from torch.utils.data import Dataset
import os
from pathlib import Path

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
        image, mask = imread(os.path.join(self.input_path, input_ID)), imread(os.path.join(self.target_path, target_ID))

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