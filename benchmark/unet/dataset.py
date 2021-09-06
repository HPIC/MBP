from skimage.io import imread
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(
        self, inputs: list, targets: list, image_transform=None, mask_transform=None
    ):
        self.inputs = inputs
        self.targets = targets
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(input_ID), imread(target_ID)

        # Preprocessing
        if self.image_transform is not None:
            x = self.image_transform(x)
        if self.mask_transform is not None:
            y = self.mask_transform(y)

        return x, y
