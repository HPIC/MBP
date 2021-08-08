from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torch.distributed as dist

import glob
import os
from PIL import Image
from random import Random

def cyclegan_dataset(path, image_size=256, batch_size=4):
    dataset = ImageDataset( dataset_path = path,
                            transform = transforms.Compose([
                                        transforms.Resize(image_size, Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                            unaligned = True )
    dataloader = DataLoader( dataset, batch_size=batch_size, shuffle=True, pin_memory=True )
    return dataloader

class ImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None, unaligned=False):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(f"{dataset_path}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(f"{dataset_path}/B") + "/*.*"))

    def __getitem__(self, index):
        A_image     = self.files_A[index % len(self.files_A)]
        item_A      = self.transform(Image.open(A_image))
        item_A_name = A_image.split('/')
        item_A_name = item_A_name[-1]

        B_image     = self.files_B[index % len(self.files_B)]
        item_B      = self.transform(Image.open(B_image))
        item_B_name = B_image.split('/')
        item_B_name = item_B_name[-1]

        return {"A": item_A, "B": item_B, "A_name": item_A_name, "B_name": item_B_name}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


