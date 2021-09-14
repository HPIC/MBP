from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

import dataset

def get_network(args, device):
    from model import unet_1156

def get_training_dataloader(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], batch_size=16, num_workers=6, shuffle=True, pin_memory=True):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    carvana_training = dataset.CarvanaTrain(root='./data', train=True, image_transform=image_transform, mask_transform=mask_transform)
    carvana_training_loader = DataLoader(carvana_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return carvana_training_loader


def get_test_dataloader(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], batch_size=16, num_workers=6, shuffle=True, pin_memory=True):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    carvana_testing = dataset.CarvanaTest(root='./data', train=False, image_transform=image_transform)
    carvana_testing_loader = DataLoader(carvana_testing, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return carvana_testing_loader



if __name__ == '__main__':
    carvana_test = get_test_dataloader()
    carvana_train = get_training_dataloader()

    i = 0
    for x,y in carvana_train:
        if i == 5: break
        print(x.shape,y.shape)
        i+=1