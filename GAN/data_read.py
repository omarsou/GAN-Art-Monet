# File where we implement the pytorch dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torchvision import transforms
import os
from PIL import Image
import numpy as np

# Function that returns the DataLoader for the training

# Dataset returning a picture and a monet painting


class Monnet_Pics(Dataset):
    def __init__(self, path_monnet, path_pictures, transform):
        self.path_monnet = path_monnet
        self.path_pictures = path_pictures
        self.transform = transform
        self.list_monnet = os.listdir(self.path_monnet)
        self.list_pictures = os.listdir(self.path_pictures)
        self.nb_pictures = len(self.list_pictures)

    def __len__(self):
        return len(self.list_monnet)

    def __getitem__(self, index):

        img_path = os.path.join(self.path_monnet, self.list_monnet[index])
        image = Image.open(img_path).convert("RGB")
        pytorch_tensor_monnet = self.transform(image)

        random_index = np.random.randint(self.nb_pictures)

        img_path = os.path.join(
            self.path_pictures, self.list_pictures[random_index])
        image = Image.open(img_path).convert("RGB")
        pytorch_tensor_pictures = self.transform(image)

        return [(pytorch_tensor_monnet-0.5)/0.5, (pytorch_tensor_pictures-0.5)/0.5]


class Pics_Monnet(Dataset):
    def __init__(self, path_monnet, path_pictures, transform):
        self.path_monnet = path_monnet
        self.path_pictures = path_pictures
        self.transform = transform
        self.list_monnet = os.listdir(self.path_monnet)
        self.list_pictures = os.listdir(self.path_pictures)
        self.nb_pictures = len(self.list_pictures)

    def __len__(self):
        return len(self.list_pictures)

    def __getitem__(self, index):

        random_index = np.random.randint(len(self.list_monnet))

        img_path = os.path.join(
            self.path_monnet, self.list_monnet[random_index])
        image = Image.open(img_path).convert("RGB")
        pytorch_tensor_monnet = self.transform(image)

        img_path = os.path.join(
            self.path_pictures, self.list_pictures[index])
        image = Image.open(img_path).convert("RGB")
        pytorch_tensor_pictures = self.transform(image)

        return [(pytorch_tensor_monnet-0.5)/0.5, (pytorch_tensor_pictures-0.5)/0.5]


def get_data_loader(path_monnet, path_pictures, batch_size, shuffle=True):

    # Transform images as tensors
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    # Image folder dataset
    ds = Monnet_Pics(
        path_monnet=path_monnet,
        path_pictures=path_pictures,
        transform=transformation
    )

    # DataLoader class from pytorch
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dl


def get_data_loader2(path_monnet, path_pictures, batch_size, shuffle=True):

    # Transform images as tensors
    transformation = transforms.Compose([
        transforms.ToTensor()
    ])

    # Image folder dataset
    ds = Pics_Monnet(
        path_monnet=path_monnet,
        path_pictures=path_pictures,
        transform=transformation
    )

    # DataLoader class from pytorch
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dl
