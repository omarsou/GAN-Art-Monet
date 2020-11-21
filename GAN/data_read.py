# File where we implement the pytorch dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torchvision import transforms

# Function that returns the DataLoader for the training


def get_data_loader(path, batch_size, shuffle=True):

    # Transform images as tensors
    transformation = transforms.ToTensor()

    # Image folder dataset
    ds = dset.ImageFolder(
        root=path,
        transform=transformation
    )

    # DataLoader class from pytorch
    dl = DataLoader(
        dataset=path,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dl
