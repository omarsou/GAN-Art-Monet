# File where we implement the pytorch dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from torchvision import transforms
import os
from PIL import Image

# Function that returns the DataLoader for the training


class Monnet(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.list_img = os.listdir(self.path)

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.list_img[index])
        image = Image.open(img_path).convert("RGB")
        pytorch_tensor = self.transform(image)
        return pytorch_tensor


def get_data_loader(path, batch_size, shuffle=True):

    # Transform images as tensors
    transformation = transforms.ToTensor()

    # Image folder dataset
    ds = Monnet(
        path=path,
        transform=transformation
    )

    print(len(ds))

    # DataLoader class from pytorch
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dl
