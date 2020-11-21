# Initialize the GAN and launch the training

import torch
import numpy as np
from GAN.model import Generator, Discriminator
from GAN.trainer import Trainer
from GAN.data_read import DataLoader
