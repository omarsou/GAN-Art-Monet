# File where we build the cycle GAN
import torch
import torch.nn as nn


class Generator(nn.Module):

    # Initialize the Generator
    def __init__(self, params):
        super(Generator, self).__init__()

    # Forward Pass
    def forward(self, input_sample):
        pass


class Discriminator(nn.Module):

    # Initialize the Discriminator
    def __init__(self, params):
        super(Discriminator, self).__init__()

    # Forward pass
    def forward(self, input_sample):
        pass
