# Initialize the GAN and launch the training

import torch
import numpy as np
from GAN.model import Generator, Discriminator
from GAN.trainer import Trainer
from GAN.data_read import get_data_loader


def main():

    # Load the data (DataLoader object)
    path = 'C:/Users/remys/GAN-Art-Monet/img'
    batch_size = 8
    dataset = get_data_loader(path, batch_size)

    # Create Generators and Discriminators and put them on GPU/TPU
    generator_AB = Generator()
    generator_BA = Generator()

    discriminator_AB = Discriminator()
    discriminator_BA = Discriminator()

    # Set optimizers
    G_AB_optimizer = torch.optim.Adam(generator_AB.parameters(), lr=1e-4)
    G_BA_optimizer = torch.optim.Adam(generator_BA.parameters(), lr=1e-4)

    D_AB_optimizer = torch.optim.Adam(discriminator_AB.parameters(), lr=1e-4)
    D_BA_optimizer = torch.optim.Adam(discriminator_BA.parameters(), lr=1e-4)

    # Set trainer
    trainer = Trainer()

    # Launch Training
    trainer.train()


if __name__ == '__main__':
    main()
