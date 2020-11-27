# Initialize the GAN and launch the training

import torch
import numpy as np
from GAN.model import Generator, Discriminator, weights_init_normal
from GAN.trainer import Trainer
from GAN.data_read import get_data_loader, get_data_loader2
import itertools


def main():

    # Load the data (DataLoader object)
    path_monnet = 'C:/Users/remys/GAN-Art-Monet/img'
    path_pictures = 'C:/Users/remys/GAN-Art-Monet/photo'
    batch_size = 1
    n_epochs = 10
    device = 'cpu'
    dataset = get_data_loader(path_monnet, path_pictures, batch_size)

    # Create Generators and Discriminators and put them on GPU/TPU
    generator_AB = Generator().to(device)
    generator_BA = Generator().to(device)

    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)

    generator_AB.apply(weights_init_normal)
    generator_BA.apply(weights_init_normal)
    discriminator_A.apply(weights_init_normal)
    discriminator_B.apply(weights_init_normal)

    # Set optimizers

    G_optimizer = torch.optim.Adam(itertools.chain(
        generator_AB.parameters(), generator_BA.parameters()), lr=2e-4)

    D_optimizer = torch.optim.Adam(itertools.chain(
        discriminator_A.parameters(), discriminator_B.parameters()), lr=2e-4)

    # Set trainer
    trainer = Trainer(
        generator_ab=generator_AB,
        generator_ba=generator_BA,
        discriminator_a=discriminator_A,
        discriminator_b=discriminator_B,
        generator_optimizer=G_optimizer,
        discriminator_optimizer=D_optimizer,
        n_epochs=n_epochs,
        dataloader=dataset,
        device=device,
    )

    # Launch Training
    trainer.train()


if __name__ == '__main__':
    main()
