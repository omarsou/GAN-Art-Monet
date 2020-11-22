# Initialize the GAN and launch the training

import torch
import numpy as np
from GAN.model import Generator, Discriminator
from GAN.trainer import Trainer
from GAN.data_read import get_data_loader


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

    # Set optimizers
    G_AB_optimizer = torch.optim.Adam(generator_AB.parameters(), lr=2e-4)
    G_BA_optimizer = torch.optim.Adam(generator_BA.parameters(), lr=2e-4)

    D_A_optimizer = torch.optim.Adam(discriminator_A.parameters(), lr=2e-4)
    D_B_optimizer = torch.optim.Adam(discriminator_B.parameters(), lr=2e-4)

    # Set trainer
    trainer = Trainer(
        generator_ab=generator_AB,
        generator_ba=generator_BA,
        discriminator_a=discriminator_A,
        discriminator_b=discriminator_B,
        generator_ab_optimizer=G_AB_optimizer,
        generator_ba_optimizer=G_BA_optimizer,
        discriminator_a_optimizer=D_A_optimizer,
        discriminator_b_optimizer=D_B_optimizer,
        n_epochs=n_epochs,
        dataloader=dataset,
        device=device,
    )

    # Launch Training
    trainer.train()


if __name__ == '__main__':
    main()
